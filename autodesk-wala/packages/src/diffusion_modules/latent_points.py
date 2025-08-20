import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from timm.models.layers import DropPath

import numpy as np


# obtained from https://github.com/1zb/3DShape2VecSet/blob/master/models_class_cond.py#L34


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.0

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, t, context=None):
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x
        x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=context))) + x
        x = self.drop_path3(self.ls3(self.ff(self.norm3(x, t)))) + x
        return x


class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        t_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # ###
        # self.pos_emb = nn.Embedding(512, inner_dim)
        # ###

    def forward(self, x, t, cond=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)

        # ###
        # x = x + self.pos_emb.weight[None]
        # ###

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)

        x = self.proj_out(x)
        return x
