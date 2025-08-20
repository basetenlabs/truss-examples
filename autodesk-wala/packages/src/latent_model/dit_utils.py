# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Code based on https://github.com/facebookresearch/DiT/blob/main/models.py 


import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from typing import Final
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
def use_fused_attn():
    # This function should return whether to use fused attention based on your requirements.
    return True


# rmsnorm


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()
        # print(self.fused_attn)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        Bc, Nc, Cc = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(context)
            .reshape(Bc, Nc, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, context_dim, num_heads, mlp_ratio=4.0, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.cross_attn = CrossAttention(
            hidden_size,
            context_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs,
        )

    def forward(self, x, c, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + self.cross_attn(x, context)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, context


class DiTBlock_Basic(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, context


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        grid_size=12,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.flatten = flatten
        self.new_grid_size = (
            grid_size // patch_size,
            grid_size // patch_size,
            grid_size // patch_size,
        )
        self.num_patches = (
            self.new_grid_size[0] * self.new_grid_size[1] * self.new_grid_size[2]
        )

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, L = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWL -> BNC
        x = self.norm(x)
        return x


class Global_Proj_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, activation=nn.SiLU()):
        super().__init__()
        self.proj_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            activation,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=True),
            activation,
            nn.Linear(output_dim, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.proj_layer(x)
        x = torch.mean(x, 1)
        x = self.mlp(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=12,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        context_dim=None,
        add_condition_time_ch=None,
        with_self_att=True,
        block_type="dit",
        add_num_register=0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size

        self.x_embedder = PatchEmbed3D(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches

        self.context_dim = context_dim
        self.add_condition_time_ch = add_condition_time_ch
        if self.context_dim is not None:
            if block_type in ["cross_dit"]:
                self.context_mlp = nn.Sequential(
                    nn.Linear(context_dim, hidden_size, bias=True),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                )
            else:
                self.context_mlp = nn.Identity()

            if add_condition_time_ch is not None:
                self.global_proj = Global_Proj_Layer(
                    context_dim,
                    hidden_size,
                    hidden_dim=hidden_size,
                    activation=nn.SiLU(),
                )

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        self.add_num_register = add_num_register
        if self.add_num_register > 0:
            self.register_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, self.add_num_register, hidden_size)),
                0.0,
                0.01,
            )

        if block_type == "cross_dit":
            self.blocks = nn.ModuleList(
                [
                    Cross_DiTBlock(
                        hidden_size, hidden_size, num_heads, mlp_ratio=mlp_ratio
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(depth)
                ]
            )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.new_grid_size[0]
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        h = w = l = self.input_size // p
        x = x.reshape(shape=(x.shape[0], h, w, l, p, p, p, c))
        x = torch.einsum("nhwlpqrc->nchpwqlr", x)
        # print(x.shape)
        out = x.reshape(shape=(x.shape[0], c, h * p, w * p, l * p))
        # print(out.shape)
        return out

    def forward(self, x, t, latent_codes=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        if self.context_dim is not None:
            latent_codes_local = self.context_mlp(latent_codes)
        else:
            latent_codes_local = None

        if self.add_condition_time_ch and self.context_dim is not None:
            global_context = self.global_proj(latent_codes)
            t = t + global_context

        if self.add_num_register > 0:
            # print(x.shape)
            x = torch.cat(
                [x, self.register_emb.repeat(x.size(0), 1, 1).type(x.type())], dim=1
            )
            # print(x.shape)

        for block in self.blocks:
            x, latent_codes_local = block(x, t, context=latent_codes_local)  # (N, T, D)

        if self.add_num_register > 0:
            # print(x.shape)
            x = x[:, : -self.add_num_register, :]
            # print(x.shape)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid depth, height and width
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_l)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use one third of dimensions to encode grid_l, grid_h, grid_w
    emb_l = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (L*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (L*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (L*H*W, D/3)

    emb = np.concatenate([emb_l, emb_h, emb_w], axis=1)  # (L*H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}


if __name__ == "__main__":
    model = DiT(
        input_size=12,
        patch_size=3,
        in_channels=4,
        hidden_size=768,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        learn_sigma=False,
        context_dim=128,
        block_type="cross_dit",
        add_condition_time_ch=True,
        add_num_register=4,
    )
    # blank_tokens = torch.ones((5, args.grid_size*args.grid_size*args.grid_size)).to(torch.int64)
    x = torch.randn([5, 4, 12, 12, 12])
    condition = torch.randn([5, 256, 128])
    t = torch.randn([5])
    output = model(x, t, latent_codes=condition)
    print(model)
    print(output.shape)
