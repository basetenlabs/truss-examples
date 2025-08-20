# https://github.com/juho-lee/set_transformer/blob/master/modules.py
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class MAB(nn.Module):
    fused_attn: bool

    def __init__(
        self,
        dim_Q,
        dim_K,
        dim_V,
        num_heads,
        qkv_bias=False,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        ln=False,
        norm_layer=nn.LayerNorm,
    ):
        super(MAB, self).__init__()
        assert dim_V % num_heads == 0, "dim_V must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_V // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = True

        # Linear layers for query, key, value
        self.q = nn.Linear(dim_Q, dim_V, bias=qkv_bias)
        self.kv = nn.Linear(dim_K, dim_V * 2, bias=qkv_bias)

        # Normalization layers
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # Dropout layers for attention and projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_V, dim_V)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional layer normalization
        self.ln1 = norm_layer(dim_V) if ln else nn.Identity()

    def forward(self, Q, K):
        B, N, _ = Q.shape
        Bk, Nk, _ = K.shape

        # Generate queries
        q = self.q(Q).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Generate keys and values
        kv = (
            self.kv(K)
            .reshape(Bk, Nk, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # Split into key and value tensors

        # Apply normalization if applicable
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention mechanism
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

        # Reshape and combine heads
        x = x.transpose(1, 2).reshape(B, N, -1)

        # Projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        # Apply final normalization
        x = self.ln1(x)

        return x


class IAB(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        num_inds,
        proj_out=None,
        qkv_bias=False,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        ln=False,
        norm_layer=nn.LayerNorm,
    ):
        super(IAB, self).__init__()

        # Initialize the induced points
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        # Multi-Head Attention Block
        self.mab0 = MAB(
            dim_Q=dim_out,
            dim_K=dim_in,
            dim_V=dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ln=ln,
            norm_layer=norm_layer,
        )

        # Activation function with approximation
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # Output projection through MLP
        self.norm = nn.LayerNorm(dim_out, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=4 * dim_out,
            out_features=proj_out,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, X):
        # Repeat induced points for the batch dimension and apply MAB
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        H = self.mlp(self.norm(H))
        return H


class IAB_Simple(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        num_inds,
        qkv_bias=False,
        qk_norm=True,
        attn_drop=0.0,
        proj_drop=0.0,
        ln=False,
        norm_layer=nn.LayerNorm,
    ):
        super(IAB_Simple, self).__init__()

        # Initialize the induced points
        self.I = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        # Multi-Head Attention Block
        self.mab0 = MAB(
            dim_Q=dim_out,
            dim_K=dim_in,
            dim_V=dim_out,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ln=ln,
            norm_layer=norm_layer,
        )

        # Activation function with approximation
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # Output projection through MLP
        self.norm = nn.LayerNorm(dim_out, elementwise_affine=False, eps=1e-6)

    def forward(self, X):
        # Repeat induced points for the batch dimension and apply MAB
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return H


class DiTBlock_Basic(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=False,
            qk_norm=True,
            norm_layer=nn.LayerNorm,
            **block_kwargs
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PointNet_Large(nn.Module):
    def __init__(
        self, output_dim=1152, pc_dims=1024, num_inds=256, num_heads=16, depth=4
    ):
        super(PointNet_Large, self).__init__()

        # Define linear layers with corresponding layer normalization
        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, 1024)
        self.linear4 = nn.Linear(1024, pc_dims)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(1024)
        self.ln4 = nn.LayerNorm(pc_dims)

        # Define pooling using IAB (Induced Attention Block)
        self.pooling = IAB(
            pc_dims,
            pc_dims,
            qkv_bias=True,
            num_heads=num_heads,
            num_inds=num_inds,
            ln=False,
            proj_out=output_dim,
        )

        # Define Transformer-like blocks
        self.t_blocks = nn.ModuleList(
            [DiTBlock_Basic(output_dim, num_heads, mlp_ratio=4) for _ in range(depth)]
        )

    def forward(self, x):
        # Apply linear layers followed by SiLU activation and layer normalization
        x = F.silu(self.ln1(self.linear1(x)))
        x = F.silu(self.ln2(self.linear2(x)))
        x = F.silu(self.ln3(self.linear3(x)))
        x = F.silu(self.ln4(self.linear4(x)))

        # Apply induced attention pooling
        x = self.pooling(x)

        # Apply transformer blocks
        for block in self.t_blocks:
            x = block(x)

        return x


class PointNet_Simple(nn.Module):
    def __init__(self, output_dim=128, pc_dims=1024, num_inds=256, num_heads=8):
        super(PointNet_Simple, self).__init__()
        # Define linear layers with corresponding layer normalization
        self.linear1 = nn.Linear(3, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, 1024)
        self.linear4 = nn.Linear(1024, pc_dims)
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(1024)
        self.ln4 = nn.LayerNorm(pc_dims)

        # Define pooling using IAB (Induced Attention Block)
        self.pooling = IAB_Simple(
            pc_dims, pc_dims, num_heads=num_heads, num_inds=num_inds, ln=False
        )

        self.linear_f1 = nn.Linear(pc_dims, output_dim)
        self.linear_f2 = nn.Linear(output_dim, output_dim)
        self.linear_f3 = nn.Linear(output_dim, output_dim)
        self.ln_f1 = nn.LayerNorm(output_dim)
        self.ln_f2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        x = F.relu(self.ln4(self.linear4(x)))
        # x = F.relu(self.ln5(self.linear5(x)))
        # print(x.shape)
        x = self.pooling(x)
        x = F.relu(self.ln_f1(self.linear_f1(x)))
        x = F.relu(self.ln_f2(self.linear_f2(x)))
        x = self.linear_f3(x)
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    x = torch.randn([10, 2048, 3])

    net = PointNet_Large()  # PointNet_Simple()
    print(net)
    print(net(x).shape)
