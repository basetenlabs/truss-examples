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
from src.latent_model.latent_dit_utils import (
    DiT,
    Global_Proj_Layer,
    TimestepEmbedder,
    DiTBlock,
    Cross_DiTBlock,
    get_3d_sincos_pos_embed,
)

from inspect import isfunction
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#################################################################################
#                              Core Resnet model stuff                          #
#################################################################################


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


### REMARK: Change to 4
def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D
    """

    def __init__(
        self, channels, out_channels=None, use_conv=False, dims=2, output_res=None
    ):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.output_res = output_res
        out_channels = default(out_channels, channels)
        if use_conv:
            self.conv = conv_nd(dims, channels, out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)

        if self.output_res is not None:
            if self.output_res < x.size(-1):
                x = x[..., :-1]
            if self.output_res < x.size(-2):
                x = x[..., :-1, :]
            if self.output_res < x.size(-3):
                x = x[..., :-1, :, :]
        return x


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        activation=nn.SiLU(),
        skip_h=None,
        learnable_skip_r=None,
        res_emb_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.res_emb_channels = res_emb_channels

        if skip_h is not None:
            self.skip_norm = normalization(channels)
            self.learnable_skip_r = learnable_skip_r
            if learnable_skip_r is not None:
                self.skip_learn_f = nn.Sequential(
                    nn.Linear(channels, channels // learnable_skip_r),
                    nn.ReLU(),
                    nn.Linear(channels // learnable_skip_r, channels),
                    nn.Sigmoid(),
                )

        self.in_norm = normalization(channels)
        self.act1 = activation
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.emb_layers = nn.Sequential(
            activation,
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.res_emb_channels is not None:
            self.linear_res_emb = linear(self.res_emb_channels, self.out_channels)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, skip_h=None, res_emb=None):
        B, H, W, L, C = x.shape
        h = self.in_norm(x)
        if skip_h is not None:
            skip_h = self.skip_norm(skip_h)
            if self.learnable_skip_r is not None:
                averaged_skip = skip_h.mean(dim=(-3, -2, -1))
                k = self.skip_learn_f(averaged_skip)
                h = h + k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * skip_h
            else:
                h = (h + skip_h) / math.sqrt(2)
        h = self.act1(h)
        h = self.in_conv(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            if self.res_emb_channels is not None:
                torch._assert(res_emb is not None, "res_emb is None")
                z_res = self.linear_res_emb(res_emb)
                while len(z_res.shape) < len(scale.shape):
                    z_res = z_res[..., None]
                h = (out_norm(h) * (1 + scale) + shift) * z_res
            else:
                h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Latent_UVIT(nn.Module):
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
        unet_model_channels=128,
        num_res_blocks=[3],
        add_condition_res_ch=None,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        dropout=0.0,
        learnable_skip_r=None,
        with_fix_pos=None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.unet_model_channels = unet_model_channels
        self.context_dim = context_dim

        self.t_embedder = TimestepEmbedder(hidden_size)

        ### Input block processing
        self.add_condition_res_ch = add_condition_res_ch
        if add_condition_res_ch is not None:
            self.res_cond_proj_layer = Global_Proj_Layer(
                context_dim,
                add_condition_res_ch,
                hidden_dim=hidden_size,
                activation=nn.SiLU(),
            )

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

        self.conv_in = conv_nd(3, in_channels, unet_model_channels, 3, padding=1)
        self.input_blocks = nn.ModuleList([])
        total_res = num_res_blocks[0]
        for _ in range(int(total_res)):
            layer = ResBlock(
                unet_model_channels,
                hidden_size,
                dropout,
                out_channels=unet_model_channels,
                dims=3,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=nn.SiLU(),
                res_emb_channels=add_condition_res_ch,
            )
            self.input_blocks.append(layer)

        ### middle block processing
        self.to_patch_att_embedding_input = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (l p3) -> b  (h w l) (p1 p2 p3 c)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
            ),
            nn.LayerNorm(patch_size * patch_size * patch_size * unet_model_channels),
            nn.Linear(
                patch_size * patch_size * patch_size * unet_model_channels, hidden_size
            ),
        )

        att_size = input_size // patch_size
        total_att_seq = att_size * att_size * att_size

        self.with_fix_pos = with_fix_pos
        self.att_size = att_size
        if with_fix_pos is not None:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, total_att_seq, hidden_size), requires_grad=False
            )
        else:
            self.pos_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, total_att_seq, hidden_size)), 0.0, 0.01
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

        self.to_patch_att_embedding_output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(
                hidden_size, patch_size * patch_size * patch_size * unet_model_channels
            ),
            Rearrange(
                "b (h w l) (p1 p2 p3 c) -> b c (h p1) (w p2) (l p3)",
                p1=patch_size,
                p2=patch_size,
                p3=patch_size,
                h=att_size,
                w=att_size,
                l=att_size,
            ),
        )

        ### Output Processing
        self.output_blocks = nn.ModuleList([])

        for _ in range(int(total_res)):
            layer = ResBlock(
                unet_model_channels,
                hidden_size,
                dropout,
                out_channels=unet_model_channels,
                dims=3,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                activation=nn.SiLU(),
                skip_h=True,
                learnable_skip_r=learnable_skip_r,
                res_emb_channels=add_condition_res_ch,
            )

            self.output_blocks.append(layer)

        self.out = nn.Sequential(
            normalization(unet_model_channels),
            nn.SiLU(),
            zero_module(
                conv_nd(3, unet_model_channels, self.out_channels, 3, padding=1)
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):

        if self.with_fix_pos is not None:
            pos_emb = get_3d_sincos_pos_embed(self.pos_emb.shape[-1], self.att_size)
            self.pos_emb.data.copy_(torch.from_numpy(pos_emb).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t, latent_codes=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        t = self.t_embedder(t)

        if self.add_condition_res_ch is not None:
            res_cond_emb = self.res_cond_proj_layer(latent_codes)
        else:
            res_cond_emb = None
        # print(res_cond_emb.shape)

        if self.context_dim is not None:
            latent_codes_local = self.context_mlp(latent_codes)
        else:
            latent_codes_local = None

        if self.add_condition_time_ch and self.context_dim is not None:
            global_context = self.global_proj(latent_codes)
            att_t = t + global_context
            # print(att_t.shape)
        else:
            att_t = t

        h = self.conv_in(x)
        hs = []
        for module in self.input_blocks:
            h = module(h, t, res_emb=res_cond_emb)
            hs.append(h)

        h = self.to_patch_att_embedding_input(h)
        # print(h.shape)
        h = h + self.pos_emb
        if self.add_num_register > 0:
            # print(h.shape)
            h = torch.cat(
                [h, self.register_emb.repeat(h.size(0), 1, 1).type(h.type())], dim=1
            )
            # print(h.shape)

        for block in self.blocks:
            h, latent_codes_local = block(h, att_t, context=latent_codes_local)
        # print(h.shape)
        if self.add_num_register > 0:
            # print(h.shape)
            h = h[:, : -self.add_num_register, :]
            # print(h.shape)

        h = self.to_patch_att_embedding_output(h)
        # print(h.shape)

        for module in self.output_blocks:
            skip_h = hs.pop()
            h = module(h, t, skip_h=skip_h, res_emb=res_cond_emb)

        out = self.out(h)
        return out


if __name__ == "__main__":
    model = Latent_UVIT(
        input_size=12,
        patch_size=1,
        in_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        context_dim=128,
        block_type="cross_dit",
        add_condition_time_ch=True,
        add_condition_res_ch=128,
        learnable_skip_r=16,
        add_num_register=4,
    )
    # blank_tokens = torch.ones((5, args.grid_size*args.grid_size*args.grid_size)).to(torch.int64)
    x = torch.randn([5, 4, 12, 12, 12])
    condition = torch.randn([5, 256, 128])
    t = torch.randn([5])
    print(model)
    print(model(x, t, latent_codes=condition).shape)
