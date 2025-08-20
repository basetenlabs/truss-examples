import math
import torch
import torch as th
import torch.nn as nn
from abc import abstractmethod
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

if hasattr(torch, "_dynamo"):
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 1024

allow_ops_in_compiled_graph()


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


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

    def __init__(self, channels, out_channels=None, use_conv=False, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        out_channels = default(out_channels, channels)
        if use_conv:
            self.conv = conv_nd(dims, channels, out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 1024

        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        if self.use_conv:
            x = self.conv(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, hidden_size, expansion_factor=4, dropout=0.0, activation=nn.SiLU()
    ):
        super().__init__()
        self.transformer_dropout = dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size * expansion_factor)
        self.out = nn.Linear(hidden_size * expansion_factor, hidden_size)
        self.activation = activation

        # Apply zero initialization
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # Initialize biases to zero

    def forward(self, x):
        B, HWL, C = x.shape
        x = self.norm(x)
        mlp_h = self.dense1(x)
        mlp_h = self.activation(mlp_h)  # F.silu(mlp_h)
        if self.transformer_dropout > 0.0:
            mlp_h = nn.functional.dropout(
                mlp_h, p=self.transformer_dropout, training=self.training
            )
        out = self.out(mlp_h)
        return out


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=4, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.norm = nn.LayerNorm(query_dim, elementwise_affine=False)

        self.heads = heads

        self.to_q = nn.Sequential(
            nn.SiLU(), nn.Linear(query_dim, inner_dim)
        )  # nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, inner_dim)
        )  # nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, inner_dim)
        )  # nn.Linear(context_dim, inner_dim, bias=False)

        self.norm_q = nn.LayerNorm(inner_dim)
        self.norm_k = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.SiLU(), zero_module(nn.Linear(inner_dim, query_dim))
        )

        # Apply zero initialization
        # nn.init.zeros_(self.to_out.weight)
        # nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context=None):
        B, HWL, C = x.shape
        if context is not None:
            B, T, TC = context.shape
        h = self.heads

        x_norm = self.norm(x)

        q = self.to_q(x_norm)

        context = default(context, x_norm)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = q * q.shape[-1] ** -0.5

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        weights = einsum("b i d, b j d -> b i j", q, k)  ## unnormalized att maps.
        weights = weights.softmax(dim=-1)  ## normalized att maps
        attn_vals = einsum("b i j, b j d -> b i d", weights, v)
        out = rearrange(attn_vals, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class Transformer_Block(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=4, dropout=0.0, activation=nn.SiLU()
    ):
        super().__init__()
        self.mlp = MLP(
            query_dim, expansion_factor=4, dropout=dropout, activation=activation
        )
        self.att = Attention(
            query_dim, context_dim=context_dim, heads=heads, dim_head=64
        )

    def forward(self, x, context=None):
        x = x + self.mlp(x)
        x = x + self.att(x, context=context)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        activation=SiLU(),
        skip_h=None,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        if skip_h is not None:
            self.skip_norm = normalization(channels)

        self.in_norm = normalization(channels)
        self.act1 = activation
        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, skip_h=None):
        B, H, W, L, C = x.shape
        h = self.in_norm(x)
        if skip_h is not None:
            # print("res", x.shape, skip_h.shape)
            skip_h = self.skip_norm(skip_h)
            h = (h + skip_h) / math.sqrt(2)
        h = self.act1(h)
        h = self.in_conv(h)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


##############################################################################################################################################
##############################################################################################################################################


class General_Encoder_Down_2(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        grid_size,
        num_heads=4,
        dropout=0.0,
        activation=SiLU(),
        transformer_layers=5,
        hidden_channel=128,
    ):
        super().__init__()
        self.actvn = F.relu
        self.transformer_layers = transformer_layers

        self.in_conv = nn.Conv3d(channel_in, hidden_channel, 3, padding=1, stride=1)
        self.res_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_1_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.down_1 = Downsample(hidden_channel, False, dims=3)
        self.res_2 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_2_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.down_2 = Downsample(hidden_channel, False, dims=3)
        self.res_3 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_3_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )

        if self.transformer_layers > 0:
            self.pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, grid_size * grid_size * grid_size, hidden_channel)
                ),
                0.0,
                0.01,
            )
            self.grid_size = grid_size

            self.t_blocks = nn.ModuleList([])
            for i in range(transformer_layers):
                self.t_blocks.append(
                    Transformer_Block(
                        hidden_channel,
                        context_dim=None,
                        heads=num_heads,
                        dropout=dropout,
                        activation=activation,
                    )
                )

        self.res_out = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )

        self.out_conv = nn.Sequential(
            normalization(hidden_channel),
            activation,
            conv_nd(3, hidden_channel, channel_out, 3, padding=1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.in_conv(x)
        x = self.res_1(x)
        x = self.res_1_1(x)
        x = self.down_1(x)  # H/2
        x = self.res_2(x)
        x = self.res_2_1(x)
        x = self.down_2(x)  # H/4
        x = self.res_3(x)
        x = self.res_3_1(x)

        if self.transformer_layers > 0:
            x = x.permute(0, 2, 3, 4, 1)  # BCHWL --> BHWLC
            x = x.view(x.size(0), -1, x.size(-1))  # BHWLC --> B(HWL)C
            x = x + self.pos_emb
            for module in self.t_blocks:
                x = module(x)
            x = x.permute(0, 2, 1)  # B(HWL)C --> BC(HWL)
            x = x.view(
                x.size(0), x.size(1), self.grid_size, self.grid_size, self.grid_size
            )  # BC(HWL) --> BCHWL

        x = self.res_out(x)
        x = self.out_conv(x)

        return x


class General_Decoder_Up_2(nn.Module):
    def __init__(
        self,
        channel_in,
        channel_out,
        grid_size,
        transformer_layers=5,
        num_heads=4,
        dropout=0.0,
        activation=SiLU(),
        hidden_channel=128,
        odd_or_even="odd",
    ):
        super().__init__()

        self.in_conv = nn.Conv3d(channel_in, hidden_channel, 3, padding=1, stride=1)
        self.transformer_layers = transformer_layers
        self.odd_or_even = odd_or_even

        self.res_in = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )

        if self.transformer_layers > 0:
            self.pos_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, grid_size * grid_size * grid_size, 128)),
                0.0,
                0.01,
            )
            self.grid_size = grid_size

            self.t_blocks = nn.ModuleList([])
            for i in range(transformer_layers):
                self.t_blocks.append(
                    Transformer_Block(
                        128,
                        context_dim=None,
                        heads=num_heads,
                        dropout=dropout,
                        activation=nn.SiLU(),
                    )
                )

        self.res_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_1_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )

        self.up_1 = Upsample(hidden_channel, use_conv=False, dims=3)
        self.res_2 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_2_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.up_2 = Upsample(hidden_channel, use_conv=False, dims=3)
        self.res_3 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.res_3_1 = ResBlock(
            hidden_channel, dropout, out_channels=hidden_channel, dims=3
        )
        self.out_conv = nn.Sequential(
            normalization(hidden_channel),
            activation,
            conv_nd(3, hidden_channel, channel_out, 3, padding=1),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.in_conv(x)
        x = self.res_in(x)

        if self.transformer_layers > 0:
            x = x.permute(0, 2, 3, 4, 1)  # BCHWL --> BHWLC
            x = x.view(x.size(0), -1, x.size(-1))  # BHWLC --> B(HWL)C
            x = x + self.pos_emb
            for module in self.t_blocks:
                x = module(x)
            x = x.permute(0, 2, 1)  # B(HWL)C --> BC(HWL)
            x = x.view(
                x.size(0), x.size(1), self.grid_size, self.grid_size, self.grid_size
            )  # BC(HWL)--> BCHWL

        x = self.res_1(x)
        x = self.res_1_1(x)
        x = self.up_1(x)  ## 2 * h
        if self.odd_or_even == "odd":
            x = x[..., :-1, :-1, :-1]
        x = self.res_2(x)
        x = self.res_2_1(x)
        x = self.up_2(x)  ## 4 * h
        x = self.res_3(x)
        x = self.res_3_1(x)
        x = self.out_conv(x)
        return x

