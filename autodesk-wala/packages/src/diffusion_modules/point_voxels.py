import functools

import torch.nn as nn
import torch
import numpy as np
from src.diffusion_modules.latent_points import CrossAttention
from src.diffusion_modules.modules import (
    SharedMLP,
    PVConv,
    PointNetSAModule,
    PointNetAModule,
    PointNetFPModule,
    Attention,
    Swish,
)


class PointVoxelEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        sa_blocks = [
            ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
            ((64, 3, 16), (512, 0.2, 32, (64, 128))),
            ((128, 3, 8), (256, 0.4, 32, (128, 256))),
            (None, (args.num_inds, 0.8, 32, (256, 256, args.pc_output_dim))),
        ]

        sa_layers, sa_in_channels, in_channels, num_centers = (
            create_pointnet2_sa_components(
                sa_blocks=sa_blocks,
                with_se=True,
                extra_feature_channels=0,
                embed_dim=0,
                eps=1e-3,
            )
        )
        self.sa_layers = nn.ModuleList(sa_layers)

    def forward(self, inputs):
        inputs = torch.permute(inputs, (0, 2, 1)).float()

        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        temb = None
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, temb = sa_blocks((features, coords, temb))

        features = torch.permute(features, (0, 2, 1)).float()

        return features


def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels), nn.GroupNorm(8, out_channels), Swish()
    )


def create_pointnet2_sa_components(
    sa_blocks,
    extra_feature_channels,
    embed_dim=64,
    use_att=False,
    dropout=0.1,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps,
                    )

                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_blocks.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(
                PointNetSAModule,
                num_centers=num_centers,
                radius=radius,
                num_neighbors=num_neighbors,
            )
        sa_blocks.append(
            block(
                in_channels=extra_feature_channels + (embed_dim if k == 0 else 0),
                out_channels=out_channels,
                include_coordinates=True,
            )
        )
        c += 1
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return (
        sa_layers,
        sa_in_channels,
        in_channels,
        1 if num_centers is None else num_centers,
    )


def create_pointnet2_fp_modules(
    fp_blocks,
    in_channels,
    sa_in_channels,
    embed_dim=64,
    use_att=False,
    dropout=0.1,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(
                in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
                out_channels=out_channels,
            )
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (
                    (c + 1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                )
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv,
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution),
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        with_se_relu=True,
                        normalize=normalize,
                        eps=eps,
                    )

                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

        c += 1

    return fp_layers, in_channels


def create_mlp_components(
    in_channels, out_channels, classifier=False, dim=2, width_multiplier=1
):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


class PVCNN2Base(nn.Module):

    def __init__(
        self,
        num_classes,
        embed_dim,
        use_att,
        context_dim,
        d_head,
        n_heads,
        dropout=0.1,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = (
            create_pointnet2_sa_components(
                sa_blocks=self.sa_blocks,
                extra_feature_channels=extra_feature_channels,
                with_se=True,
                embed_dim=embed_dim,
                use_att=use_att,
                dropout=dropout,
                width_multiplier=width_multiplier,
                voxel_resolution_multiplier=voxel_resolution_multiplier,
            )
        )
        self.sa_layers = nn.ModuleList(sa_layers)
        self.sa_cross_attn = [
            CrossAttention(
                query_dim=self.sa_blocks[idx][1][3][-1],
                context_dim=context_dim,
                dim_head=d_head,
                heads=n_heads,
            )
            for idx in range(len(sa_layers))
        ]
        self.sa_cross_attn = nn.ModuleList(self.sa_cross_attn)

        self.global_att = (
            None if not use_att else Attention(channels_sa_features, 8, D=1)
        )

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fp_layers = nn.ModuleList(fp_layers)
        self.fp_cross_attn = [
            CrossAttention(
                query_dim=self.fp_blocks[idx][0][-1], context_dim=context_dim
            )
            for idx in range(len(fp_layers))
        ]
        self.fp_cross_attn = nn.ModuleList(self.fp_cross_attn)

        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier,
        )
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, t, cond=None):
        ### convert to B C N
        inputs = torch.permute(inputs, (0, 2, 1)).float()
        temb = (
            self.embedf(self.get_timestep_embedding(t, inputs.device))[:, :, None]
            .expand(-1, -1, inputs.shape[-1])
            .float()
        )

        # inputs : [B, in_channels + S, N]
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks(
                    (torch.cat([features, temb], dim=1), coords, temb)
                )

            features = torch.permute(features, (0, 2, 1))
            features = self.sa_cross_attn[i](features, cond)
            features = torch.permute(features, (0, 2, 1))
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks(
                (
                    coords_list[-1 - fp_idx],
                    coords,
                    torch.cat([features, temb], dim=1),
                    in_features_list[-1 - fp_idx],
                    temb,
                )
            )
            features = torch.permute(features, (0, 2, 1))
            features = self.fp_cross_attn[fp_idx](features, cond)
            features = torch.permute(features, (0, 2, 1))

        output_pts = self.classifier(features)
        output_pts = torch.permute(output_pts, (0, 2, 1))
        return output_pts


class PVCNN2(PVCNN2Base):
    sa_blocks = [
        ((32, 2, 32), (4096, 0.1, 32, (32, 64))),
        ((64, 3, 16), (512, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        num_classes,
        embed_dim,
        use_att,
        dropout,
        context_dim,
        d_head,
        n_heads,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            context_dim=context_dim,
            d_head=d_head,
            n_heads=n_heads,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
