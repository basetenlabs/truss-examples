import math
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from src.diffusion_modules.fp16_util import (
    convert_module_to_f16,
    convert_module_to_f32,
)
from src.latent_model.gaussian_diffusion import (
    GaussianDiffusion,
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from src.diffusion_modules.resample import (
    UniformSampler,
    LossSecondMomentResampler,
)
from src.diffusion_modules.nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)
from src.latent_model.utils import *
from src.latent_model.latent_dit_utils import DiT
from src.latent_model.latent_uvit_utils import Latent_UVIT

# Main Model


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.args = args

        # Condition network initialization
        self.context_emb_dim = self.get_context_dim()
        self.cond_mapping_network, self.cond_pos_emb, self.camera_emb = (
            self.get_condition_networks()
            if self.context_emb_dim
            else (None, None, None)
        )

        # Initialize the main network based on network type
        self.unet = self.initialize_network()

        # Initialize diffusion modules
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        # Initialize sampler
        self.sampler = self.initialize_sampler()


    def reset_diffusion_module(self):
        betas = get_named_beta_schedule(
            self.args.diffusion_beta_schedule,
            self.args.diffusion_step,
            self.args.diffusion_scale_ratio,
        )
        self.diffusion_module = GaussianDiffusion(
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
            rescale_timesteps=(
                self.args.diffusion_rescale_timestep
                if hasattr(self.args, "diffusion_rescale_timestep")
                else False
            ),
        )

        self.inference_diffusion_module = SpacedDiffusion(
            use_timesteps=space_timesteps(
                self.args.diffusion_step, [self.args.diffusion_rescale_timestep]
            ),
            betas=betas,
            model_var_type=self.args.diffusion_model_var_type,
            model_mean_type=self.args.diffusion_model_mean_type,
            loss_type=self.args.diffusion_loss_type,
        )

        # Initialize sampler
        self.sampler = self.initialize_sampler()

    def get_context_dim(self):
        """Determine the context embedding dimension based on the conditions."""
        if (
            hasattr(self.args, "use_wavelet_conditions")
            and self.args.use_wavelet_conditions
        ):
            return self.args.ae_z_dim
        if (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            return self.args.pc_output_dim
        if (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            return self.args.voxel_context_dim
        if (
            hasattr(self.args, "use_multiple_views_inferences")
            and self.args.use_multiple_views_inferences
        ):
            return self.args.cond_grid_emb_size
        if (
            hasattr(self.args, "use_multiple_views_grids")
            and self.args.use_multiple_views_grids
        ):
            return self.args.cond_grid_emb_size
        if (
            hasattr(self.args, "use_image_conditions")
            and self.args.use_image_conditions
        ):
            return self.args.cond_grid_emb_size
        if (
            hasattr(self.args, "use_depth_conditions")
            and self.args.use_depth_conditions
        ):
            return self.args.cond_grid_emb_size
        return None

    def get_condition_networks(self):
        """Initialize condition networks."""
        cond_mapping_network = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if self.args.cond_num_mapping_layers == 0
                    else nn.Sequential(
                        nn.Linear(self.context_emb_dim, self.context_emb_dim), nn.SiLU()
                    )
                )
                for _ in range(self.args.cond_num_mapping_layers)
            ]
        )

        # Positional encoding for network
        cond_pos_emb = None
        if (
            hasattr(self.args, "use_pointcloud_conditions")
            and self.args.use_pointcloud_conditions
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(torch.zeros(1, self.args.num_inds, self.context_emb_dim)),
                0.0,
                0.02,
            )
        elif (
            hasattr(self.args, "use_voxel_conditions")
            and self.args.use_voxel_conditions
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, self.args.voxel_dim**3, self.context_emb_dim)
                ),
                0.0,
                0.02,
            )
        elif (
            hasattr(self.args, "use_multiple_views_inferences")
            and self.args.use_multiple_views_inferences
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, self.args.cond_grid_size, self.context_emb_dim)
                ),
                0.0,
                0.02,
            )
        elif (
            hasattr(self.args, "use_multiple_views_grids")
            and self.args.use_multiple_views_grids
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, self.args.cond_grid_size, self.context_emb_dim)
                ),
                0.0,
                0.02,
            )
        elif (
            hasattr(self.args, "use_image_conditions")
            and self.args.use_image_conditions
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, self.args.cond_grid_size, self.context_emb_dim)
                ),
                0.0,
                0.02,
            )
        elif (
            hasattr(self.args, "use_depth_conditions")
            and self.args.use_depth_conditions
        ):
            cond_pos_emb = nn.init.trunc_normal_(
                nn.Parameter(
                    torch.zeros(1, self.args.cond_grid_size, self.context_emb_dim)
                ),
                0.0,
                0.02,
            )

        if (
            self.args.use_multiple_views_grids == True
            or self.args.use_camera_index == True
        ):
            if self.args.training_views is not None:
                camera_emb = nn.Embedding(
                    len(self.args.training_views), self.context_emb_dim
                )
            else:
                camera_emb = nn.Embedding(
                    self.args.max_images_num, self.context_emb_dim
                )
        else:
            camera_emb = None

        return cond_mapping_network, cond_pos_emb, camera_emb

    def initialize_network(self):
        """Initialize the network based on the specified network type."""
        network_params = {
            "input_size": self.args.grid_size,
            "patch_size": self.args.att_patch_size,
            "in_channels": self.args.e_dim,
            "hidden_size": self.args.att_hidden_size,
            "depth": self.args.transformer_num_blocks,
            "num_heads": self.args.transformer_num_heads,
            "mlp_ratio": 4.0,
            "context_dim": self.context_emb_dim,
            "block_type": self.args.dit_block_type,
            "add_condition_time_ch": self.args.add_condition_time_ch,
            "add_num_register": self.args.transformer_add_num_register,
        }

        if self.args.network_type == "latent_dit":
            return DiT(**network_params)
        else:
            # Additional parameters specific to Latent_UVIT
            latent_uvit_params = {
                "unet_model_channels": self.args.unet_model_channels,
                "num_res_blocks": self.args.unet_num_res_blocks,
                "add_condition_res_ch": self.args.add_condition_res_ch,
                "use_scale_shift_norm": True,
                "dropout": 0.0,
                "learnable_skip_r": self.args.learnable_skip_r,
                "with_fix_pos": self.args.with_fix_pos,
            }
            # Merge shared and specific parameters
            network_params.update(latent_uvit_params)
            return Latent_UVIT(**network_params)

    def initialize_sampler(self):
        """Initialize the sampler based on the specified sampler type."""
        if self.args.diffusion_sampler == "uniform":
            return UniformSampler(self.diffusion_module)
        elif self.args.diffusion_sampler == "second-order":
            return LossSecondMomentResampler(self.diffusion_module)
        else:
            raise Exception("Unknown Sampler...")

 

    def process_condition(self, condition, image_index=None):
        """Process and apply condition mappings."""
        if condition is None:
            return None

        if self.args.use_camera_index and image_index is None:
            assert False, "Error: image index is missing"

        cross_condition = condition
        for layer in self.cond_mapping_network:
            cross_condition = layer(cross_condition)

        if self.cond_pos_emb is not None:
            if (
                hasattr(self.args, "use_multiple_views_inferences")
                and self.args.use_multiple_views_inferences
            ):
                cross_condition = cross_condition + self.cond_pos_emb.unsqueeze(0)
            elif (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                cross_condition = cross_condition + self.cond_pos_emb.unsqueeze(0)
            else:
                cross_condition = cross_condition + self.cond_pos_emb

        if image_index is not None:
            if self.args.training_views is None:
                camera_emb = self.camera_emb(image_index)
            else:
                camera_emb = self.camera_emb.weight
            # print("camera", camera_emb.shape)
            if getattr(self.args, "use_multiple_views_grids", False):
                camera_emb = camera_emb.unsqueeze(0).unsqueeze(2)
            elif len(camera_emb.size()) != len(cross_condition.size()):
                camera_emb = camera_emb.unsqueeze(1)

            # print("image_index_2", cross_condition.shape, camera_emb.shape)
            cross_condition = cross_condition + camera_emb

        if getattr(self.args, "use_multiple_views_grids", False):
            cross_condition = cross_condition.view(
                cross_condition.size(0), -1, cross_condition.size(-1)
            )
        # print("cross_condition", cross_condition.shape)
        return cross_condition

    def inference(self, batch_size, condition=None, scale=3, image_index=None):
        """Perform inference using the diffusion module."""
        if self.args.dp_cond is not None and condition is not None:
            condition_zero = (
                torch.zeros_like(condition)
                if self.args.dp_cond_type is None
                else self.cond_zero_emb.repeat_interleave(condition.size(0), dim=0)
            )
            condition = torch.cat([condition_zero, condition])
            condition_zero, condition = self.process_condition(
                condition, image_index=image_index
            ).chunk(2)
        else:
            condition_zero = None
            condition = self.process_condition(condition, image_index=image_index)

        shape = (
            batch_size,
            self.args.e_dim,
            self.args.grid_size,
            self.args.grid_size,
            self.args.grid_size,
        )
        device = next(self.parameters()).device
        samples, _ = self.inference_diffusion_module.p_sample_loop(
            model=self.unet,
            shape=shape,
            device=device,
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "latent_codes": condition,
                "condition_zero": condition_zero,
                "guidance_scale": scale,
                "dp_cond": self.args.dp_cond,
            },
        )

        return samples

    def training_losses(self, data_input, condition=None, image_index=None):
        """Compute training losses."""
        t, weights = self.sampler.sample(data_input.size(0), device=data_input.device)

        if self.args.dp_cond is not None and condition is not None:
            anti_mask = (
                prob_mask_like(
                    (condition.size(0)), 1 - self.args.dp_cond, condition.device
                )
                .unsqueeze(1)
                .unsqueeze(1)
            )
            if (
                hasattr(self.args, "use_multiple_views_grids")
                and self.args.use_multiple_views_grids
            ):
                anti_mask = anti_mask.unsqueeze(1)
            # print(condition.shape, anti_mask.shape, image_index)
            condition = (
                torch.where(anti_mask, condition, self.cond_zero_emb)
                if self.args.dp_cond_type
                else condition * anti_mask
            )

        condition = self.process_condition(condition, image_index=image_index)
        loss = self.diffusion_module.training_losses(
            self.unet, x_start=data_input, t=t, latent_codes=condition
        )
        return torch.mean(weights * loss["loss"])
