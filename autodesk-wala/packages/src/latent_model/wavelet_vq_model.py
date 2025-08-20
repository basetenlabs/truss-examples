import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.latent_model.quantize import VectorQuantizer2
from src.diffusion_modules.dwt import DWTInverse3d
from src.diffusion_modules.sparse_network import SparseComposer
from src.experiments.utils.wavelet_utils import (
    extract_wavelet_coefficients,
    extract_full_indices,
    extract_highs_from_values,
    pad_with_batch_idx,
)
from src.latent_model.abstract_volume_nn import (
    General_Decoder_Up_2,
    General_Encoder_Down_2,
)
from src.experiments.utils.wavelet_utils import WaveletData


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def get_encoder(args, input_chan):
    if args.encoder_type == "General_Encoder_Down_2":
        return General_Encoder_Down_2(
            channel_in=input_chan,
            channel_out=args.e_dim,
            grid_size=args.grid_size,
            transformer_layers=args.encoder_num_tran,
            dropout=args.dropout,
            num_heads=4,
            hidden_channel=128,
        )
    else:
        raise ValueError(
            f"Unsupported encoder type: {args.encoder_type}. Expected 'General_Encoder_Down_2'."
        )


def get_decoder(args, output_channel):
    if args.decoder_type == "General_Decoder_Up_2":
        return General_Decoder_Up_2(
            channel_in=args.e_dim,
            channel_out=output_channel,
            grid_size=args.grid_size,
            transformer_layers=args.decoder_num_tran,
            dropout=args.dropout,
            num_heads=4,
            hidden_channel=128,
        )
    else:
        raise ValueError(
            f"Unsupported decoder type: {args.decoder_type}. Expected 'General_Decoder_Up_2'."
        )


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()

        ### Local Model Hyperparameters
        self.args = args
        self.encoder_type = args.encoder_type
        self.decoder_type = args.decoder_type
        self.reconstruct_loss_type = args.reconstruct_loss_type
        self.n_e = args.n_e
        self.e_dim = args.e_dim
        self.beta = args.beta  # 0.25
        self.input_type = args.input_type
        self.output_type = args.output_type
        self.grid_size = args.grid_size
        self.use_sample_training = args.use_sample_training
        self.no_rebalance_loss = args.no_rebalance_loss

        coeff_index = self.args.max_depth - self.args.max_training_level

        if coeff_index == 0:
            self.current_stage = 3
            input_chan = 512
            output_chan = 512
        elif coeff_index == 1:
            self.current_stage = 2
            input_chan = 64
            output_chan = 64
        elif coeff_index == 2:
            self.current_stage = 1
            input_chan = 8
            output_chan = 8
        elif coeff_index == 3:
            self.current_stage = 0
            input_chan = 1
            output_chan = 1

        self.input_chan = input_chan
        self.output_chan = output_chan
        ### Sub-Network def
        self.encoder = get_encoder(args, input_chan)
        self.decoder = get_decoder(args, output_chan)

        self.n_e = args.n_e
        # self.grid_size = 12

        self.normalize_latent = args.normalize_latent

        self.quantize = VectorQuantizer2(
            args.n_e, args.e_dim, beta=args.beta, normalize=args.normalize_latent
        )

        ### Local Model Hyperparameters
        high_size = (
            511
            if not hasattr(self.args, "max_training_level")
            or self.args.max_training_level == self.args.max_depth
            else (2**3) ** self.args.max_training_level - 1
        )
        low_avg = self.args.low_avg if hasattr(self.args, "low_avg") else 2.20
        print(f"Low avg used : {low_avg} high value: {high_size}")
        # self.avg_value = torch.from_numpy(np.array([low_avg] + [0] * high_size)) ### HARD_CODE first
        # self.scale_value = torch.ones_like(self.avg_value)
        ### sparse models
        self.dwt_sparse_composer = SparseComposer(
            input_shape=[args.resolution, args.resolution, args.resolution],
            J=args.max_depth,
            wave=args.wavelet,
            mode=args.padding_mode,
            inverse_dwt_module=None,
        )

        self.dwt_inverse_3d = DWTInverse3d(
            J=args.max_depth, wave=args.wavelet, mode=args.padding_mode
        )

    def indices_to_z(self, indices):
        ix_to_vectors = self.quantize.embedding(indices).reshape(
            indices.shape[0], self.grid_size, self.grid_size, self.grid_size, -1
        )
        quant = ix_to_vectors.permute(0, 4, 1, 2, 3)
        return quant

    @torch.no_grad()
    def indices_to_shape(self, indices, query_points=None):
        ix_to_vectors = self.quantize.embedding(indices).reshape(
            indices.shape[0], self.grid_size, self.grid_size, self.grid_size, -1
        )
        ix_to_vectors = ix_to_vectors.permute(0, 4, 1, 2, 3)
        shape = self.decode(ix_to_vectors)
        return shape

    def data_process(
        self,
        low,
        high_indices,
        high_values,
        high_values_mask=None,
        high_indices_empty=None,
    ):
        wavelet_data = WaveletData(
            shape_list=self.dwt_sparse_composer.shape_list,
            output_stage=self.args.max_training_level,
            max_depth=self.args.max_depth,
            low=low,
            highs_indices=high_indices,
            highs_values=high_values,
        )
        data_samples = wavelet_data.convert_wavelet_volume()
        return data_samples

    def encode(
        self,
        low,
        high_indices,
        high_values,
        high_values_mask=None,
        high_indices_empty=None,
    ):
        data_samples = self.data_process(
            low,
            high_indices,
            high_values,
            high_values_mask=high_values_mask,
            high_indices_empty=high_indices_empty,
        )
        h = self.encoder(data_samples)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, {}

    def encode_to_pre_quant(
        self,
        low,
        high_indices,
        high_values,
        high_values_mask=None,
        high_indices_empty=None,
    ):
        data_samples = self.data_process(
            low,
            high_indices,
            high_values,
            high_values_mask=high_values_mask,
            high_indices_empty=high_indices_empty,
        )
        z = self.encoder(data_samples)
        return z

    def encode_to_z(
        self,
        low,
        high_indices,
        high_values,
        high_indices_empty=None,
        high_values_mask=None,
    ):
        data_samples = self.data_process(
            low,
            high_indices,
            high_values,
            high_values_mask=high_values_mask,
            high_indices_empty=high_indices_empty,
        )
        h = self.encoder(data_samples)
        quant, emb_loss, (_, _, _, indices) = self.quantize(h)
        indices = indices.view(quant.shape[0], -1)
        return h, indices.to(torch.int64), quant

    def convert_to_post_quant(self, h):
        quant, _, _ = self.quantize(h)
        return quant

    def decode_from_pre_quant(self, h):
        quant, emb_loss, info = self.quantize(h)
        dec = self.decoder(quant)
        return dec

    def decode(self, embs):
        dec = self.decoder(embs)
        return dec

    def forward(
        self,
        low,
        high_indices,
        high_values,
        high_values_mask=None,
        high_indices_empty=None,
        get_coeff=False,
    ):
        quant, diff, info, model_kawrgs = self.encode(
            low,
            high_indices,
            high_values,
            high_values_mask=high_values_mask,
            high_indices_empty=high_indices_empty,
        )
        dec = self.decode(quant)

        if get_coeff == True:
            wavelet_data_pred = WaveletData(
                shape_list=self.dwt_sparse_composer.shape_list,
                output_stage=self.args.max_training_level,
                max_depth=self.args.max_depth,
                wavelet_volume=dec,
            )
            # print(dec.shape)
            low_pred, highs_pred = wavelet_data_pred.convert_low_highs()
            return dec, diff, info, model_kawrgs, low_pred, highs_pred

        # print(dec.shape)
        return dec, diff, info, model_kawrgs

    def surface_loss(
        self,
        target,
        model_output,
        high_indices,
        high_indices_empty=None,
        high_values_mask=None,
    ):
        batch_size = target.size(0)
        non_empty_indices = high_indices[:, :, 1:].long()
        training_indices = torch.cat((non_empty_indices, high_indices_empty), dim=1)
        training_indices = pad_with_batch_idx(training_indices)
        indices = training_indices.view(-1, 4)
        masks = high_values_mask.repeat(1, 2).view(-1)

        # print(masks.shape, indices.shape, d1.shape, pred.shape, indices[0])
        # raise "err"
        target_idx = torch.permute(target, (0, 2, 3, 4, 1))
        target_idx = target_idx[
            indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], :
        ]
        model_output_idx = torch.permute(model_output, (0, 2, 3, 4, 1))
        model_output_idx = model_output_idx[
            indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], :
        ]

        # compute loss
        losses_order_idx = (target_idx - model_output_idx) ** 2
        losses_order_idx = losses_order_idx.view(
            -1, losses_order_idx.size(-1)
        )  ## get reshape

        # print(losses_order_idx.shape, model_output_idx.shape, target_idx.shape)

        ## masking un related loss
        losses_order_idx_masked = losses_order_idx * masks.unsqueeze(1)
        losses_order_idx_masked = losses_order_idx_masked.view(
            batch_size, -1, losses_order_idx_masked.size(-1)
        )

        ## compute the loss
        loss = sum_flat(losses_order_idx_masked) / torch.sum(high_values_mask, dim=1)
        return loss

    def reconstruction_loss(
        self,
        pred,
        low,
        high_indices,
        high_values,
        high_values_mask=None,
        high_indices_empty=None,
    ):
        data_samples = self.data_process(
            low,
            high_indices,
            high_values,
            high_values_mask=high_values_mask,
            high_indices_empty=high_indices_empty,
        )
        base_loss = mean_flat((pred[:, 0] - data_samples[:, 0]) ** 2)
        final_loss = base_loss
        batch_size = pred.size(0)
        stage_losses = []

        start_cnt = 0
        eps = 1e-10
        terms = {}

        for order_idx in range(self.current_stage):
            order = (2 ** (order_idx + 1)) ** 3
            idx = order - 1
            # print(-idx, self.output_chan - start_cnt)
            # print(data_samples[:, -idx:self.output_chan - start_cnt].shape, pred[:, -idx:self.output_chan - start_cnt].shape)
            if self.use_sample_training == True:
                loss_order_idx = self.surface_loss(
                    data_samples[:, -idx : self.output_chan - start_cnt],
                    pred[:, -idx : self.output_chan - start_cnt],
                    high_indices,
                    high_indices_empty=high_indices_empty,
                    high_values_mask=high_values_mask,
                ).mean()
            else:
                loss_order_idx = mean_flat(
                    (
                        pred[:, -idx : self.output_chan - start_cnt]
                        - data_samples[:, -idx : self.output_chan - start_cnt]
                    )
                    ** 2
                )

            if self.no_rebalance_loss:
                final_loss = final_loss + loss_order_idx
            else:
                final_loss = (
                    final_loss
                    + (base_loss.detach() / (loss_order_idx.detach() + eps))
                    * loss_order_idx
                )

            stage_losses.append(loss_order_idx)
            start_cnt = idx

        terms["mse"] = final_loss
        terms["base_loss"] = base_loss

        ### logging the losses
        for order_idx in range(self.current_stage):  #
            terms[f"loss_{order_idx+1}"] = stage_losses[order_idx]
        return terms


if __name__ == "__main__":
    pass
