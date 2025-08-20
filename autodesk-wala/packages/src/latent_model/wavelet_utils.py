import numpy as np
import torch


class WaveletData(object):
    def __init__(
        self,
        shape_list,
        output_stage,
        max_depth,
        data_stage=None,
        low=None,
        highs_values=None,
        highs_indices=None,
        wavelet_volume=None,
    ):

        if data_stage is None:
            self.data_stage = max(1, output_stage)
        else:
            self.data_stage = data_stage

        self.max_depth = max_depth
        self.output_stage = output_stage
        self.shape_list = shape_list
        self.wavelet_volume = wavelet_volume
        self.highs_indices = (
            highs_indices[..., 1:] if highs_indices is not None else highs_indices
        )  # extract only the last three dimension
        self.highs_values = highs_values
        self.low = low

    def convert_wavelet_volume(self):

        ### if wavelet volume is given then return it
        if self.wavelet_volume is not None:
            return self.wavelet_volume
        else:
            # if output stage == 0 --> C0 only
            if self.output_stage == 0:
                return self.low
            else:

                ## assertion
                assert self.low.size(0) == self.highs_values.size(
                    0
                ) and self.highs_values.size(0) == self.highs_indices.size(0)
                assert self.data_stage != 0  # if 0 -> the output stage should be 0
                assert self.highs_values.size(1) == self.highs_indices.size(1)

                batch_size = self.low.size(0)

                ### create zero array
                output_dim = 8**self.output_stage - 1  # compute
                data_dim = 8**self.data_stage - 1
                highs_volume = torch.zeros(
                    (
                        batch_size,
                        self.shape_list[-1][0],
                        self.shape_list[-1][1],
                        self.shape_list[-1][2],
                        data_dim,
                    )
                ).to(
                    self.low.device
                )  # output high volumes

                ### extend batch_size to the indices
                batch_pad = (
                    torch.arange(batch_size, device=self.highs_indices.device)
                    .unsqueeze(1)
                    .repeat((1, self.highs_indices.size(1)))
                    .unsqueeze(2)
                    .long()
                )  # B * P * 1
                high_indices_filled = torch.cat(
                    (batch_pad, self.highs_indices), dim=2
                )  # B * P * 4

                ### fill the values
                high_indices_filled = high_indices_filled.reshape((-1, 4))
                highs_volume[
                    high_indices_filled[:, 0],
                    high_indices_filled[:, 1],
                    high_indices_filled[:, 2],
                    high_indices_filled[:, 3],
                    :,
                ] = self.highs_values.reshape((-1, data_dim))

                ### remove those unwanted dim
                highs_volume = highs_volume[..., -output_dim:]

                ### change back to channel first and concat with low
                highs_volume = torch.permute(highs_volume, (0, 4, 1, 2, 3))
                wavelet_volume = torch.cat((self.low, highs_volume), dim=1)

                return wavelet_volume

    def convert_low_highs(self):

        ### compute the volume if not given
        if self.wavelet_volume is None:
            wavelet_volume = self.convert_wavelet_volume()
        else:
            wavelet_volume = self.wavelet_volume

        ## extract part
        low = wavelet_volume[:, :1]  ## extract low volume

        if self.output_stage == 0:  # fill with empty zeros
            batch_size = low.size(0)
            highs = batch_extract_highs_from_values(
                output_stage=self.output_stage,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
                device=low.device,
                batch_size=batch_size,
            )
        else:
            highs_volume = wavelet_volume[:, 1:]

            output_dim = 8**self.output_stage - 1

            highs_indices = extract_full_indices(
                device=wavelet_volume.device,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
            )  # N * 511 * 4 (N = 46^3)
            highs_indices = highs_indices[:, -output_dim:]  # N * O * 4 (N = 46^3)

            # padding
            batch_size = wavelet_volume.size(0)
            highs_indices = padding_high_indices(
                batch_size, highs_indices
            )  # B * N * O * 5

            ## convert high volume to channel list
            highs_volume = torch.permute(highs_volume, (0, 2, 3, 4, 1))
            highs_values = highs_volume.view(batch_size, -1, output_dim)

            ## obtain the highs
            highs = batch_extract_highs_from_values(
                output_stage=self.output_stage,
                max_depth=self.max_depth,
                shape_list=self.shape_list,
                device=wavelet_volume.device,
                highs_indices=highs_indices,
                highs_values=highs_values,
                batch_size=batch_size,
            )

        return low, highs


def batch_extract_highs_from_values(
    output_stage,
    max_depth,
    shape_list,
    device,
    batch_size,
    highs_indices=None,
    highs_values=None,
):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        current_stage = (max_depth - 1) - idx

        padding_size = (
            shape_list[-1][0] * (2**current_stage) - shape_list[idx + 1][0]
        ) // 2
        high_new = (
            torch.zeros(
                (
                    batch_size,
                    1,
                    7,
                    shape_list[idx + 1][0] + padding_size * 2,
                    shape_list[idx + 1][1] + padding_size * 2,
                    shape_list[idx + 1][2] + padding_size * 2,
                )
            )
            .type(highs_values.type())
            .to(device)
        )

        ### only those in output will be computed
        if current_stage < output_stage:
            assert highs_indices.size(0) == highs_values.size(0)

            ## reshape to fit the shape
            indices_len = 7 * ((2**current_stage) ** 3)
            high_values_idx = highs_values[:, :, cnt : cnt + indices_len]
            high_indices_idx = highs_indices[:, :, cnt : cnt + indices_len]
            high_indices_idx = high_indices_idx.reshape((-1, 5)).long()
            high_values_idx = high_values_idx.reshape((-1))

            # set the values
            high_new[
                high_indices_idx[:, 0],
                0,
                high_indices_idx[:, 1],
                high_indices_idx[:, 2],
                high_indices_idx[:, 3],
                high_indices_idx[:, 4],
            ] = high_values_idx
            cnt += indices_len

        # remove padding
        high_new = high_new[
            :,
            :,
            :,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            padding_size : high_new.size(5) - padding_size,
        ]
        highs_recon.append(high_new)

    return highs_recon


def padding_high_indices(batch_size, highs_indices):
    batch_size_pad = torch.arange(batch_size, device=highs_indices.device)
    for _ in range(3):  # result in B * 1 * 1 * 1
        batch_size_pad = batch_size_pad.unsqueeze(1)
    batch_size_pad = batch_size_pad.repeat(
        1, highs_indices.size(0), highs_indices.size(1), 1
    )  # B * N * O * 1 (N = 46^3)
    highs_indices = highs_indices.unsqueeze(0).repeat(
        batch_size, 1, 1, 1
    )  # B * N * O * 4 (N = 46^3)
    highs_indices = torch.cat((batch_size_pad, highs_indices), dim=-1)
    return highs_indices


def extract_wavelet_coefficients(
    data_item, spatial_shapes, max_depth, keep_level, device=None
):
    # data_item : (N, 1 + 7 * D, low, low, low)

    low = data_item[:, 0:1]
    highs = []
    for j in range(max_depth):
        spatial_shape = spatial_shapes[j + 1]
        if j < keep_level:
            high = np.zeros(
                (1, 1, 7, spatial_shape[0], spatial_shape[1], spatial_shape[2])
            )
        else:
            high = data_item[
                :, (j - keep_level) * 7 + 1 : (j - keep_level + 1) * 7 + 1
            ]  # extract coefficients
            high = high[:, None]
        highs.append(high)

    low = torch.from_numpy(low).float()
    highs = [torch.from_numpy(high).float() for high in highs]

    if device is not None:
        low = low.to(device)
        highs = [high.to(device) for high in highs]

    return low, highs


def multi_dim_argsort(tensor, descending=True):
    tensor_size = list(tensor.size())
    tensor = tensor.reshape((-1))
    indices = torch.argsort(tensor, descending=descending)
    result_indices = []
    for size_i in tensor_size[::-1]:
        indices_i = indices % size_i
        indices = indices // size_i
        indices = indices.long()
        result_indices.append(indices_i.unsqueeze(1))

    result_indices = torch.cat(result_indices[::-1], dim=1)
    return result_indices


def extract_highs_from_colors(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2**order_expand) ** 3)
        high_values_idx = high_values[:, cnt : cnt + indices_len]
        high_indices_idx = high_indices[:, cnt : cnt + indices_len]

        padding_size = (
            shape_list[-1][0] * (2**order_expand) - shape_list[idx + 1][0]
        ) // 2
        high_new = torch.zeros(
            (
                1,
                7,
                shape_list[idx + 1][0] + padding_size * 2,
                shape_list[idx + 1][1] + padding_size * 2,
                shape_list[idx + 1][2] + padding_size * 2,
                3,
            )
        ).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1, 3))

        # set the values
        high_new[
            0,
            high_indices_idx[:, 0],
            high_indices_idx[:, 1],
            high_indices_idx[:, 2],
            high_indices_idx[:, 3],
            :,
        ] = high_values_idx

        high_new = high_new[
            :,
            :,
            padding_size : high_new.size(2) - padding_size,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            :,
        ]  # remove padding
        high_new = torch.permute(high_new, (0, 5, 1, 2, 3, 4))
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon


def extract_highs_from_values(high_indices, high_values, max_depth, shape_list):
    cnt = 0
    highs_recon = []
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        indices_len = 7 * ((2**order_expand) ** 3)
        high_values_idx = high_values[:, cnt : cnt + indices_len]
        high_indices_idx = high_indices[:, cnt : cnt + indices_len]

        padding_size = (
            shape_list[-1][0] * (2**order_expand) - shape_list[idx + 1][0]
        ) // 2
        high_new = torch.zeros(
            (
                1,
                1,
                7,
                shape_list[idx + 1][0] + padding_size * 2,
                shape_list[idx + 1][1] + padding_size * 2,
                shape_list[idx + 1][2] + padding_size * 2,
            )
        ).to(high_indices_idx.device)

        ## reshape to fit the shape
        high_indices_idx = high_indices_idx.reshape((-1, 4)).long()
        high_values_idx = high_values_idx.reshape((-1))

        # set the values
        high_new[
            0,
            0,
            high_indices_idx[:, 0],
            high_indices_idx[:, 1],
            high_indices_idx[:, 2],
            high_indices_idx[:, 3],
        ] = high_values_idx

        high_new = high_new[
            :,
            :,
            :,
            padding_size : high_new.size(3) - padding_size,
            padding_size : high_new.size(4) - padding_size,
            padding_size : high_new.size(5) - padding_size,
        ]  # remove padding
        highs_recon.append(high_new)
        cnt += indices_len
    return highs_recon


def pad_with_batch_idx(tensor):
    batch_size = tensor.size(0)
    padding_tensor = (
        torch.arange(batch_size, device=tensor.device)
        .long()
        .unsqueeze(1)
        .repeat(1, tensor.size(1))
        .unsqueeze(2)
    )
    padded_tensor = torch.cat((padding_tensor, tensor), dim=-1)
    return padded_tensor


def create_coordinates(resolution, space_range, channel_dim=7):
    channels_samples = np.linspace(0, channel_dim - 1, channel_dim)
    dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
    ch, x, y, z = np.meshgrid(
        channels_samples, dimensions_samples, dimensions_samples, dimensions_samples
    )
    ch, x, y, z = (
        ch[:, :, :, :, np.newaxis],
        x[:, :, :, :, np.newaxis],
        y[:, :, :, :, np.newaxis],
        z[:, :, :, :, np.newaxis],
    )
    coordinates = np.concatenate((ch, x, y, z), axis=4)
    coordinates = coordinates.reshape((-1, 4))
    coordinates = torch.from_numpy(coordinates).long()
    return coordinates


def extract_full_indices(device, max_depth, shape_list):
    highs_full_indices = []
    highs_full_indices_last = create_coordinates(
        shape_list[-1][0], space_range=(0, shape_list[-1][0] - 1), channel_dim=1
    ).to(device)
    for idx in range(max_depth):
        order_expand = (max_depth - 1) - idx
        highs_full_indices_idx = compute_level_indices(
            highs_full_indices_last, order_expand=order_expand
        )
        highs_full_indices_idx = highs_full_indices_idx.reshape(
            (-1, 7 * ((2**order_expand) ** 3), 4)
        )
        highs_full_indices.append(highs_full_indices_idx)
    highs_full_indices = torch.cat(highs_full_indices, dim=1)
    return highs_full_indices


def compute_level_indices(indices_keep_last, order_expand):
    indices_grid = create_coordinates(
        2**order_expand, (0, 2**order_expand - 1), channel_dim=7
    ).to(indices_keep_last.device)
    mutiplier = (
        torch.from_numpy(
            np.array([1, 2**order_expand, 2**order_expand, 2**order_expand])
        )
        .unsqueeze(0)
        .to(indices_keep_last.device)
    )
    indices_keep = (
        indices_grid.unsqueeze(0) + indices_keep_last.unsqueeze(1) * mutiplier
    )
    return indices_keep
