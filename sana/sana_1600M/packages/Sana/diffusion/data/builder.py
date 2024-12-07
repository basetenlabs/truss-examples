# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time

from mmcv import Registry, build_from_cfg
from termcolor import colored
from torch.utils.data import DataLoader

from diffusion.data.transforms import get_transform
from diffusion.utils.logger import get_root_logger

DATASETS = Registry("datasets")

DATA_ROOT = "data"


def set_data_root(data_root):
    global DATA_ROOT
    DATA_ROOT = data_root


def get_data_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return os.path.join(DATA_ROOT, data_dir)


def get_data_root_and_path(data_dir):
    if os.path.isabs(data_dir):
        return data_dir
    global DATA_ROOT
    return DATA_ROOT, os.path.join(DATA_ROOT, data_dir)


def build_dataset(cfg, resolution=224, **kwargs):
    logger = get_root_logger()

    dataset_type = cfg.get("type")
    logger.info(f"Constructing dataset {dataset_type}...")
    t = time.time()
    transform = cfg.pop("transform", "default_train")
    transform = get_transform(transform, resolution)
    dataset = build_from_cfg(
        cfg,
        DATASETS,
        default_args=dict(transform=transform, resolution=resolution, **kwargs),
    )
    logger.info(
        f"{colored(f'Dataset {dataset_type} constructed: ', 'green', attrs=['bold'])}"
        f"time: {(time.time() - t):.2f} s, length (use/ori): {len(dataset)}/{dataset.ori_imgs_nums}"
    )
    return dataset


def build_dataloader(dataset, batch_size=256, num_workers=4, shuffle=True, **kwargs):
    if "batch_sampler" in kwargs:
        dataloader = DataLoader(
            dataset,
            batch_sampler=kwargs["batch_sampler"],
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )
    return dataloader
