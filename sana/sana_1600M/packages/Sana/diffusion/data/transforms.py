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

import torchvision.transforms as T

TRANSFORMS = dict()


def register_transform(transform):
    name = transform.__name__
    if name in TRANSFORMS:
        raise RuntimeError(f"Transform {name} has already registered.")
    TRANSFORMS.update({name: transform})


def get_transform(type, resolution):
    transform = TRANSFORMS[type](resolution)
    transform = T.Compose(transform)
    transform.image_size = resolution
    return transform


@register_transform
def default_train(n_px):
    transform = [
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(n_px),  # Image.BICUBIC
        T.CenterCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
    return transform
