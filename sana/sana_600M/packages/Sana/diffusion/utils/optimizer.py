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

import math
from typing import Callable, Optional, Tuple

import torch
from came_pytorch import CAME
from mmcv import Config
from mmcv.runner import OPTIMIZER_BUILDERS, OPTIMIZERS, DefaultOptimizerConstructor
from mmcv.runner import build_optimizer as mm_build_optimizer
from mmcv.utils import _BatchNorm, _InstanceNorm
from torch.nn import GroupNorm, LayerNorm
from torch.optim.optimizer import Optimizer

from .logger import get_root_logger


def auto_scale_lr(effective_bs, optimizer_cfg, rule="linear", base_batch_size=256):
    assert rule in ["linear", "sqrt"]
    logger = get_root_logger()
    # scale by world size
    if rule == "sqrt":
        scale_ratio = math.sqrt(effective_bs / base_batch_size)
    elif rule == "linear":
        scale_ratio = effective_bs / base_batch_size
    optimizer_cfg["lr"] *= scale_ratio
    logger.info(
        f'Automatically adapt lr to {optimizer_cfg["lr"]:.5f} (using {rule} scaling rule).'
    )
    return scale_ratio


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix="", is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module

        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        # sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", 1.0)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", 1.0)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

        for name, param in module.named_parameters(recurse=False):
            base_lr = self.base_lr
            if name == "bias" and not (is_norm or is_dcn_module):
                base_lr *= bias_lr_mult

            # apply weight decay policies
            base_wd = self.base_wd
            if self.base_wd is not None:
                # norm decay
                if is_norm:
                    base_wd *= norm_decay_mult
                # bias lr and decay
                elif name == "bias" and not is_dcn_module:
                    # TODO: current bias_decay_mult will have affect on DCN
                    base_wd *= bias_decay_mult

            param_group = {"params": [param]}
            if not param.requires_grad:
                param_group["requires_grad"] = False
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                logger = get_root_logger()
                logger.warn(
                    f"{prefix} is duplicate. It is skipped since "
                    f"bypass_duplicate={bypass_duplicate}"
                )
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in custom_keys:
                if isinstance(key, tuple):
                    scope, key_name = key
                else:
                    scope, key_name = None, key
                if scope is not None and scope not in f"{prefix}":
                    continue
                if key_name in f"{prefix}.{name}":
                    is_custom = True
                    if "lr_mult" in custom_keys[key]:
                        # if 'base_classes' in f'{prefix}.{name}' or 'attn_base' in f'{prefix}.{name}':
                        #     param_group['lr'] = self.base_lr
                        # else:
                        param_group["lr"] = self.base_lr * custom_keys[key]["lr_mult"]
                    elif "lr" not in param_group:
                        param_group["lr"] = base_lr
                    if self.base_wd is not None:
                        if "decay_mult" in custom_keys[key]:
                            param_group["weight_decay"] = (
                                self.base_wd * custom_keys[key]["decay_mult"]
                            )
                        elif "weight_decay" not in param_group:
                            param_group["weight_decay"] = base_wd

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if base_lr != self.base_lr:
                    param_group["lr"] = base_lr
                if base_wd != self.base_wd:
                    param_group["weight_decay"] = base_wd
            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(
                params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module
            )


def build_optimizer(model, optimizer_cfg):
    # default parameter-wise config
    logger = get_root_logger()

    if hasattr(model, "module"):
        model = model.module
    # set optimizer constructor
    optimizer_cfg.setdefault("constructor", "MyOptimizerConstructor")
    # parameter-wise setting: cancel weight decay for some specific modules
    custom_keys = dict()
    for name, module in model.named_modules():
        if hasattr(module, "zero_weight_decay"):
            custom_keys.update(
                {(name, key): dict(decay_mult=0) for key in module.zero_weight_decay}
            )

    paramwise_cfg = Config(dict(cfg=dict(custom_keys=custom_keys)))
    given_cfg = optimizer_cfg.get("paramwise_cfg")
    if given_cfg:
        paramwise_cfg.merge_from_dict(dict(cfg=given_cfg))
    optimizer_cfg["paramwise_cfg"] = paramwise_cfg.cfg
    # build optimizer
    optimizer = mm_build_optimizer(model, optimizer_cfg)

    weight_decay_groups = dict()
    lr_groups = dict()
    for group in optimizer.param_groups:
        if not group.get("requires_grad", True):
            continue
        lr_groups.setdefault(group["lr"], []).append(group)
        weight_decay_groups.setdefault(group["weight_decay"], []).append(group)

    learnable_count, fix_count = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            learnable_count += 1
        else:
            fix_count += 1
    fix_info = f"{learnable_count} are learnable, {fix_count} are fix"
    lr_info = "Lr group: " + ", ".join(
        [f"{len(group)} params with lr {lr:.5f}" for lr, group in lr_groups.items()]
    )
    wd_info = "Weight decay group: " + ", ".join(
        [
            f"{len(group)} params with weight decay {wd}"
            for wd, group in weight_decay_groups.items()
        ]
    )
    opt_info = f"{optimizer.__class__.__name__} Optimizer: total {len(optimizer.param_groups)} param groups, {fix_info}. {lr_info}; {wd_info}."
    logger.info(opt_info)

    return optimizer


@OPTIMIZERS.register_module()
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @staticmethod
    def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
        # stepweight decay
        p.data.mul_(1 - lr * wd)

        # weight update
        update = exp_avg.clone().lerp_(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)

    @staticmethod
    def exists(val):
        return val is not None

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if self.exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: self.exists(p.grad), group["params"]):

                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss


@OPTIMIZERS.register_module()
class CAMEWrapper(CAME):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
