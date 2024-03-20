# Copyright 2024 The GPT-Accelera Team
# Copyright 2023 The Alpaca Team
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from training_utils.memory_efficient_adam import MemoryEfficientAdamW
from arguments import Arguments

from models.model import TransformerBlock
from models.tp import get_data_parallel_group, get_data_parallel_world_size


def create_optimizer(
    args: Arguments,
    model: nn.Module,
    optimizer_cpu_offload: bool = False,
    model_cpu_offload: bool = False,
) -> optim.Optimizer:
    if not model_cpu_offload:
        model_device = next(iter(model.parameters())).device

        optimizer = MemoryEfficientAdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            optim_dtype=args.optim_dtype,
            optim_device=(
                torch.device("cpu") if optimizer_cpu_offload else model_device
            ),
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            fused=True,
        )

    return optimizer


def create_fsdp_model_for_finetune(
    args: Arguments,
    model: nn.Module,
    bf16_all_reduce_upper_bound: int = 16,
) -> FSDP:
    model = FSDP(
        module=model,
        process_group=get_data_parallel_group(),
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                TransformerBlock,
            },
        ),
        mixed_precision=MixedPrecision(
            param_dtype=args.compute_dtype,
            reduce_dtype=(
                torch.float32
                if get_data_parallel_world_size() >= bf16_all_reduce_upper_bound
                else args.compute_dtype
            ),
            keep_low_precision_grads=(args.optim_dtype != torch.float32),
            buffer_dtype=args.compute_dtype,
        ),
        cpu_offload=False,
        use_orig_params=False,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    return model


# https://github.com/huggingface/transformers/blob/976189a6df796a2ff442dd81b022626c840d8c27/src/transformers/optimization.py
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    warmup_start_ratio: float,
    eta_min_ratio: float,
):
    if current_step < num_warmup_steps:
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * float(
            current_step
        ) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return eta_min_ratio + (1.0 - eta_min_ratio) * max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    warmup_start_ratio: float = 0.0,
    eta_min_ratio: float = 0.0,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    assert 0.0 <= warmup_start_ratio <= 1.0, "warmup_start_ratio should be in [0, 1]"
    assert 0.0 <= eta_min_ratio <= 1.0, "eta_min_ratio should be in [0, 1]"

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=warmup_epochs,
        num_training_steps=max_epochs,
        warmup_start_ratio=warmup_start_ratio,
        eta_min_ratio=eta_min_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
