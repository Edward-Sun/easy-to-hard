# Copyright 2024 The GPT-Accelera Team
# Copyright 2023 The Self-Align Team
# Copyright 2023 The Alpaca Team
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

from dataclasses import dataclass
import math
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.model import ModelArgs, Transformer


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


@dataclass
class RewardArgs:
    backbone_args: ModelArgs

    @classmethod
    def from_name(cls, name: str):
        return cls(backbone_args=ModelArgs.from_name(name))


class RewardModel(nn.Module):
    def __init__(self, config: RewardArgs, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.backbone_model = Transformer(config.backbone_args, **kwargs)

    def forward(
        self,
        idx: Tensor,
        eos_pos: Optional[Tensor] = None,
    ) -> Tensor:
        input_pos = torch.arange(0, idx.size(-1), device=idx.device)
        rewards = self.backbone_model(idx, input_pos=input_pos, fully_causal=True)
        rewards = rewards.mean(dim=-1)

        if eos_pos is not None:
            eos_pos = eos_pos.unsqueeze(-1)
            rewards = batch_select(rewards, eos_pos).squeeze(-1)

        return rewards

    @classmethod
    def from_name(cls, name: str, **kwargs):
        return cls(RewardArgs.from_name(name), **kwargs)


def apply_reward_modeling_head(
    transformer: Transformer, requires_grad=False, init_sceheme="zeros"
):
    output_module = transformer.output
    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)

    # Temp fix due to https://github.com/pytorch/pytorch/issues/106951
    reward_head_weight = torch.zeros_like(output_module.weight)[:2, :]
    if init_sceheme == "zeros":
        output_module.weight = nn.Parameter(
            reward_head_weight,
            requires_grad=requires_grad,
        )
    elif init_sceheme == "semantic":
        # ['### Preferred Output is '] [835, 4721, 14373, 10604, 338, 29871]
        # ['### Preferred Output is 1.'] [835, 4721, 14373, 10604, 338, 29871, 29896, 29889]
        # ['### Preferred Output is 2.'] [835, 4721, 14373, 10604, 338, 29871, 29906, 29889]
        token_1_id = 29896
        token_2_id = 29906
        reward_head_weight[0, :] = output_module.weight[token_2_id, :]
        reward_head_weight[1, :] = -output_module.weight[token_1_id, :]
        output_module.weight = nn.Parameter(
            reward_head_weight,
            requires_grad=requires_grad,
        )
    elif init_sceheme == "random":
        generator = torch.Generator(device=reward_head_weight.device)
        generator.manual_seed(42)
        nn.init.kaiming_uniform_(
            reward_head_weight, a=math.sqrt(5), generator=generator
        )
        output_module.weight = nn.Parameter(
            reward_head_weight * math.sqrt(2.0),
            requires_grad=requires_grad,
        )
    else:
        raise ValueError(f"Unknown init_scheme: {init_sceheme}")
    setattr(output_module, "out_features", 2)


def compute_pairwise_reward_modeling_loss(model, inputs, return_outputs=False):
    # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
    # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
    # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
    input_ids, eos_pos, index_0, index_1, choice = unpack_dict(
        inputs, keys=("input_ids", "eos_pos", "index_0", "index_1", "choice")
    )
    num_candidates, num_pairs = input_ids.size(1), choice.size(1)
    input_ids_flat = einops.rearrange(input_ids, "b c l -> (b c) l")
    eos_pos_flat = einops.rearrange(eos_pos, "b c -> (b c)")
    input_pos_flat = torch.arange(
        0, input_ids_flat.size(-1), device=input_ids_flat.device
    )
    outputs = model(
        input_ids=input_ids_flat,
        input_pos=input_pos_flat,
        eos_pos=eos_pos_flat,
    )
    rewards_flat = outputs.rewards
    rewards = einops.rearrange(
        rewards_flat, "(b c) -> b c", c=num_candidates
    )  # Size: (bsz, num_candidates).

    rewards_0, rewards_1 = tuple(
        batch_select(rewards, index) for index in (index_0, index_1)
    )  # Size: (bsz, num_pairs).
    logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
    # Type casting of `choice` is due to amp.autocast context manager.
    loss = F.binary_cross_entropy_with_logits(
        logits, choice.to(logits.dtype), reduction="mean"
    )
    return (loss, dict(logits=logits)) if return_outputs else loss


def compute_pairwise_reward_modeling_metrics(
    predictions: torch.Tensor, label_ids: torch.Tensor
) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(predictions).squeeze(-1)
    labels = torch.tensor(label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
    )
