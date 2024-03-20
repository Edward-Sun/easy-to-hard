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

"""Model classes that are shared across different algorithms.

WARNING:
    Do not tamper with the state_dict function for any of these classes.
    If you tamper, make sure the keys are the same, otherwise FSDP will get confused.
"""

import abc
import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import _functional_collectives as funcol

from arguments import Arguments
from models.model import Transformer
from models.tokenizer_utils import AcceleraTokenizer
from models.tp import get_model_parallel_group, compute_vocab_parallel_logprobs

logger = logging.getLogger(__name__)


class Policy(nn.Module, abc.ABC):
    def __init__(
        self,
        args: Arguments,
        base_model: Transformer,
        base_tokenizer: AcceleraTokenizer,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer

        global decode_one_token

        if decode_one_token is None:
            if self.args.compile:
                decode_one_token = torch.compile(
                    _decode_one_token, mode="default", fullgraph=True
                )
            else:
                decode_one_token = _decode_one_token

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(
            self._respond(queries, query_attn_masks, temperature, num_return_sequences)
        )

    @abc.abstractmethod
    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


class AutoregressivePolicy(Policy):
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        # TODO(lxuechen): Refactor attention mask. Here query_attn_masks overrides padding-based attention mask.
        if mode == "respond":
            return self.respond(queries, query_attn_masks, temperature)

        assert responses is not None
        if temperature is None:
            temperature = self.args.temperature
        input_ids = torch.cat([queries, responses], dim=1)
        attention_mask = input_ids.ne(self.base_tokenizer.pad_id)
        attention_mask[:, : queries.size(1)] = query_attn_masks

        batch_size, T = input_ids.size(0), input_ids.size(1)
        device = input_ids.device

        inputs, shifts = prepare_right_pad_sequences(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.base_tokenizer.pad_id,
        )
        input_pos = torch.arange(0, T, device=device)

        logits = self.base_model(inputs, input_pos, fully_causal=True).float()
        logits = restore_from_right_pad_sequences(logits, shifts)

        original_logits = logits[:, -self.args.target_max_len - 1 : -1]
        logits = original_logits / temperature
        labels = input_ids[:, -self.args.target_max_len :]

        with torch.autocast(device_type="cuda", enabled=False):
            dtype_logits = logits.float()
            if self.base_model.vocab_parallel:
                logprobs = compute_vocab_parallel_logprobs(
                    dtype_logits, labels, ignore_index=self.base_tokenizer.pad_id
                )
            else:
                logprobs = compute_logprobs(
                    dtype_logits, labels, ignore_index=self.base_tokenizer.pad_id
                )
            entropies = -(
                dtype_logits.softmax(dim=-1) * dtype_logits.log_softmax(dim=-1)
            ).sum(dim=-1)
            non_ignore_mask = labels.ne(self.base_tokenizer.pad_id).to(
                dtype=entropies.dtype
            )
            reg_entropies = entropies * non_ignore_mask
        return dict(
            logprobs=logprobs,
            entropies=entropies,
            reg_entropies=reg_entropies,
            reg_entropies_weight=non_ignore_mask,
        )

    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        del num_return_sequences  # Unused.

        unwrapped_base_model = self.base_model
        if isinstance(self.base_model, DDP):
            unwrapped_base_model = self.base_model.module

        B, T = queries.size(0), queries.size(1)
        T_new = T + self.args.target_max_len
        assert T_new <= unwrapped_base_model.config.block_size

        device, dtype = queries.device, queries.dtype
        with torch.device(device):
            unwrapped_base_model.setup_caches(max_batch_size=B, max_seq_length=T_new)

        if temperature is None:
            temperature = self.args.temperature

        # create an zero's tensor of the expected final shape and fill in the current tokens
        empty = torch.zeros((B, T_new), dtype=dtype, device=device)
        empty[:, :T] = queries
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        sampling_kwargs = dict(
            temperature=temperature,
            top_k=50,
        )

        shifts = prepare_left_pad_mask_pos(
            queries,
            attention_mask=query_attn_masks,
            pad_token_id=self.base_tokenizer.pad_id,
        )

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            next_token = prefill(
                unwrapped_base_model, queries, input_pos, shifts, **sampling_kwargs
            )

        seq[:, T] = next_token.view(B)

        input_pos = torch.tensor([T], device=device, dtype=torch.int)

        generated_tokens, _, _ = decode_n_tokens(
            unwrapped_base_model,
            next_token.view(B, -1),
            input_pos,
            shifts,
            self.args.target_max_len - 1,
            self.base_tokenizer.eos_id,
            **sampling_kwargs,
        )

        generated_tokens = torch.cat(generated_tokens, dim=-1).view(B, -1)
        seq[:, T + 1 : T + 1 + generated_tokens.size(1)] = generated_tokens
        assert seq[:, T:].size(1) == self.args.target_max_len

        return dict(
            responses=seq[:, T:]
        )  # Size (bsz * num_return_sequences, response_len).


class Value(nn.Module, abc.ABC):
    def __init__(
        self,
        args: Arguments,
        base_model: Transformer,
        base_tokenizer: AcceleraTokenizer,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.initialized = False

    @abc.abstractmethod
    def forward(
        self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class AutoregressiveValue(Value):
    def forward(
        self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor
    ) -> Dict[str, Tensor]:
        assert self.initialized, "Value model must be initialized before forward pass."

        sequences = torch.cat([queries, responses], dim=1)
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_id)
        sequence_attn_masks[:, : queries.size(1)] = query_attn_masks

        B, T = sequences.size(0), sequences.size(1)
        inputs, shifts = prepare_right_pad_sequences(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            pad_token_id=self.base_tokenizer.pad_id,
        )

        device = queries.device
        values = self.base_model(
            inputs, torch.arange(0, T, device=device), fully_causal=True
        )
        values = values.mean(dim=-1)

        values = restore_from_right_pad_sequences(values, shifts)
        values = values[:, queries.size(1) - 1 : -1]
        assert values.size(1) == responses.size(1)

        return dict(values=values)


def make_policy_with_base_model(
    args: Arguments,
    base_model: Transformer,
    base_tokenizer: AcceleraTokenizer,
) -> AutoregressivePolicy:
    policy = AutoregressivePolicy(args, base_model, base_tokenizer)
    return policy


def make_value_with_base_model(
    args: Arguments,
    base_model: Transformer,
    base_tokenizer: AcceleraTokenizer,
) -> AutoregressiveValue:
    value_model = AutoregressiveValue(args, base_model, base_tokenizer)
    value_model.initialized = True
    return value_model


def prepare_right_pad_sequences(input_ids, attention_mask=None, pad_token_id=0):
    # Assuming '0' is the padding value
    if attention_mask is None:
        attention_mask = input_ids != pad_token_id
    # torch.argmax: If there are multiple maximal values
    # then the indices of the first maximal value are returned.
    shifts = torch.argmax(attention_mask.to(torch.int), dim=1)

    # if (shifts == 0).all():
    #     return input_ids, None

    ind0 = torch.arange(input_ids.size(0), device=input_ids.device)
    ind0 = ind0[:, None].expand(-1, input_ids.size(1))
    ind1 = torch.arange(input_ids.size(1), device=input_ids.device)
    ind1 = ind1[None, :].expand(input_ids.size(0), -1)

    rolled_input_ids = input_ids[
        ind0, (ind1 + shifts[:, None] + input_ids.size(1)) % input_ids.size(1)
    ]
    return rolled_input_ids, shifts


def restore_from_right_pad_sequences(inputs, shifts):
    if shifts is None:
        return inputs

    ind0 = torch.arange(inputs.size(0), device=inputs.device)
    ind0 = ind0[:, None].expand(-1, inputs.size(1))
    ind1 = torch.arange(inputs.size(1), device=inputs.device)
    ind1 = ind1[None, :].expand(inputs.size(0), -1)

    rolled_inputs = inputs[
        ind0, (ind1 - shifts[:, None] - inputs.size(1)) % inputs.size(1)
    ]
    return rolled_inputs


def prepare_left_pad_mask_pos(input_ids, attention_mask=None, pad_token_id=0):
    # Assuming '0' is the padding value
    if attention_mask is None:
        attention_mask = input_ids != pad_token_id
    shifts = torch.argmax(attention_mask.to(torch.int), dim=1)
    return shifts


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
    # return torch.argmax(probs_sort, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).view(-1, 1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits, vocab_parallel, temperature: float = 1.0, top_k: Optional[int] = None
):
    with torch.autocast(device_type="cuda", enabled=False):
        logits = logits[:, -1].float()

        if vocab_parallel:
            logits = funcol.all_gather_tensor(
                logits, gather_dim=-1, group=get_model_parallel_group()
            )

        probs = logits_to_probs(logits, temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)[0]


def _decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


decode_one_token = None


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    num_new_tokens: int,
    eos_id: Optional[int] = None,
    **sampling_kwargs,
):
    eos_flag = None
    if eos_id is not None:
        eos_flag = torch.zeros_like(
            cur_token, dtype=torch.bool, device=cur_token.device
        )

    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, left_pad_mask_pos, **sampling_kwargs
            )
        input_pos += 1
        new_tokens.append(next_token.clone().view(-1, 1))
        new_probs.append(next_prob.clone().view(-1, 1))
        cur_token = next_token.view(-1, 1)

        if eos_flag is not None:
            eos_flag = eos_flag | (next_token == eos_id)

        if eos_flag is not None and eos_flag.all():
            break

    return new_tokens, new_probs, i


def compute_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )
