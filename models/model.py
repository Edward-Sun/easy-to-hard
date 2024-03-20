# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import torch.utils.checkpoint as activation_checkpoint

import models.frozen_layers as frozen_layers
from models.frozen_layers import (
    FrozenEmbedding,
    FrozenLinear,
    FrozenRMSNorm,
)


try:
    from apex.normalization.fused_layer_norm import FusedRMSNormFunction

    # print(
    #     "`apex` is installed. You can use fused RMSNorm by set_global_compile_mode(False)."
    # )
except ImportError as e:
    FusedRMSNormFunction = None
    # print("`apex` is not installed. Reverting to non-fused RMSNorm.")


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config in str(name).upper() or config in str(name)
        ]
        assert len(config) >= 1, name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "llemma-7b": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    "deepseek-math-7b": dict(
        block_size=4096,
        vocab_size=102400,
        n_layer=30,
        n_head=32,
        dim=4096,
        intermediate_size=11008,
        rope_base=10000,
        norm_eps=1e-6,
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        block_size=4096,
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[: k_val.size(0), :, input_pos] = k_val
        v_out[: k_val.size(0), :, input_pos] = v_val
        return k_out[: k_val.size(0)], v_out[: k_val.size(0)]


class Transformer(nn.Module):
    def __init__(
        self,
        config: ModelArgs,
        freeze_tok_embeddings: bool = False,
        freeze_norm: bool = False,
        freeze_output: bool = False,
        vocab_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = (
            FrozenEmbedding(config.vocab_size, config.dim)
            if freeze_tok_embeddings
            else nn.Embedding(config.vocab_size, config.dim)
        )

        self.layers = nn.ModuleList(
            TransformerBlock(config, freeze_norm=freeze_norm)
            for _ in range(config.n_layer)
        )
        self.norm = (
            FrozenRMSNorm(config.dim, eps=config.norm_eps)
            if freeze_norm
            else RMSNorm(config.dim, eps=config.norm_eps)
        )
        self.output = (
            FrozenLinear(config.dim, config.vocab_size, bias=False)
            if freeze_output
            else nn.Linear(config.dim, config.vocab_size, bias=False)
        )

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.kv_cache_enabled = False
        self.vocab_parallel = False

    def setup_caches(self, max_batch_size, max_seq_length, kv_cache=True):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            if self.kv_cache_enabled or not kv_cache:
                return

        if (self.max_seq_length > 0 and self.max_seq_length < max_seq_length) or (
            self.max_batch_size > 0 and self.max_batch_size < max_batch_size
        ):
            raise ValueError(
                "Cannot increase the size of the cache after compiled. "
                "Please create a new model with the desired cache size."
            )

        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            if kv_cache:
                b.attention.kv_cache = KVCache(
                    max_batch_size, max_seq_length, self.config.n_local_heads, head_dim
                )
                self.kv_cache_enabled = True
            else:
                b.attention.kv_cache = None
        self.kv_cache_enabled = kv_cache

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.self_mask = torch.eye(self.max_seq_length, dtype=torch.bool)

    def forward(
        self,
        idx: Tensor,
        input_pos: Optional[Tensor] = None,
        left_pad_mask_pos: Optional[Tensor] = None,
        fully_causal: bool = False,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        assert not (fully_causal and left_pad_mask_pos is not None), "Invalid mask"
        mask = self.causal_mask[None, None, input_pos]

        if left_pad_mask_pos is not None:
            pad_mask = torch.arange(mask.size(-1), device=mask.device).view(
                1, -1
            ) >= left_pad_mask_pos.view(-1, 1)
            mask = torch.logical_and(mask, pad_mask[:, None, None, :].contiguous())
            mask = torch.logical_or(mask, self.self_mask[None, None, input_pos])

        x = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[input_pos].to(dtype=x.dtype)

        for i, layer in enumerate(self.layers):
            if self.training:
                x = activation_checkpoint.checkpoint(
                    layer,
                    x,
                    input_pos,
                    freqs_cis,
                    mask,
                    fully_causal,
                    use_reentrant=False,
                )
            else:
                x = layer(x, input_pos, freqs_cis, mask, fully_causal)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str, **kwargs):
        return cls(ModelArgs.from_name(name), **kwargs)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, freeze_norm: bool = False) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = (
            FrozenRMSNorm(config.dim, config.norm_eps)
            if freeze_norm
            else RMSNorm(config.dim, config.norm_eps)
        )
        self.attention_norm = (
            FrozenRMSNorm(config.dim, config.norm_eps)
            if freeze_norm
            else RMSNorm(config.dim, config.norm_eps)
        )

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Union[Tensor, str],
        fully_causal: bool = False,
    ) -> Tensor:
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, mask, input_pos, fully_causal
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Union[Tensor, str],
        input_pos: Optional[Tensor] = None,
        fully_causal: bool = False,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if not fully_causal:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if q.size(2) == k.size(2) and fully_causal:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.in_compile_mode = frozen_layers._GLOBAL_IN_COMPILE_MODE

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.in_compile_mode or FusedRMSNormFunction is None:
            with torch.autocast(device_type="cuda", enabled=False):
                output = self._norm(x.float()).to(dtype=x.dtype)
            return output * self.weight
        else:
            with torch.autocast(device_type="cuda", enabled=False):
                output = FusedRMSNormFunction.apply(
                    x,
                    self.weight.size(),
                    self.eps,
                    False,
                )
            return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def set_global_compile_mode(mode: bool):
    frozen_layers._GLOBAL_IN_COMPILE_MODE = mode
