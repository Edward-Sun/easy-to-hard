# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor


try:
    from apex.normalization.fused_layer_norm import FusedRMSNormFunction

    # print(
    #     "`apex` is installed. You can use fused RMSNorm by set_global_compile_mode(False)."
    # )
except ImportError as e:
    FusedRMSNormFunction = None
    # print("`apex` is not installed. Reverting to non-fused RMSNorm.")

# whether to use fused RMSNorm or not (default: no)
_GLOBAL_IN_COMPILE_MODE = True


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class FrozenEmbedding(nn.Module):
    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self.vocab_start_index = None
        self.vocab_end_index = None
        self.num_embeddings_per_partition = None
        self.register_buffer(
            "weight", torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.num_embeddings_per_partition is None:
            return F.embedding(
                input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        else:
            # Build the mask.
            print("vocab_start_index", self.vocab_start_index)
            print("vocab_end_index", self.vocab_end_index)
            input_mask = (input < self.vocab_start_index) | (
                input >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
            # Get the embeddings.
            output_parallel = F.embedding(
                masked_input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            # Mask the output embedding.
            output_parallel[input_mask, :] = 0.0
            return output_parallel

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2.0:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)


class FrozenRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

        global _GLOBAL_IN_COMPILE_MODE
        self.in_compile_mode = _GLOBAL_IN_COMPILE_MODE

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


class FrozenLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.register_buffer("bias", torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_buffer("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
