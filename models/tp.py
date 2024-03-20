# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import functools
from functools import partial
from typing import Optional, List, Tuple, Any, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.distributed.fsdp import _runtime_utils as fsdp_utils
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    TrainingState,
    _get_grad_norm,
)

from models.model import Transformer, Attention, FeedForward
from models.quantize import WeightOnlyInt4Linear


# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Model parallel rank of the current process.
_MODEL_PARALLEL_RANK = None
# Data parallel rank of the current process.
_DATA_PARALLEL_RANK = None
# Model parallel size.
_MODEL_PARALLEL_SIZE = None
# Data parallel size.
_DATA_PARALLEL_SIZE = None


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_local():
    return _get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        local_rank = _get_local_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"rank: {rank}, local_rank: {local_rank}, world_size: {world_size}")
    return rank


def _apply_tp_embedding(
    embedding: nn.Embedding,  # actually should be FrozenEmbedding
) -> None:
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    shard_dim, size_attr = 0, "num_embeddings"

    # ensure we can shard evenly
    assert getattr(embedding, size_attr) % world_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    # shard
    sharded_weight = shard(embedding.weight, shard_dim)

    # local_break()
    # assert the sharded weight is already registed as buffer
    assert hasattr(embedding, "weight")
    assert not isinstance(embedding.weight, nn.Parameter)
    embedding.register_buffer("weight", sharded_weight)
    setattr(embedding, size_attr, getattr(embedding, size_attr) // world_size)

    embedding.register_forward_hook(aggregate_forward_tensor_hook)

    global_vocab_size = embedding.num_embeddings
    per_partition_vocab_size = divide_and_check_no_remainder(
        global_vocab_size, world_size
    )
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    embedding.num_embeddings_per_partition = per_partition_vocab_size
    embedding.vocab_start_index = index_f
    embedding.vocab_end_index = index_l


def _apply_tp_linear(
    linear: nn.Linear,
    style: str,
    weight_splits: Optional[List[int]] = None,
    requires_grad: bool = False,
) -> None:
    if weight_splits is None:
        weight_splits = []

    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank]

    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q, k, v), dim=dim)

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        if isinstance(linear, WeightOnlyInt4Linear):
            sharded_weight = shard_qkv(
                linear.weight, shard_dim, [i // 8 for i in weight_splits]
            )
            linear.scales_and_zeros = shard_qkv(
                linear.scales_and_zeros, 1 - shard_dim, weight_splits
            )
        else:
            sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if isinstance(linear, WeightOnlyInt4Linear):
            linear.scales_and_zeros = shard(linear.scales_and_zeros, 1 - shard_dim)
            if style == "rowwise":
                assert (
                    linear.scales_and_zeros.shape[0] * 32
                    == sharded_weight.shape[1]
                    * sharded_weight.shape[2]
                    * sharded_weight.shape[3]
                )
                assert linear.scales_and_zeros.shape[1] == sharded_weight.shape[0] * 8
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    # local_break()
    if isinstance(linear.weight, nn.Parameter):
        linear.weight = nn.Parameter(sharded_weight, requires_grad=requires_grad)
    else:
        assert hasattr(linear, "weight")
        assert not isinstance(linear.weight, nn.Parameter)
        linear.register_buffer("weight", sharded_weight)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)

    # shape info should still be synced
    # assert linear.weight.shape == (linear.out_features, linear.in_features)


def _apply_tp_ffn(
    mlp: FeedForward, requires_grad: bool = False, sequence_parallel: bool = False
) -> None:
    assert hasattr(mlp, "w1")
    assert hasattr(mlp, "w3")
    assert hasattr(mlp, "w2")

    _apply_tp_linear(mlp.w1, "colwise", requires_grad=requires_grad)
    _apply_tp_linear(mlp.w3, "colwise", requires_grad=requires_grad)
    _apply_tp_linear(mlp.w2, "rowwise", requires_grad=requires_grad)

    if not sequence_parallel:
        mlp.register_forward_hook(aggregate_forward_tensor_hook)
        if requires_grad:
            mlp.register_full_backward_hook(aggregate_grad_hook)
    else:
        mlp.register_forward_pre_hook(gather_from_sequence_parallel_region_pre_hook)
        mlp.register_forward_hook(reduce_scatter_to_sequence_parallel_region_hook)


def _apply_tp_attn(
    attn: Attention, requires_grad: bool = False, sequence_parallel: bool = False
) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")

    kv_size = attn.n_local_heads * attn.head_dim
    _apply_tp_linear(
        attn.wqkv, "colwise", [attn.dim, kv_size, kv_size], requires_grad=requires_grad
    )
    _apply_tp_linear(attn.wo, "rowwise", requires_grad=requires_grad)

    # overwrite
    world_size = get_model_parallel_world_size()
    attn.n_head = attn.n_head // world_size
    attn.dim = attn.dim // world_size
    attn.head_dim = attn.dim // attn.n_head
    attn.n_local_heads = attn.n_local_heads // world_size

    if not sequence_parallel:
        attn.register_forward_hook(aggregate_forward_tensor_hook)
        if requires_grad:
            attn.register_full_backward_hook(
                partial(aggregate_grad_hook, num_postional_args=5)
            )
    else:
        attn.register_forward_pre_hook(gather_from_sequence_parallel_region_pre_hook)
        attn.register_forward_hook(reduce_scatter_to_sequence_parallel_region_hook)


def _apply_tp_Transformer(Transformer: Transformer) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = get_model_parallel_world_size()
    Transformer.config.n_head = Transformer.config.n_head // world_size
    Transformer.config.dim = Transformer.config.dim // world_size
    Transformer.config.n_local_heads = Transformer.config.n_local_heads // world_size


def apply_tp(
    model: Transformer, requires_grad: bool = False, sequence_parallel: bool = False
) -> None:
    if get_model_parallel_world_size() > 1:
        _apply_tp_Transformer(model)
        # apply tensor parallel on the embedding layer
        if model.vocab_parallel:
            _apply_tp_embedding(model.tok_embeddings)

        if sequence_parallel:
            # implement sequence parallelism
            # See https://arxiv.org/pdf/2205.05198.pdf
            model.tok_embeddings.register_forward_hook(
                scatter_to_sequence_parallel_region_hook
            )
            model.output.register_forward_pre_hook(
                partial(
                    gather_from_sequence_parallel_region_pre_hook,
                    tensor_parallel_output_grad=False,
                )
            )

        # apply tensor parallel on each block
        for block in model.layers:
            # Apply to MLP
            _apply_tp_ffn(block.feed_forward, requires_grad, sequence_parallel)
            _apply_tp_attn(block.attention, requires_grad, sequence_parallel)

        # apply tensor parallel on the output layer
        if model.vocab_parallel:
            _apply_tp_linear(model.output, "colwise", requires_grad=requires_grad)


def apply_reward_head_tp(model: nn.Module, requires_grad: bool = False) -> None:
    if get_model_parallel_world_size() > 1:
        reward_head: nn.Linear = model.output
        _apply_tp_linear(reward_head, "rowwise", requires_grad=requires_grad)
        reward_head.register_forward_pre_hook(scatter_to_model_parallel_region_pre_hook)
        reward_head.register_forward_hook(reduce_from_model_parallel_region_hook)


def aggregate_forward_tensor_hook(
    module: nn.Module,
    input: Optional[Tuple[torch.Tensor, ...]],
    output: Optional[Tuple[torch.Tensor, ...]],
):
    # TODO (zhiqings): all-reduce with fp32
    # see https://github.com/NVIDIA/nccl/issues/1026
    aggregated_output = funcol.all_reduce(output, "sum", get_model_parallel_group())

    if module.training:
        # add gradient infomation to aggregated_output
        aggregated_output = aggregated_output - output.detach().clone() + output

    return aggregated_output


def aggregate_grad_hook(
    _module: nn.Module,
    grad_input: Optional[Tuple[torch.Tensor, ...]],
    grad_output: Optional[Tuple[torch.Tensor, ...]],
    num_postional_args: int = 1,
):
    if grad_input[0] is None:
        return tuple([None] * (num_postional_args))

    # TODO (zhiqings): all-reduce with fp32
    # see https://github.com/NVIDIA/nccl/issues/1026
    aggregated_grad_input = (
        funcol.all_reduce(grad_input[0], "sum", get_model_parallel_group()),
    )

    if num_postional_args == 1:
        return aggregated_grad_input
    else:
        return aggregated_grad_input + tuple([None] * (num_postional_args - 1))


def reduce_from_model_parallel_region_hook(
    module: nn.Module,
    input: Optional[Tuple[torch.Tensor, ...]],
    output: Optional[Tuple[torch.Tensor, ...]],
):
    # TODO (zhiqings): all-reduce with fp32
    # see https://github.com/NVIDIA/nccl/issues/1026
    if module.training:
        return _ReduceFromModelParallelRegion.apply(output)
    else:
        return funcol.all_reduce(output, "sum", get_model_parallel_group())


def scatter_to_model_parallel_region_pre_hook(
    module: nn.Module,
    args: Tuple[torch.Tensor, ...],
):
    assert len(args) == 1
    if module.training:
        return _ScatterToModelParallelRegion.apply(args[0])
    else:
        world_size = get_model_parallel_world_size()
        input_list = split_tensor_along_last_dim(args[0], world_size)

        rank = get_model_parallel_rank()
        output = input_list[rank].contiguous()
        return output


def scatter_to_sequence_parallel_region_hook(
    module: nn.Module,
    input: Optional[Tuple[torch.Tensor, ...]],
    output: Optional[Tuple[torch.Tensor, ...]],
):
    if module.training:
        return _ScatterToSequenceParallelRegion.apply(output)
    else:
        # identity as we only split the activation in training
        return output


def gather_from_sequence_parallel_region_pre_hook(
    module: nn.Module,
    args: Tuple[torch.Tensor, ...],
    tensor_parallel_output_grad: bool = True,
):
    if module.training:
        if len(args) == 1:
            return _GatherFromSequenceParallelRegion.apply(
                args[0], tensor_parallel_output_grad
            )
        else:
            return (
                _GatherFromSequenceParallelRegion.apply(
                    args[0], tensor_parallel_output_grad
                ),
            ) + args[1:]
    else:
        # identity as we only split the activation in training
        return args


def reduce_scatter_to_sequence_parallel_region_hook(
    module: nn.Module,
    input: Optional[Tuple[torch.Tensor, ...]],
    output: Optional[Tuple[torch.Tensor, ...]],
):
    if module.training:
        return _ReduceScatterToSequenceParallelRegion.apply(output)
    else:
        # identity as we only split the activation in training
        return output


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def initialize_model_parallel(
    model_parallel_size_: int,
    *,
    model_parallel_backend: Optional[str] = None,
    ddp_backend: Optional[str] = None,
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size: number of GPUs used to parallelize model.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = int(min(model_parallel_size_, world_size))
    ensure_divisibility(world_size, model_parallel_size)
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / model_parallel_size)

    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size_))
        print("> initializing ddp with size {}".format(data_parallel_size))

    groups = torch.LongTensor(range(world_size)).reshape(
        data_parallel_size, model_parallel_size
    )

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for k in range(model_parallel_size):
        group = torch.distributed.new_group(groups[:, k].tolist(), backend=ddp_backend)
        if k == found[1]:
            _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        group = torch.distributed.new_group(
            groups[i, :].tolist(), backend=model_parallel_backend
        )
        if i == found[0]:
            _MODEL_PARALLEL_GROUP = group


def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    global _MODEL_PARALLEL_SIZE
    if _MODEL_PARALLEL_SIZE is None:
        _MODEL_PARALLEL_SIZE = torch.distributed.get_world_size(
            group=get_model_parallel_group()
        )
    return _MODEL_PARALLEL_SIZE


def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    global _MODEL_PARALLEL_RANK
    if _MODEL_PARALLEL_RANK is None:
        _MODEL_PARALLEL_RANK = torch.distributed.get_rank(
            group=get_model_parallel_group()
        )
    return _MODEL_PARALLEL_RANK


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    global _DATA_PARALLEL_SIZE
    if _DATA_PARALLEL_SIZE is None:
        _DATA_PARALLEL_SIZE = torch.distributed.get_world_size(
            group=get_data_parallel_group()
        )
    return _DATA_PARALLEL_SIZE


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    global _DATA_PARALLEL_RANK
    if _DATA_PARALLEL_RANK is None:
        _DATA_PARALLEL_RANK = torch.distributed.get_rank(
            group=get_data_parallel_group()
        )
    return _DATA_PARALLEL_RANK


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide_and_check_no_remainder(
        tensor.size()[last_dim], num_partitions
    )
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


# -----------------
# Helper functions.
# -----------------


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _gather(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    group = get_model_parallel_group()

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # TODO (zhiqings): all-reduce with fp32
    # see https://github.com/NVIDIA/nccl/issues/1026
    torch.distributed.all_reduce(input_, group=group)

    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    group = get_model_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    with torch.no_grad():
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        torch.distributed.all_gather(tensor_list, input_, group=group)
    tensor_list[rank] = input_

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_model_parallel_group()
    )

    return output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_model_parallel_group()
    )
    return output


def clip_grad_norm_(
    fsdp_module,
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2.0,
    process_group: Optional[Any] = None,
) -> torch.Tensor:
    if isinstance(fsdp_module, FSDP):
        fsdp_utils._lazy_init(fsdp_module, fsdp_module)
        if not fsdp_module._is_root:
            raise RuntimeError(
                "`clip_grad_norm_()` should only be called on the root FSDP instance"
            )
        fsdp_module._assert_state(TrainingState.IDLE)

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    sharded_params = set()
    nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
    grads: List[torch.Tensor] = []

    if isinstance(fsdp_module, FSDP) and get_model_parallel_world_size() == 1:
        # If every FSDP instance uses `NO_SHARD`, then we can directly use
        # the normal `nn.utils` on targeting local gradients
        all_no_shard = all(
            not handle.uses_sharded_strategy for handle in fsdp_module._all_handles
        )
        if all_no_shard:
            return torch.nn.utils.clip_grad_norm_(
                fsdp_module.parameters(),
                max_norm,
                norm_type,
                error_if_nonfinite=True,
            )

    if isinstance(fsdp_module, FSDP):
        # Otherwise, there exists some FSDP instance using a sharded strategy,
        # where sharded and non-sharded parameters must be handled separately
        for handle in fsdp_module._all_handles:
            target_set = (
                sharded_params if handle.uses_sharded_strategy else nonsharded_params
            )
            if handle._use_orig_params:
                for param in handle.flat_param._params:
                    target_set.add(param)
                    if param.grad is not None:
                        grads.append(param.grad)
            else:
                target_set.add(handle.flat_param)
                if handle.flat_param.grad is not None:
                    grads.append(handle.flat_param.grad)

    for param in fsdp_module.parameters():
        not_fsdp_managed = (
            param not in sharded_params and param not in nonsharded_params
        )
        if not_fsdp_managed:
            nonsharded_params.add(param)
            if param.grad is not None:
                grads.append(param.grad)

    # Compute local norms (forced to be in FP32)
    if isinstance(fsdp_module, FSDP):
        compute_device = fsdp_module.compute_device
    else:
        # use first param's device
        compute_device = next(iter(fsdp_module.parameters())).device

    local_sharded_norm = _get_grad_norm(sharded_params, norm_type).to(compute_device)
    local_nonsharded_norm = _get_grad_norm(nonsharded_params, norm_type).to(
        compute_device
    )
    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
        dist.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=process_group
        )
    else:
        total_norm = (local_sharded_norm**norm_type) + (
            local_nonsharded_norm**norm_type
        )
        dist.all_reduce(total_norm, group=process_group)
        total_norm = total_norm ** (1.0 / norm_type)

    if isinstance(fsdp_module, FSDP) and fsdp_module.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        return total_norm
    total_norm_dtype = functools.reduce(
        lambda dtype1, dtype2: torch.promote_types(dtype1, dtype2),
        [grad.dtype for grad in grads],
    )
    return total_norm.to(total_norm_dtype)


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):  # type: ignore

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        vocab_start_index = rank * partition_vocab_size
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)


def compute_vocab_parallel_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    reduction: str = "none",
) -> torch.Tensor:
    # """Compute per-token logprobs, zeroing out places with ignore_index (padding)."""
    ce_scores = -vocab_parallel_cross_entropy(logits, labels)
    ignore_mask = labels == ignore_index
    ce_scores[ignore_mask] = 0.0

    if reduction == "none":
        return ce_scores
    elif reduction == "mean":
        ignore_mask = ignore_mask.float()
        return ce_scores.sum() / (ignore_mask.numel() - ignore_mask.sum())
    elif reduction == "sum":
        return ce_scores.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
