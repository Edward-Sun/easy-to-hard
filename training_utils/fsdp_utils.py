# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""This file fixed a bug in fsdp's _map_param_key_to_optim_keys & _broadcast(_processed)_state"""

import warnings
import copy
from contextlib import ExitStack
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._debug_utils import SimpleProfiler
from torch.distributed.fsdp._runtime_utils import _reset_flat_param_grad_info_if_needed
from torch.utils._pytree import tree_map_only


from torch.distributed.fsdp._optim_utils import (
    FSDPParamInfo,
    FlatParameter,
    _OptimStateKey,
    _PosDimTensorInfo,
    _get_flat_param_to_fqn,
    _get_fqn_to_fsdp_param_info,
    _get_param_id_to_param_from_optim_input,
    _get_param_key_to_param,
    _is_named_optimizer,
    _unflatten_param_groups,
    _convert_state_with_orig_params,
    _convert_state_with_flat_params,
    _check_missing_keys_on_rank,
    _rekey_sharded_optim_state_dict,
    _shard_orig_param_state,
    _flatten_optim_state,
)


def fixed_scatter_full_optim_state_dict(
    full_optim_state_dict: Optional[Dict[str, Any]],
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    group: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Scatters the full optimizer state dict from rank 0 to all other ranks,
    returning the sharded optimizer state dict on each rank. The return
    value is the same as :meth:`shard_full_optim_state_dict`, and on rank
    0, the first argument should be the return value of
    :meth:`full_optim_state_dict`.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> model, optim = ...
        >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
        >>> # Define new model with possibly different world size
        >>> new_model, new_optim, new_group = ...
        >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
        >>> new_optim.load_state_dict(sharded_osd)

    .. note:: Both :meth:`shard_full_optim_state_dict` and
        :meth:`scatter_full_optim_state_dict` may be used to get the
        sharded optimizer state dict to load. Assuming that the full
        optimizer state dict resides in CPU memory, the former requires
        each rank to have the full dict in CPU memory, where each rank
        individually shards the dict without any communication, while the
        latter requires only rank 0 to have the full dict in CPU memory,
        where rank 0 moves each shard to GPU memory (for NCCL) and
        communicates it to ranks appropriately. Hence, the former has
        higher aggregate CPU memory cost, while the latter has higher
        communication cost.

    Args:
        full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
            dict corresponding to the unflattened parameters and holding
            the full non-sharded optimizer state if on rank 0; the argument
            is ignored on nonzero ranks.
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            correspond to the optimizer state in ``full_optim_state_dict``.
        optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
            Input passed into the optimizer representing either a
            :class:`list` of parameter groups or an iterable of parameters;
            if ``None``, then this method assumes the input was
            ``model.parameters()``. This argument is deprecated, and there
            is no need to pass it in anymore. (Default: ``None``)
        optim (Optional[torch.optim.Optimizer]): Optimizer that will load
            the state dict returned by this method. This is the preferred
            argument to use over ``optim_input``. (Default: ``None``)
        group (dist.ProcessGroup): Model's process group or ``None`` if
            using the default process group. (Default: ``None``)

    Returns:
        Dict[str, Any]: The full optimizer state dict now remapped to
        flattened parameters instead of unflattened parameters and
        restricted to only include this rank's part of the optimizer state.
    """
    FullyShardedDataParallel._warn_legacy_optim_state_dict(
        "scatter_full_optim_state_dict", "optim_state_dict_to_load"
    )
    return fixed_optim_state_dict_to_load_impl(
        optim_state_dict=full_optim_state_dict,
        model=model,
        optim_input=optim_input,
        optim=optim,
        full_state_dict=True,
        rank0_only=True,
        is_named_optimizer=False,
        group=group,
    )


def fixed_optim_state_dict_to_load_impl(
    optim_state_dict: Dict[str, Any],
    model: torch.nn.Module,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    full_state_dict: bool = True,
    rank0_only: bool = False,
    is_named_optimizer: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

    This is the internal API that is used by all the load optim_state_dict implementations.
    Given model, optim, and the saved optim_state_dict, this API adds the FSDP
    internal information and internal sharding to the optim_state_dict.
    """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
    else:
        using_optim_input = False
        assert optim_input is None and not rank0_only

    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all(
        use_orig_params == m._use_orig_params
        for m in FullyShardedDataParallel.fsdp_modules(model)
    ), "Not all FSDP modules have the same _use_orig_params value"

    if rank0_only and dist.get_rank(group) > 0:
        optim_state_dict = {}
    sharded_osd = fixed_flatten_optim_state_dict(
        optim_state_dict,
        model=model,
        use_orig_params=use_orig_params,
        optim=(optim if is_named_optimizer else None),
        rank0_only=rank0_only,
        group=group,
    )
    return _rekey_sharded_optim_state_dict(
        sharded_osd,
        model=model,
        optim=optim,
        optim_input=optim_input,
        using_optim_input=using_optim_input,
        is_named_optimizer=is_named_optimizer,
    )


def fixed_flatten_optim_state_dict(
    optim_state_dict: Dict[str, Any],
    model: nn.Module,
    use_orig_params: bool = False,
    optim: Optional[torch.optim.Optimizer] = None,
    rank0_only: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()

    unflat_osd = optim_state_dict
    if "state" not in unflat_osd and not rank0_only:
        raise ValueError(
            '`optim_state_dict` must have the keys "state"'
            "to be a valid optimizer state dict"
        )
    param_to_fqns = _get_param_to_fqns(model)
    fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)
    fsdp_state = next(iter(fqn_to_fsdp_param_info.values())).state

    # Broadcast unflat_osd without non-scalar tensor if rank0_only is True.
    if rank0_only:
        unflat_osd = fixed_broadcast_processed_state(
            fsdp_state, unflat_osd, group=group
        )

    # Construct the "state" part
    flat_osd_state: Dict[Union[_OptimStateKey, str], Any] = {}
    unflat_osd_state = unflat_osd["state"]
    all_state_keys = set(unflat_osd_state.keys())

    for param, fqns in param_to_fqns.items():
        fqn = fqns[0]
        if fqn not in unflat_osd_state:
            continue
        all_state_keys.difference_update(fqns)

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name in unflat_osd_state[fqn].keys():
                    unflat_osd_state[fqn][state_name] = fixed_broadcast_state(
                        fsdp_state, unflat_osd_state[fqn][state_name], group=group
                    )
            fqn = fqns[0]
        if fqn in fqn_to_fsdp_param_info:
            fsdp_param_info = fqn_to_fsdp_param_info[fqn]
            if use_orig_params:
                with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
                    flat_state = _shard_orig_param_state(
                        fsdp_param_info,
                        fqn,
                        unflat_osd_state[fqn],
                    )
            else:
                flat_state = _flatten_optim_state(
                    fsdp_param_info,
                    unflat_osd_state,
                    fqns,
                )
            key = _OptimStateKey(tuple(fqns), True)
            # Only include non-empty states since as expected by
            # `torch.optim.Optimizer` s unless the optimizer is KeyedOptimizer
            # or NamedOptimizer.
            if flat_state:
                flat_osd_state[key] = flat_state
            elif use_orig_params:
                assert (
                    len(fqns) == 1
                ), f"use_orig_params is True but there are multiple FQNs, {fqns}."
                if optim is not None:  # NamedOptimizer or KeyedOptimizer case.
                    state = optim.state.get(param, None)  # type: ignore[call-overload]
                    if state is not None:
                        flat_osd_state[key] = copy.deepcopy(state)
                    else:
                        warnings.warn(
                            f"optim_state[{key}] is not on rank{fsdp_state.rank}."
                        )

            else:
                raise RuntimeError(
                    f"The state of {key} is empty. This should happen when "
                    "use_orig_params=True."
                )
        else:  # do not flatten non-FSDP parameters' states
            assert len(fqns) == 1
            key = _OptimStateKey(tuple(fqns), False)
            flat_osd_state[key] = copy.copy(unflat_osd_state[fqn])

        if rank0_only:
            for fqn in fqns:
                if not unflat_osd_state[fqn]:
                    continue
                for state_name, param_state in list(unflat_osd_state[fqn].items()):
                    if fsdp_state.rank > 0:
                        # Deference the tensor so that PyTorch can collect the memory.
                        del unflat_osd_state[fqn][state_name]
                    else:
                        # Move the tensor in the original osd back to CPU to make the
                        # original osd unaffected.
                        unflat_osd_state[fqn][state_name] = unflat_osd_state[fqn][
                            state_name
                        ].cpu()

    # Handle user-defined state, states that are not associated with parameters.
    for key in all_state_keys:
        user_state = unflat_osd_state[key]
        if isinstance(user_state, torch.Tensor) and rank0_only and use_orig_params:
            user_state = fixed_broadcast_state(fsdp_state, user_state, group=group)
        flat_osd_state[key] = copy.copy(user_state)

    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict() profiling: ")
    # Construct the "param_groups" part -- copy as is since it will be
    # rekeyed later according to the target rank's optimizer
    # Only copy param_groups if it exists in unflat_osd
    if "param_groups" in unflat_osd:
        flat_osd_param_groups = copy.deepcopy(unflat_osd["param_groups"])
        return {"state": flat_osd_state, "param_groups": flat_osd_param_groups}
    else:
        return {"state": flat_osd_state}


def fixed_broadcast_processed_state(
    fsdp_state: _FSDPState,
    optim_state: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> Dict[str, Any]:
    objects: List[Any] = [None]
    if fsdp_state.rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )
    # Comment (zhiqings): the line below is a workaround for a bug in
    rank_0_global_rank = 0
    if group is not None:
        rank_0_global_rank = dist.get_global_rank(group, 0)
    dist.broadcast_object_list(objects, src=rank_0_global_rank, group=group)
    if fsdp_state.rank == 0:
        return optim_state
    else:
        return objects[0]


def fixed_broadcast_state(
    fsdp_state: _FSDPState, state: Any, group: Optional[dist.ProcessGroup]
) -> Any:
    if fsdp_state.rank == 0:
        if not isinstance(state, torch.Tensor) or state.dim() == 0:
            return state
        tensor = state.to(fsdp_state.compute_device)
    else:
        if isinstance(state, torch.Tensor):
            assert state.dim() == 0, (
                "For non-zero ranks, a tensor state should have zero dimension, "
                "but got the state with shape {state.shape()}."
            )
            return state
        elif not isinstance(state, _PosDimTensorInfo):
            return state
        tensor = torch.zeros(
            state.shape, dtype=state.dtype, device=fsdp_state.compute_device
        )
    # Comment (zhiqings): the line below is a workaround for a bug in
    rank_0_global_rank = 0
    if group is not None:
        rank_0_global_rank = dist.get_global_rank(group, 0)
    dist.broadcast(tensor, src=rank_0_global_rank, group=group)
    return tensor


def fixed_full_optim_state_dict(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    rank0_only: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """Return the full optimizer state-dict.

    Consolidates the full optimizer state on rank 0 and returns it
    as a :class:`dict` following the convention of
    :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
    and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
    contained in ``model`` are mapped back to their unflattened parameters.

    .. warning:: This needs to be called on all ranks since it uses
        collective communications. However, if ``rank0_only=True``, then
        the state dict is only populated on rank 0, and all other ranks
        return an empty :class:`dict`.

    .. warning:: Unlike ``torch.optim.Optimizer.state_dict()``, this method
        uses full parameter names as keys instead of parameter IDs.

    .. note:: Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
        contained in the optimizer state dict are not cloned, so there may
        be aliasing surprises. For best practices, consider saving the
        returned optimizer state dict immediately, e.g. using
        ``torch.save()``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
            Input passed into the optimizer ``optim`` representing either a
            :class:`list` of parameter groups or an iterable of parameters;
            if ``None``, then this method assumes the input was
            ``model.parameters()``. This argument is deprecated, and there
            is no need to pass it in anymore. (Default: ``None``)
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        group (dist.ProcessGroup): Model's process group or ``None`` if using
            the default process group. (Default: ``None``)

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
        then nonzero ranks return an empty :class:`dict`.
    """
    FullyShardedDataParallel._warn_legacy_optim_state_dict(
        "full_optim_state_dict", "optim_state_dict"
    )
    return fixed_optim_state_dict_impl(
        model=model,
        optim=optim,
        optim_state_dict=optim.state_dict(),
        optim_input=optim_input,
        rank0_only=rank0_only,
        group=group,
        full_state_dict=True,
    )


def fixed_optim_state_dict_impl(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[torch.nn.Parameter],
        ]
    ] = None,
    rank0_only: bool = True,
    full_state_dict: bool = True,
    group: Optional[dist.ProcessGroup] = None,
    cpu_offload: bool = True,
) -> Dict[str, Any]:
    """Transform the state-dict of an optimizer corresponding to a sharded model.

    This is the internal API that is used by all the optim_state_dict implementations.
    Given model, optim, the original optim_state_dict, this API removes the
    FSDP internal information and internal sharding from the optim_state_dict.
    """
    if full_state_dict:
        FullyShardedDataParallel._warn_optim_input(optim_input)
        using_optim_input = FullyShardedDataParallel._is_using_optim_input(
            optim_input,
            optim,
        )
    else:
        using_optim_input = False
        assert optim_input is None and not rank0_only

    use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[0]._use_orig_params
    assert all(
        use_orig_params == m._use_orig_params
        for m in FullyShardedDataParallel.fsdp_modules(model)
    ), "Not all FSDP modules have the same _use_orig_params value"

    return fixed_optim_state_dict(
        model=model,
        optim=optim,
        optim_state_dict=optim_state_dict,
        optim_input=optim_input,
        rank0_only=rank0_only,
        shard_state=not full_state_dict,
        group=group,
        using_optim_input=using_optim_input,
        use_orig_params=use_orig_params,
        cpu_offload=cpu_offload,
    )


@torch.no_grad()
def fixed_optim_state_dict(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    optim_state_dict: Dict[str, Any],
    optim_input: Optional[
        Union[
            List[Dict[str, Any]],
            Iterable[nn.Parameter],
        ]
    ],
    rank0_only: bool,
    shard_state: bool,
    group: Optional[dist.ProcessGroup],
    using_optim_input: bool,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> Dict[str, Any]:
    """
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flat parameters in ``FSDP`` modules contained in ``model`` are mapped
    back to their unflattened parameters.

    Parameter keys are not well-defined. For a regular optimizer, the optimizer
    state_dict contains a mapping from parameter IDs to parameter states.
    Parameter IDs are the order of parameters in ``optim.param_groups()`` across
    all the groups. This API also allows user to pass ``optim_input`` for the
    mapping between parameters and parameter IDs. Using ``optim_input`` is being
    deprecated.

    If the optimizer is a ``NamedOptimizer``, the optimizer state_dict does not
    contain parameter IDs mapping but a mapping from parameter FQNs to parameter
    states. This API finds the mapping from FQNs to parameters if the optimizer
    is a ``NamedOptimizer``.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- those are managed by other parallelisms and FSDP does not
    know how to handle/aggregate them.

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` 's
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` 's original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    """
    SimpleProfiler.reset()
    cm = ExitStack()
    cm.enter_context(SimpleProfiler.profile(SimpleProfiler.Type.ALL))
    _reset_flat_param_grad_info_if_needed(traversal_utils._get_fsdp_handles(model))
    to_save = not rank0_only or dist.get_rank(group) == 0 or shard_state

    with SimpleProfiler.profile("preprocessing"):
        param_to_fqns = _get_param_to_fqns(model)
        flat_param_to_fqn = _get_flat_param_to_fqn(model)
        is_named_optimizer = _is_named_optimizer(optim_state_dict)

        param_key_to_param = cast(
            Dict[Union[int, str], nn.Parameter],
            (
                _get_param_id_to_param_from_optim_input(model, optim_input)
                if using_optim_input
                else _get_param_key_to_param(
                    optim, model, is_named_optimizer, param_to_fqns, flat_param_to_fqn
                )
            ),
        )
        fqn_to_fsdp_param_info = _get_fqn_to_fsdp_param_info(model)

    with SimpleProfiler.profile("preprocessing_with_comm"):
        (
            all_optim_state_keys,
            optim_state_key_to_param_key,
        ) = fixed_map_param_key_to_optim_keys(
            optim_state_dict,
            group,
            param_key_to_param,
            param_to_fqns,
            fqn_to_fsdp_param_info,
            merge_keys=use_orig_params,
        )

    with SimpleProfiler.profile("state_converting"):
        convert_fn = (
            _convert_state_with_orig_params
            if use_orig_params
            else _convert_state_with_flat_params
        )
        fsdp_osd_state = convert_fn(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            fqn_to_fsdp_param_info,
            optim_state_dict["state"],
            to_save,
            shard_state,
            cpu_offload,
        )

    # At this point, communication is complete and ranks can return early if nothing
    # will be saved on that rank.
    if not to_save:
        return {}

    fsdp_osd: Dict[str, Any] = {"state": fsdp_osd_state}

    flat_param_fqns = set(flat_param_to_fqn.values())
    for key, value in optim_state_dict["state"].items():
        if key in fsdp_osd_state:
            continue
        if key in flat_param_fqns:
            continue
        if key in param_key_to_param:
            continue
        # This key is not recognized by FSDP. It may be a user-defined state
        # or some parameters state that FSDP is unable to map from
        # ``optim.param_groups``.
        warnings.warn(
            f"Found a optim state, {key}, that FSDP cannot process. FSDP "
            "will directly copy everything to the returned state_dict. In "
            "most cases, this is a user-defined state that is not "
            "associated with any particular parameter. Another possible "
            "case is this state is managed by TorchRec. Otherwise, there may "
            " be a mismatched assumption of optim_state_dict of this mode."
        )
        fsdp_osd_state[key] = value

    if "param_groups" in optim_state_dict:
        fsdp_osd["param_groups"] = _unflatten_param_groups(
            optim_state_dict, param_key_to_param, param_to_fqns
        )

    cm.close()
    SimpleProfiler.dump_and_reset("FSDP _optim_state_dict() profiling: ")

    return fsdp_osd


def fixed_map_param_key_to_optim_keys(
    optim_state_dict: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
    param_key_to_param: Dict[Union[int, str], nn.Parameter],
    param_to_fqns: Dict[nn.Parameter, List[str]],
    fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
    merge_keys: bool = False,
) -> Tuple[List[_OptimStateKey], Dict[_OptimStateKey, Union[int, str]]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
    rank = dist.get_rank(group)
    optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]] = {}  # local
    all_optim_state_keys: List[_OptimStateKey] = []

    for param_key, param in param_key_to_param.items():
        # Do not include parameters without state to avoid empty mappings
        # just like in normal `torch.optim.Optimizer.state_dict()`
        if param_key not in optim_state_dict["state"]:
            continue
        fqns = param_to_fqns[param]
        is_fsdp_managed = isinstance(param, FlatParameter)
        if is_fsdp_managed:
            assert fqns[0] in fqn_to_fsdp_param_info, (
                fqns[0],
                list(fqn_to_fsdp_param_info.keys()),
            )
        is_fsdp_managed = fqns[0] in fqn_to_fsdp_param_info
        optim_state_key = _OptimStateKey(
            unflat_param_names=tuple(fqns),
            is_fsdp_managed=is_fsdp_managed,
        )
        if rank == 0 or merge_keys:
            all_optim_state_keys.append(optim_state_key)
        optim_state_key_to_param_key[optim_state_key] = param_key

    if merge_keys:
        all_keys: List[List[_OptimStateKey]] = [
            [] for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather_object(all_keys, all_optim_state_keys, group=group)
        merge_all_optim_state_keys = [
            key for local_keys in all_keys for key in local_keys
        ]
        all_optim_state_keys = sorted(set(merge_all_optim_state_keys))
    else:
        key_obj_list: List[Optional[List[_OptimStateKey]]] = (
            [all_optim_state_keys] if rank == 0 else [None]
        )
        # Comment (zhiqings): the line below is a workaround for a bug in
        rank_0_global_rank = 0
        if group is not None:
            rank_0_global_rank = dist.get_global_rank(group, 0)
        dist.broadcast_object_list(key_obj_list, src=rank_0_global_rank, group=group)
        assert key_obj_list[0] is not None
        all_optim_state_keys = key_obj_list[0]
        _check_missing_keys_on_rank(
            all_optim_state_keys,
            optim_state_key_to_param_key,
            param_key_to_param,
            group,
        )

    return all_optim_state_keys, optim_state_key_to_param_key
