# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, DefaultDict, Dict, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict


class MemoryEfficientAdamW(Optimizer):
    """
    Arguments:
        model_params (iterable): iterable of parameters of dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adamw_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        model_params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        adamw_mode=True,
        optim_dtype=torch.bfloat16,
        optim_device=torch.device("cpu"),
    ):
        default_args = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(MemoryEfficientAdamW, self).__init__(model_params, default_args)
        self.adamw_mode = adamw_mode
        self.optim_dtype = optim_dtype
        self.optim_device = optim_device

    def torch_adam_update_cpu(
        self,
        data,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
        use_adamw=False,
    ):
        assert data.dtype == grad.dtype
        if weight_decay != 0:
            if use_adamw:
                data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(data, alpha=weight_decay)

        non_blocking = self.optim_device.type == "cpu"

        exp_avg_cuda, exp_avg_sq_cuda = (
            exp_avg.to(data.device, non_blocking=non_blocking),
            exp_avg_sq.to(data.device, non_blocking=non_blocking),
        )

        dtype_grad = grad.to(dtype=self.optim_dtype)
        exp_avg_cuda.mul_(beta1).add_(dtype_grad, alpha=1 - beta1)
        exp_avg_sq_cuda.mul_(beta2).addcmul_(dtype_grad, dtype_grad, value=1 - beta2)
        denom_cuda = (exp_avg_sq_cuda.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        data.addcdiv_(
            exp_avg_cuda.to(dtype=data.dtype),
            denom_cuda.to(dtype=data.dtype),
            value=-step_size,
        )

        # Write back to cpu
        exp_avg.copy_(exp_avg_cuda, non_blocking=non_blocking)
        exp_avg_sq.copy_(exp_avg_sq_cuda, non_blocking=non_blocking)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for _, group in enumerate(self.param_groups):
            for _, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                state = self.state[p]
                assert (
                    p.device.type == "cuda"
                ), f"PinMemoryCPUAdam assume all parameters are on cuda"
                if len(state) == 0:
                    state["step"] = 0
                    # gradient momentums
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        device=self.optim_device,
                        dtype=self.optim_dtype,
                    )
                    # gradient variances
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        device=self.optim_device,
                        dtype=self.optim_dtype,
                    )
                    if self.optim_device.type == "cpu":
                        state["exp_avg"] = state["exp_avg"].pin_memory()
                        state["exp_avg_sq"] = state["exp_avg_sq"].pin_memory()

                state["step"] += 1
                beta1, beta2 = group["betas"]

                assert (
                    p.data.numel() == p.grad.data.numel()
                ), "parameter and gradient should have the same size"
                assert (
                    state["exp_avg"].device.type == self.optim_device.type
                ), f"exp_avg should stay on {self.optim_device.type}"
                assert (
                    state["exp_avg_sq"].device.type == self.optim_device.type
                ), f"exp_avg should stay on {self.optim_device.type}"
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                self.torch_adam_update_cpu(
                    p.data,
                    p.grad.data,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                    group["weight_decay"],
                    bias_correction1,
                    bias_correction2,
                    self.adamw_mode,
                )
        return loss

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                if param.is_floating_point():
                    casted_value = value.to(
                        dtype=self.optim_dtype, device=self.optim_device
                    )
                    if self.optim_device.type == "cpu":
                        casted_value = casted_value.pin_memory()
                else:
                    casted_value = Optimizer._process_value_according_to_param_policy(
                        param, value, param_id, param_groups, key
                    )
                return casted_value
            elif isinstance(value, dict):
                return {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param, v, param_id=k, param_groups=state_dict["param_groups"]
                )
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(
            group: Dict[str, Any], new_group: Dict[str, Any]
        ) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)
