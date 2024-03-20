# Copyright 2024 The GPT-Accelera Team
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

import abc
import contextlib
import logging
import gc
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from arguments import Arguments
import trainers.common_utils as utils
import training_utils.trainer_utils as trainer_utils

from models.tokenizer_utils import AcceleraTokenizer
from models.tp import (
    clip_grad_norm_,
    get_model_parallel_group,
    get_model_parallel_world_size,
)
from models.rl_model import Policy, Value
from models.reward_model import RewardModel


logger = logging.getLogger(__name__)

FIRST_STEP_IDX = 1
IGNORE_INDEX = -100


class KLController(abc.ABC):
    value: Union[int, float]

    def step(self, *args, **kwargs):
        pass


class FixedKLController(KLController):
    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = kl_coef


class RLTrainer(object):
    def __init__(
        self,
        args: Arguments,
        train_dataset: Dataset,
        data_collator: Callable,
        tokenizer: AcceleraTokenizer,
        policy: Policy,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        value_model: Optional[Value] = None,
        ref_policy: Optional[Policy] = None,
        reward_model: Optional[RewardModel] = None,
        policy_optimizer: Optional[torch.optim.Optimizer] = None,
        value_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_lr_scheduler: Optional[LRScheduler] = None,
        value_lr_scheduler: Optional[LRScheduler] = None,
        reward_tokenizer: Optional[AcceleraTokenizer] = None,
        unwrapped_policy: Optional[Policy] = None,
    ):
        super(RLTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.data_collator = data_collator
        self.policy: Policy = policy
        self.value_model: Optional[Value] = value_model
        self.ref_policy: Optional[Policy] = ref_policy
        self.reward_model: Optional[RewardModel] = reward_model
        self.unwrapped_policy: Optional[Policy] = unwrapped_policy
        self.tokenizer: AcceleraTokenizer = tokenizer
        self.policy_optimizer: Optional[torch.optim.Optimizer] = policy_optimizer
        self.value_optimizer: Optional[torch.optim.Optimizer] = value_optimizer
        self.policy_lr_scheduler: Optional[LRScheduler] = policy_lr_scheduler
        self.value_lr_scheduler: Optional[LRScheduler] = value_lr_scheduler
        self.kl_ctl = FixedKLController(kl_coef=args.kl_coef)
        self.log_history = []

        if reward_tokenizer is None:
            self.reward_tokenizer = self.tokenizer
        else:
            self.reward_tokenizer = reward_tokenizer

        self.best_metrics = None

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_value_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        raise NotImplementedError

    @property
    def policy_optimizable_params(self):
        return [
            p
            for p in self.policy.parameters()
            if p.requires_grad and p.grad is not None
        ]

    @property
    def value_optimizable_params(self):
        return [
            p
            for p in self.value_model.parameters()
            if p.requires_grad and p.grad is not None
        ]

    @property
    def optimizable_params(self):
        if self.value_model is None:
            return self.policy_optimizable_params
        else:
            return self.policy_optimizable_params + self.value_optimizable_params

    @torch.inference_mode()
    def _compute_grad_norm_policy(self):
        grad_norm = torch.stack(
            [p.grad.norm(2) for p in self.policy_optimizable_params]
        ).norm(2)
        return grad_norm

    @torch.inference_mode()
    def _compute_grad_norm_value(self):
        grad_norm = torch.stack(
            [p.grad.norm(2) for p in self.value_optimizable_params]
        ).norm(2)
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        return param_norm

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts)
        stats_list_policy = []
        stats_list_value = []

        accumulate_steps = self.args.gradient_accumulation_steps
        loss_scale = 1.0 / accumulate_steps

        gc.collect()
        torch.cuda.empty_cache()

        self.policy_optimizer.zero_grad(set_to_none=True)
        for epoch_idx in range(self.args.noptepochs):
            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1),
                total=len(rollouts_dataloader),
                disable=not self.is_main_process,
                desc="gradstep_policy",
            ):
                if self.args.policy_model_fsdp and self.args.policy_model_cpu_offload:
                    policy_no_sync = self.policy.base_model.no_sync()
                elif isinstance(self.policy.base_model, DDP):
                    policy_no_sync = self.policy.base_model.no_sync()
                else:
                    policy_no_sync = no_op_context_manager()

                if batch_idx % accumulate_steps != 0:
                    with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                        with policy_no_sync:
                            policy_loss, policy_stats = self.compute_policy_loss(
                                rollouts_batch
                            )
                    (loss_scale * policy_loss).backward()
                else:
                    with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                        stats_for_this_step = {}
                        policy_loss, policy_stats = self.compute_policy_loss(
                            rollouts_batch
                        )
                        stats_for_this_step.update(policy_stats)
                    (loss_scale * policy_loss).backward()

                    # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                    grad_norm = None
                    if self.args.max_grad_norm is not None:
                        grad_norm = clip_grad_norm_(
                            self.policy.base_model,
                            self.args.max_grad_norm,
                        )
                    stats_for_this_step["loss/grad_norm_policy"] = (
                        grad_norm or self._compute_grad_norm_policy()
                    )
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    stats_list_policy.append(stats_for_this_step)

            self.value_optimizer.zero_grad(set_to_none=True)
            if self.value_model is not None:
                for batch_idx, rollouts_batch in tqdm.tqdm(
                    enumerate(rollouts_dataloader, 1),
                    total=len(rollouts_dataloader),
                    disable=not self.is_main_process,
                    desc="gradstep_value",
                ):
                    # In FSDP, no_sync can help users to run micro_batches and allow gradient accumulation among iterations.
                    # It can help reduce communications using micro_batches with no_sync context manager.
                    # disable no_sync() comparatively has more network bandwidth demand
                    # but less GPU memory requirement per worker.
                    if self.args.value_model_fsdp and self.args.value_model_cpu_offload:
                        value_no_sync = self.value_model.base_model.no_sync()
                    elif isinstance(self.value_model.base_model, DDP):
                        value_no_sync = self.value_model.base_model.no_sync()
                    else:
                        value_no_sync = no_op_context_manager()

                    if batch_idx % accumulate_steps != 0:
                        with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                            with value_no_sync:
                                value_loss, value_stats = self.compute_value_loss(
                                    rollouts_batch
                                )
                        (loss_scale * value_loss).backward()
                    else:
                        with torch.cuda.amp.autocast(dtype=self.args.compute_dtype):
                            stats_for_this_step = {}
                            value_loss, value_stats = self.compute_value_loss(
                                rollouts_batch
                            )
                            stats_for_this_step.update(value_stats)
                        (loss_scale * value_loss).backward()

                        grad_norm = None
                        if self.args.max_grad_norm is not None:
                            grad_norm = clip_grad_norm_(
                                self.value_model.base_model,
                                self.args.max_grad_norm,
                            )
                        stats_for_this_step["loss/grad_norm_value"] = (
                            grad_norm or self._compute_grad_norm_value()
                        )
                        self.value_optimizer.step()
                        self.value_optimizer.zero_grad(set_to_none=True)
                        stats_list_value.append(stats_for_this_step)

        stats_policy = utils.merge_dict(stats_list_policy, torch.stack)
        stats_value = utils.merge_dict(stats_list_value, torch.stack)
        stats = {**stats_policy, **stats_value}
        return stats

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)
        ]
        rollouts = self.rollout(queries_batches)

        eval_stats = None
        if self.args.eval_steps is not None and step_idx % self.args.eval_steps == 0:
            eval_stats = self.evaluate(step_idx)

        if step_idx % self.args.save_steps == 0:
            if self.args.save_strategy == "steps":
                self.save_model(
                    step_idx=step_idx,
                )
            elif self.args.save_strategy == "no":
                pass
            else:
                assert self.args.save_steps % self.args.eval_steps == 0
                assert self.args.save_strategy in eval_stats
                save_metric = self.args.save_strategy

                if self.best_metrics is None:
                    self.best_metrics = eval_stats
                    self.save_model(
                        step_idx=step_idx,
                        metrics=eval_stats,
                    )
                elif eval_stats[save_metric] > self.best_metrics[save_metric]:
                    self.best_metrics = eval_stats
                    self.save_model(
                        step_idx=step_idx,
                        metrics=eval_stats,
                    )

        train_stats = self.step_with_rollouts(rollouts)
        if self.policy_lr_scheduler is not None:
            self.policy_lr_scheduler.step()
        if self.value_lr_scheduler is not None:
            self.value_lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
            kl_coef=self.kl_ctl.value,
        )
        self.kl_ctl.step(stats["objective/kl_sum_seq"])
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        self.policy_optimizer = trainer_utils.create_optimizer(
            args=self.args,
            model=self.policy,
            optimizer_cpu_offload=self.args.policy_optimizer_cpu_offload,
            model_cpu_offload=self.args.policy_model_cpu_offload,
        )
        self.policy_lr_scheduler = trainer_utils.get_cosine_schedule_with_warmup(
            optimizer=self.policy_optimizer,
            warmup_epochs=self.args.ppo_warmup_steps,
            max_epochs=num_training_steps,
            warmup_start_ratio=0.0,
            eta_min_ratio=0.01,
        )
        if self.value_model is not None:
            self.value_optimizer = trainer_utils.create_optimizer(
                args=self.args,
                model=self.value_model,
                optimizer_cpu_offload=self.args.value_optimizer_cpu_offload,
                model_cpu_offload=self.args.value_model_cpu_offload,
            )
            self.value_lr_scheduler = trainer_utils.get_cosine_schedule_with_warmup(
                optimizer=self.value_optimizer,
                warmup_epochs=self.args.ppo_warmup_steps,
                max_epochs=num_training_steps,
                warmup_start_ratio=0.0,
                eta_min_ratio=0.01,
            )

    def train(self, resume_training_ckpt: Optional[str] = None):
        """Entry point for training."""
        total_epochs = self.args.num_train_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa

        if self.is_main_process:
            logger.warning(
                f"***Training starts***\n"
                f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
            )

        self.create_optimizer_and_scheduler(total_steps)

        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            print(
                f"Resuming training from {resume_training_ckpt} at step {skipping_steps}."
            )

        infinite_train_dataloader = self.get_train_dataloader()
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, total_steps + FIRST_STEP_IDX),
            disable=not self.is_main_process,
            desc="steps",
            total=total_steps,
        ):
            if step_idx < skipping_steps:
                for _ in range(self.args.rollout_accumulation_steps):
                    next(infinite_train_dataloader)
                continue

            stats = self.step(infinite_train_dataloader, step_idx)
            self.log_history.append(stats)
        return self.log_history

    @abc.abstractmethod
    @torch.no_grad()
    def evaluate(self, step_idx: int, temperature: float = 0.0):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.no_grad()
    def save_model(self, step_idx: int, metrics: Optional[Dict] = None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.no_grad()
    def resume_training(self, checkpoint_dir: str):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        if self.is_main_process:
            logger.warning(
                f"Batch size of {loader_name} dataloader: {batch_size}",
                # main_process_only=True,
            )

    def get_train_dataloader(self):
        if self.is_main_process:
            logger.warning(
                f"Train dataset size: {len(self.train_dataset)}",
                # main_process_only=True
            )  # noqa
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
            )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=(sampler is None),
            drop_last=True,
            sampler=sampler,
            collate_fn=self.data_collator,
        )

        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(
        self, rollouts: Dict[str, torch.Tensor], drop_last=True, keys=None
    ):
        if keys is None:
            keys = tuple(rollouts.keys())

        def collate_rollouts(instances: Sequence[tuple]):
            return {
                key: torch.stack([instance[idx] for instance in instances])
                for idx, key in enumerate(keys)
            }

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])

        batch_size = (
            self.args.step_per_device_batch_size * get_model_parallel_world_size()
        )

        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=batch_size,
            collate_fn=collate_rollouts,
            shuffle=False,  # to avoid sync rollouts across tp devices
            drop_last=drop_last,
        )
        # Do not prepare, since we don't need to shard the rollouts sampled on each batch.
        return rollouts_dataloader

    @property
    def is_main_process(self):
        if dist.is_initialized():
            return dist.get_rank() == 0
        else:
            return True

    @property
    def device(self):
        if isinstance(self.policy, FSDP):
            compute_device = self.policy.compute_device
        else:
            # use first param's device
            compute_device = next(iter(self.policy.parameters())).device
        return compute_device

    @staticmethod
    @torch.inference_mode()
    def prepare_tp_batch(
        input_ids: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        use_tp: Optional[bool] = None,
        sync_group: Optional[dist.ProcessGroup] = None,
    ):
        if use_tp is None:
            use_tp = dist.is_initialized() and (get_model_parallel_world_size() > 1)

        if use_tp and sync_group is None:
            sync_group = get_model_parallel_group()

        if use_tp:
            # aggregate (concat) all the inputs across tp sync_group

            new_labels = None
            if isinstance(input_ids, torch.Tensor):
                assert input_ids.dim() == 2
                new_input_ids = torch.empty_like(input_ids).repeat(sync_group.size(), 1)

                input_handle = dist.all_gather_into_tensor(
                    new_input_ids, input_ids, group=sync_group, async_op=True
                )

                label_handle = None
                if labels is not None:
                    new_labels = torch.empty_like(labels).repeat(sync_group.size(), 1)
                    label_handle = dist.all_gather_into_tensor(
                        new_labels, labels, group=sync_group, async_op=True
                    )

                input_handle.wait()
                if label_handle is not None:
                    label_handle.wait()

            elif isinstance(input_ids, dict):
                assert labels is None
                new_input_ids = {}
                handles = []
                for key, value in input_ids.items():
                    repeats = [sync_group.size()] + [1] * (value.dim() - 1)
                    new_input_ids[key] = torch.empty_like(value).repeat(*repeats)

                    handle = dist.all_gather_into_tensor(
                        new_input_ids[key], value, group=sync_group, async_op=True
                    )
                    handles.append(handle)

                for handle in handles:
                    handle.wait()
            else:
                raise ValueError(
                    f"Unsupported input_ids type: {type(input_ids)} for TP"
                )

            input_ids = new_input_ids
            labels = new_labels

        if labels is not None:
            return input_ids, labels
        else:
            return input_ids


def truncate_after_eos(completions, eos_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[:end_idx]
        except ValueError:
            pass
    return clean_completions


def truncate_after_eos_padded(completions, eos_token_id, pad_token_id):
    # We truncate tokens after eos_token_id
    dtype, device = completions.dtype, completions.device
    clean_completions = completions.tolist()
    max_length = len(clean_completions[0])
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[:end_idx]
        except ValueError:
            pass

    # Pad the completions to the original length
    for idx, completion in enumerate(clean_completions):
        if len(completion) < max_length:
            clean_completions[idx] = completion + [pad_token_id] * (
                max_length - len(completion)
            )
    return torch.tensor(clean_completions, dtype=dtype, device=device)


@contextlib.contextmanager
def no_op_context_manager():
    yield None
