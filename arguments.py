# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Whether to run eval on the test set."}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )

    # Data
    source_max_len: int = field(
        default=1024, metadata={"help": "Maximum length of source sequence."}
    )

    target_max_len: int = field(
        default=256, metadata={"help": "Maximum length of target sequence."}
    )

    total_max_len: int = field(
        default=None, metadata={"help": "Maximum length of total sequence."}
    )

    dataset: str = field(
        default="alpaca", metadata={"help": "Which dataset to finetune on."}
    )

    eval_dataset: Optional[str] = field(
        default=None, metadata={"help": "Which dataset to evaluate on."}
    )

    test_dataset: Optional[str] = field(
        default=None, metadata={"help": "Which dataset to test on."}
    )

    dataset_format: str = field(
        default=None, metadata={"help": "Which dataset format is used."}
    )

    add_eos_to_target: bool = field(
        default=False, metadata={"help": "Whether to add an EOS token to the target."}
    )

    add_eos_to_marked_target: bool = field(
        default=False,
        metadata={"help": "Whether to add an EOS token to the marked target."},
    )

    eval_dataset_size: Optional[int] = field(
        default=None, metadata={"help": "Number of examples to use for evaluation."}
    )

    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of samples to evaluate."}
    )

    eval_size: int = field(
        default=0, metadata={"help": "Number of examples to use for evaluation."}
    )

    # Training
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs."}
    )

    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})

    lr_eta_min: float = field(
        default=0.0, metadata={"help": "Learning rate min in schedule."}
    )

    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Batch size per GPU for training."}
    )

    per_device_eval_batch_size: int = field(
        default=None, metadata={"help": "Batch size per GPU for evaluation."}
    )

    micro_train_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Micro batch size for training."}
    )

    num_train_steps: Optional[int] = field(
        default=None, metadata={"help": "Number of training steps."}
    )

    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of samples to train on."}
    )

    train_on_source: bool = field(
        default=False, metadata={"help": "Whether to train on source."}
    )

    train_on_every_token: bool = field(
        default=False,
        metadata={"help": "Whether to train on every token for reward modeling."},
    )

    warmup_ratio: float = field(
        default=0.2,
        metadata={"help": "Linear warmup ratio for learning rate scheduler."},
    )

    tensor_parallel_size: Optional[int] = field(
        default=None, metadata={"help": "Tensor parallel size."}
    )

    vocab_parallel: bool = field(
        default=False, metadata={"help": "Whether to use vocabulary parallelism."}
    )

    sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether to use sequence parallelism for activations."},
    )

    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1."})

    adam_beta2: float = field(default=0.999, metadata={"help": "Adam beta2."})

    adam_eps: float = field(default=1e-8, metadata={"help": "Adam epsilon."})

    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay."})

    optimizer_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload optimizer states to CPU during fine-tuning."
        },
    )

    # Reward modeling

    reward_head_init_scheme: str = field(
        default="zeros",
        metadata={"help": "Whether to initialize the reward head with zeros."},
    )

    process_reward_with_answer: bool = field(
        default=False,
        metadata={"help": "Whether to apply process reward with answer."},
    )

    # Direct Preference Optimization

    dpo_variant: str = field(
        default="dpo",
        metadata={"help": "Which variant of DPO to use."},
    )

    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "Beta for DPO."},
    )

    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "Label smoothing for DPO."},
    )

    dpo_pm_checkpoint_path: Optional[Path] = field(
        default=None,
        metadata={"help": "Path to the preference model checkpoint to load."},
    )

    dpo_pm_total_max_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of total sequence for preference model."},
    )

    # Mixed precision

    param_dtype: str = field(
        default="bf16",
        metadata={"help": "Parameter datatype for mixed precision training."},
    )

    compute_dtype: str = field(
        default="bf16",
        metadata={"help": "Reduce operation datatype for mixed precision training."},
    )

    optim_dtype: Optional[str] = field(
        default=None,
        metadata={"help": "Optimizer state datatype for mixed precision training."},
    )

    # Checkpointing

    checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the checkpoint to load."}
    )

    sft_checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the sft checkpoint to load."}
    )

    rm_checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the reward model checkpoint to load."}
    )

    # save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"steps"`):
    #     The checkpoint save strategy to adopt during training. Possible values are:

    #         - `"no"`: No save is done during training.
    #         - `"epoch"`: Save is done at the end of each epoch.
    #         - `"steps"`: Save is done every `save_steps`.

    save_strategy: str = field(
        default="no",
        metadata={"help": "The checkpoint save strategy to use."},
    )

    save_dir: Optional[Path] = field(
        default=None, metadata={"help": "Directory to save checkpoints to."}
    )

    save_steps: Optional[int] = field(
        default=None, metadata={"help": "Save checkpoint every X steps."}
    )

    eval_steps: Optional[int] = field(
        default=None, metadata={"help": "Evaluate every X steps."}
    )

    resume_from_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to resume from checkpoint."}
    )

    save_total_limit: int = field(
        default=1, metadata={"help": "Maximum number of checkpoints to save."}
    )

    save_only_model: bool = field(
        default=False, metadata={"help": "Whether to only save the model."}
    )

    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )

    wandb_name: Optional[str] = field(
        default=None, metadata={"help": "Name of the wandb run."}
    )

    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Name of the wandb project."}
    )

    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "Name of the wandb entity."}
    )

    # Misc

    compile: bool = field(
        default=False, metadata={"help": "Compile the model forward function."}
    )

    profile: Optional[Path] = field(
        default=None,
        metadata={
            "help": "Profile the model forward function and save the results to the given path."
        },
    )

    seed: int = field(default=42, metadata={"help": "Random seed."})

    print_training_examples: bool = field(
        default=False, metadata={"help": "Whether to print training examples."}
    )

    # PPO

    temperature: float = field(
        default=1.0, metadata={"help": "Temperature for rollout in PPO."}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )

    rollout_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "How many rollouts to accumulate before PPO update"},
    )

    noptepochs: int = field(
        default=2, metadata={"help": "Number of epochs in a single PPO update."}
    )

    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Maximum gradient norm for clipping."}
    )

    policy_model_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to use Fully Sharded Data Parallelism for policy."},
    )

    policy_model_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload policy (params, grads, optim_states) to CPU."
        },
    )

    policy_optimizer_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload policy optimizer states to CPU."},
    )

    value_model_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to use Fully Sharded Data Parallelism for value."},
    )

    value_model_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload value (params, grads, optim_states) to CPU."
        },
    )

    value_optimizer_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload value optimizer states to CPU."},
    )

    reward_model_fsdp: bool = field(
        default=False,
        metadata={"help": "Whether to use Fully Sharded Data Parallelism for reward."},
    )

    reward_model_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload reward (params, grads, optim_states) to CPU."
        },
    )

    reward_model_bits: int = field(
        default=16,
        metadata={"help": "Number of bits for reward model quantization."},
    )

    ref_policy_model_fsdp: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Fully Sharded Data Parallelism for ref policy."
        },
    )

    ref_policy_model_cpu_offload: bool = field(
        default=False,
        metadata={
            "help": "Whether to offload ref policy (params, grads, optim_states) to CPU."
        },
    )

    ref_policy_model_bits: int = field(
        default=16,
        metadata={"help": "Number of bits for ref policy model quantization."},
    )

    ppo_warmup_steps: int = field(
        default=5, metadata={"help": "Number of warmup steps for PPO."}
    )

    rollout_batch_size: int = field(
        default=32, metadata={"help": "Batch size for rollouts."}
    )

    rollout_per_device_batch_size: int = field(
        default=1, metadata={"help": "Per device batch size for rollouts."}
    )

    reward_model_per_device_batch_size: int = field(default=None)

    step_batch_size: int = field(
        default=32, metadata={"help": "Batch size for gradient updates."}
    )

    step_per_device_batch_size: int = field(
        default=1, metadata={"help": "Per device batch size for gradient updates."}
    )

    penalize_no_stop_token: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token."
        },
    )

    penalty_reward_value: float = field(
        default=-1.0,
        metadata={
            "help": "Reward assigned to sequences that are truncated, "
            "e.g., due to outputting incomplete sentences for given context window."
        },
    )

    relative_stop_token_penalty: bool = field(
        default=False,
        metadata={
            "help": "Whether to penalize sequences that do not contain stop_token "
            "with a relative penalty based on the original reward."
        },
    )

    maxent_normalization: str = field(
        default="full",
        metadata={"help": "Normalization for maxent regularization."},
    )
    maxent_coef: float = field(default=0.0)
    ent_reg_coef: float = field(default=0.0)
    kl_coef: float = field(default=0.2)
    kl_approximator: str = field(default="k1")
    whiten_rewards: bool = field(default=True)
    whitening_async_stats: str = field(
        default="per_gpu",
        metadata={"help": "How to sync statistics for advantage whitening."},
    )
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    stop_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token to stop generation with."},
    )

    min_seq_len: int = field(
        default=0, metadata={"help": "Minimum sequence length for reward modeling."}
    )
    min_seq_len_coef: float = field(
        default=0.0,
        metadata={"help": "Coefficient for penalizing too short sequences."},
    )

    base_checkpoint_path: Path = field(
        default="Undefined", metadata={"help": "Path to the policy base model to load."}
    )

    reward_base_checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the reward base model to load."}
    )

    value_base_checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the value base model to load."}
    )

    policy_checkpoint_path: Path = field(
        default="Undefined", metadata={"help": "Path to the sft model to load."}
    )

    reward_checkpoint_path: Path = field(
        default="Undefined", metadata={"help": "Path to the reward model to load."}
    )

    value_checkpoint_path: Optional[Path] = field(
        default=None, metadata={"help": "Path to the value model to load."}
    )

    init_value_with_reward: bool = field(
        default=True,
        metadata={"help": "Initialize the value model with the reward model."},
    )

    outcome_reward: bool = field(
        default=False,
        metadata={"help": "Whether to use outcome reward."},
    )

    easy_outcome_reward: bool = field(
        default=False,
        metadata={"help": "Whether to use outcome reward on easy problems."},
    )

    fsdp_consolidate_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to offload FSDP consolidation to CPU."},
    )

    # Process Reward Model in PPO

    apply_process_reward: bool = field(
        default=False,
        metadata={"help": "Whether to apply process reward in PPO."},
    )

    apply_terminal_process_reward: bool = field(
        default=False,
        metadata={"help": "Whether to apply aggregated process reward in PPO."},
    )

    process_reward_upper_bound: float = field(
        default=1.0,
        metadata={"help": "Upper bound for process reward."},
    )

    process_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scale for process reward."},
    )

    process_reward_scheme: str = field(
        default="min",
        metadata={"help": "How to compute process reward."},
    )

    minimal_reasoning_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of minimal reasoning steps."},
    )

    truncate_on_negative_step: bool = field(
        default=False,
        metadata={"help": "Whether to truncate on negative step."},
    )

    slow_cross_node_comm: bool = field(
        default=False,
        metadata={"help": "Whether to optimize for cross-node communication."},
    )

    def __post_init__(self):
        if self.param_dtype == "bf16":
            self.param_dtype = torch.bfloat16
        elif self.param_dtype == "fp32":
            self.param_dtype = torch.float32
        else:
            raise ValueError(f"Unknown param_dtype: {self.param_dtype}")

        if self.compute_dtype == "bf16":
            self.compute_dtype = torch.bfloat16
        elif self.compute_dtype == "fp32":
            self.compute_dtype = torch.float32
        else:
            raise ValueError(f"Unknown compute_dtype: {self.compute_dtype}")

        if self.optim_dtype is None:
            self.optim_dtype = self.param_dtype
        elif self.optim_dtype == "bf16":
            self.optim_dtype = torch.bfloat16
        elif self.optim_dtype == "fp32":
            self.optim_dtype = torch.float32
        else:
            raise ValueError(f"Unknown optim_dtype: {self.optim_dtype}")

        if self.base_checkpoint_path.name != "Undefined":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                logger.warning(f"Base checkpoint path: {self.base_checkpoint_path}")

            world_size = int(os.environ.get("WORLD_SIZE", 1))

            # Checks on rollout_batch_size only matter for PPO.
            assert (
                self.rollout_batch_size
                >= self.rollout_per_device_batch_size * world_size
            ), (
                "rollout_batch_size is smaller than rollout_per_device_batch_size * world_size. "
                "Increase the former or decrease the latter to fix this."
            )
            assert (
                self.rollout_batch_size
                % (self.rollout_per_device_batch_size * world_size)
                == 0
            ), "rollout_batch_size is not a multiple of rollout_per_device_batch_size * world_size. "

            assert (
                self.step_batch_size >= self.step_per_device_batch_size * world_size
            ), (
                "step_batch_size is smaller than step_per_device_batch_size * world_size. "
                "Increase the former or decrease the latter to fix this."
            )
            assert (
                self.step_batch_size % (self.step_per_device_batch_size * world_size)
                == 0
            ), "step_batch_size is not a multiple of step_per_device_batch_size * world_size. "

            if local_rank == 0:
                logger.warning(
                    f"Rollout stats:\n"
                    f"\trollout_batch_size: {self.rollout_batch_size}\n"
                    f"\trollout_per_device_batch_size: {self.rollout_per_device_batch_size}\n"
                    f"\tworld_size: {world_size}\n",
                )
            assert (
                self.rollout_batch_size // self.rollout_per_device_batch_size
            ) % world_size == 0
            self.rollout_accumulation_steps = (
                self.rollout_batch_size
                // self.rollout_per_device_batch_size
                // world_size
            )

            if local_rank == 0:
                logger.warning(
                    f"Step stats:\n"
                    f"\tstep_batch_size: {self.step_batch_size}\n"
                    f"\tstep_per_device_batch_size: {self.step_per_device_batch_size}\n"
                    f"\tworld_size: {world_size}\n",
                )
            assert (
                self.step_batch_size // self.step_per_device_batch_size
            ) % world_size == 0
            self.gradient_accumulation_steps = (
                self.step_batch_size // self.step_per_device_batch_size // world_size
            )

            if local_rank == 0:
                logger.warning(
                    f"Accumulation steps:\n"
                    f"\trollout_accumulation_steps: {self.rollout_accumulation_steps}\n"
                    f"\tgradient_accumulation_steps: {self.gradient_accumulation_steps}\n"
                )
