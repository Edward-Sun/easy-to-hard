# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
from pathlib import Path
import logging

import torch

import torch._inductor.config
import torch._dynamo.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import wandb
except ImportError:
    wandb = None

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.tokenizer_utils import AcceleraTokenizer
from models.tp import (
    maybe_init_dist,
    initialize_model_parallel,
)
from trainers.ppo_trainer import PPOTrainer, make_models
from trainers.common_utils import manual_seed

from data_utils.data_utils_ppo import make_rl_data_module

from hf_argparser import HfArgumentParser
from arguments import Arguments as TrainingArguments
from checkpoint_utils import get_latest_checkpoint_path

logger = logging.getLogger(__name__)


def main(args: TrainingArguments):
    base_model_name_or_path = args.base_checkpoint_path
    tokenizer_path = base_model_name_or_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = base_model_name_or_path.parent

    global print
    device_id = maybe_init_dist()
    use_tp = device_id is not None
    if use_tp:
        tp_size = args.tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        torch.distributed.barrier()
        if device_id != 0:
            # only print on rank 0
            print = lambda *_args, **_kwargs: None

    checkpoint_dir, _, _ = get_latest_checkpoint_path(args.save_dir, prefix="policy_")
    checkpoint_dir = Path(checkpoint_dir).parent if checkpoint_dir is not None else None

    torch.distributed.barrier()
    if args.report_to == "wandb" and wandb is not None:
        if device_id == 0:
            wandb_logging_dir = os.path.join(
                tempfile.gettempdir(), f"{os.getuid()}_wandb"
            )
            if not os.path.exists(wandb_logging_dir):
                os.makedirs(wandb_logging_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = wandb_logging_dir
            wandb.init(
                name=args.wandb_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                resume="allow",
                magic=True,
                dir=wandb_logging_dir,
                force=True,
            )
            wandb.config.update(vars(args))

    if checkpoint_dir is None:
        print("Training from scratch.")
    else:
        print("Loading from checkpoint:", checkpoint_dir)

    tokenizer = AcceleraTokenizer(tokenizer_path)
    tokenizer.pad_id = tokenizer.unk_id

    manual_seed(args.seed)

    data_module: dict = make_rl_data_module(tokenizer, args)

    for i in range(3):
        token_ids = data_module["train_dataset"][i]["queries"]
        print(tokenizer.decode(token_ids, skip_special_tokens=True))
        print("=" * 20)

    model_module = make_models(
        tokenizer,
        args,
        resume_from_checkpoint=(
            checkpoint_dir if args.resume_from_checkpoint else None
        ),
    )

    trainer = PPOTrainer(
        args=args,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )

    trainer.train(
        resume_training_ckpt=checkpoint_dir if args.resume_from_checkpoint else None
    )


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
