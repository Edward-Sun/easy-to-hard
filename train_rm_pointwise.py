# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import time
from pathlib import Path
import itertools

import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

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

from models.model import set_global_compile_mode
from models.reward_model import RewardModel
from models.tokenizer_utils import AcceleraTokenizer
from models.tp import (
    maybe_init_dist,
    initialize_model_parallel,
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_data_parallel_world_size,
    clip_grad_norm_,
)

from trainers.common_utils import manual_seed
from data_utils.data_utils_rm_pointwise import (
    make_pointwise_reward_modeling_data_module,
)

from hf_argparser import HfArgumentParser
from arguments import Arguments as TrainingArguments
from checkpoint_utils import (
    checkpoint_hook,
    get_latest_checkpoint_path,
    load_checkpoint,
    load_reward_model_from_sft_ckpt,
)
from training_utils.trainer_utils import (
    create_optimizer,
    create_fsdp_model_for_finetune,
    get_cosine_schedule_with_warmup,
)

IGNORE_INDEX = -100


def model_forward(model, x):
    return model(x)


def model_forward_with_loss(
    model: RewardModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the loss for a given model and prompts.
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    batch_size, T = input_ids.size(0), input_ids.size(1)

    device = input_ids.device
    with torch.device(device):
        model.backbone_model.setup_caches(
            max_batch_size=batch_size, max_seq_length=T, kv_cache=False
        )

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        logits = model_forward(model, input_ids)

    with torch.autocast(device_type="cuda", enabled=False):
        dtype_logits = logits.float()
        loss = F.binary_cross_entropy_with_logits(
            dtype_logits.view(-1).float(),
            labels.view(-1).float(),
            reduction="none",
        )

        weights = weights.view(-1).float()
        loss = (loss * weights).sum() / (weights.sum() + 1e-6)
    return loss


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def main(
    args: TrainingArguments,
) -> None:
    """Finetune a model on a given dataset."""
    checkpoint_path = args.checkpoint_path
    sft_checkpoint_path = args.sft_checkpoint_path
    compile = args.compile
    assert checkpoint_path.is_file(), checkpoint_path

    if sft_checkpoint_path is not None:
        assert sft_checkpoint_path.is_dir(), sft_checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent

    set_global_compile_mode(args.compile)

    global print
    device_id = maybe_init_dist()
    use_tp = device_id is not None
    if use_tp:
        tp_size = args.tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        torch.distributed.barrier()
        tp_group = get_model_parallel_group()

        if device_id != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

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

    device = "cuda"
    precision = args.param_dtype

    print("Loading model ...")
    t0 = time.time()

    resume_from_checkpoint = None
    resume_epoch = 0
    resume_global_step = 0

    if args.resume_from_checkpoint:
        (
            resume_from_checkpoint,
            resume_epoch,
            resume_global_step,
        ) = get_latest_checkpoint_path(args.save_dir)

    if resume_from_checkpoint is not None:
        sft_checkpoint_path = None

    model = load_reward_model_from_sft_ckpt(
        checkpoint_path,
        sft_checkpoint_path,
        device,
        precision,
        use_tp,
        requires_grad=True,
        sequence_parallel=args.sequence_parallel,
    )

    torch.cuda.synchronize()
    if use_tp:
        torch.distributed.barrier()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AcceleraTokenizer(tokenizer_path)

    data_module = make_pointwise_reward_modeling_data_module(
        tokenizer=tokenizer,
        args=args,
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    print(f"Model size: {model_size / 1e6:.02f} MB")
    manual_seed(args.seed)

    sampler = None
    if use_tp:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=data_collator,
    )

    if args.print_training_examples:
        print("Training examples:")
        cnt = 16
        for batch in train_loader:
            print(
                "Input:",
                tokenizer.decode(
                    batch["input_ids"][0].tolist(), skip_special_tokens=False
                ),
            )
            cnt -= 1
            if cnt == 0:
                break

    if compile:
        model = torch.compile(model)

    trainable_param_names = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    use_fsdp = False

    if get_data_parallel_world_size() > 1:
        use_fsdp = True
        model = create_fsdp_model_for_finetune(args, model)
        print("Using FSDP ...")
        print(model)

    optimizer = create_optimizer(
        args,
        model=model,
        optimizer_cpu_offload=args.optimizer_cpu_offload,
        model_cpu_offload=False,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=len(train_loader) * args.warmup_ratio,
        max_epochs=len(train_loader) * args.num_train_epochs,
        warmup_start_ratio=0.0,
        eta_min_ratio=args.lr_eta_min / args.learning_rate,
    )

    if resume_from_checkpoint is not None:
        print(
            f"Resuming from checkpoint: {resume_from_checkpoint} (epoch {resume_epoch}, global step {resume_global_step})"
        )
        load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, use_fsdp=use_fsdp
        )

    micro_train_batch_size = (
        args.micro_train_batch_size or args.per_device_train_batch_size
    )

    assert (
        args.per_device_train_batch_size % micro_train_batch_size == 0
    ), f"per_device_train_batch_size ({args.per_device_train_batch_size}) must be divisible by micro_train_batch_size ({micro_train_batch_size})"
    accumulate_steps = args.per_device_train_batch_size // micro_train_batch_size

    print(
        "Batch size per GPU for training: {}\n".format(
            args.per_device_train_batch_size
        ),
        "Micro batch size for training: {}\n".format(micro_train_batch_size),
        "Gradient accumulation steps: {}\n".format(accumulate_steps),
    )

    micro_train_batch_size = micro_train_batch_size * torch.distributed.get_world_size()

    epoch_length = len(train_loader)

    if args.do_train:
        print("Starting training ...")
        t0 = time.time()
        for epoch in tqdm.trange(
            args.num_train_epochs, desc="Epoch", disable=device_id != 0
        ):
            if sampler is not None:
                train_loader.sampler.set_epoch(epoch)
            pbar = tqdm.tqdm(
                enumerate(train_loader),
                desc="Iteration",
                disable=device_id != 0,
                total=len(train_loader),
            )
            for it, batch in pbar:
                global_step = epoch * epoch_length + it
                if global_step < resume_global_step:
                    continue

                # torch.cuda.synchronize()
                model.zero_grad()

                input_ids = batch["input_ids"].to(device=device)
                labels = batch["labels"].to(device=device)
                weights = batch["weights"].to(device=device)

                input_ids, labels, weights = prepare_batch(
                    input_ids,
                    labels,
                    weights,
                    tokenizer=tokenizer,
                    use_tp=use_tp,
                    sync_group=tp_group,
                )

                loss_scale = 1.0 / accumulate_steps
                for ex_idx in range(0, input_ids.size(0), micro_train_batch_size):
                    if ex_idx + micro_train_batch_size < input_ids.size(0):
                        with torch.cuda.amp.autocast(dtype=args.compute_dtype):
                            loss = model_forward_with_loss(
                                model,
                                input_ids[ex_idx : ex_idx + micro_train_batch_size],
                                labels[ex_idx : ex_idx + micro_train_batch_size],
                                weights[ex_idx : ex_idx + micro_train_batch_size],
                            )
                        (loss_scale * loss).backward()
                    else:
                        with torch.cuda.amp.autocast(dtype=args.compute_dtype):
                            loss = model_forward_with_loss(
                                model,
                                input_ids[ex_idx:],
                                labels[ex_idx:],
                                weights[ex_idx:],
                            )
                        (loss_scale * loss).backward()
                        grad_norm = clip_grad_norm_(model, 1.0)
                        optimizer.step()
                        scheduler.step()

                if it % 5 == 0:
                    loss_copy = loss.detach().clone()
                    torch.distributed.all_reduce(loss_copy)
                    avg_loss = (loss_copy / torch.distributed.get_world_size()).item()
                    grad_norm_copy = grad_norm.detach().clone().item()

                    if device_id == 0:
                        if args.report_to == "wandb" and wandb is not None:
                            wandb.log(
                                {
                                    "loss": avg_loss,
                                    "learning_rate": scheduler.get_last_lr()[0],
                                    "epoch": epoch,
                                    "step": it,
                                    "grad_norm": grad_norm_copy,
                                },
                                step=global_step,
                            )
                        else:
                            # Just print to stdout.
                            print(
                                {
                                    "loss": avg_loss,
                                    "learning_rate": scheduler.get_last_lr()[0],
                                    "epoch": epoch,
                                    "step": it,
                                    "grad_norm": grad_norm_copy,
                                }
                            )

                checkpoint_hook(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    epoch_length,
                    use_fsdp=use_fsdp,
                    trainable_param_names=trainable_param_names,
                )

        torch.cuda.synchronize()

        epoch = args.num_train_epochs

        checkpoint_hook(
            args,
            model,
            optimizer,
            scheduler,
            epoch,
            epoch * epoch_length,
            epoch_length,
            use_fsdp=use_fsdp,
            trainable_param_names=trainable_param_names,
        )

        print(f"Time to train: {time.time() - t0:.02f} seconds")


def prepare_batch(input_ids, labels, weights, tokenizer, use_tp, sync_group):
    pad_id = tokenizer.pad_id
    unk_id = tokenizer.unk_id
    # if pad_id < 0, replace pad_id with unk_id
    labels[labels == pad_id] = IGNORE_INDEX
    if pad_id < 0:
        input_ids[input_ids == pad_id] = unk_id

    if use_tp and get_model_parallel_world_size() > 1:
        # aggregate (concat) all the inputs across tp sync_group
        new_input_ids = torch.empty_like(input_ids).repeat(sync_group.size(), 1)
        new_labels = torch.empty_like(labels).repeat(sync_group.size(), 1)
        new_weights = torch.empty_like(weights).repeat(sync_group.size(), 1)

        torch.distributed.all_gather_into_tensor(
            new_input_ids, input_ids, group=sync_group
        )
        torch.distributed.all_gather_into_tensor(new_labels, labels, group=sync_group)
        torch.distributed.all_gather_into_tensor(new_weights, weights, group=sync_group)

        return new_input_ids, new_labels, new_weights

    return input_ids, labels, weights


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
