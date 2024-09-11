# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import glob

from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)

from models.model import Transformer
from models.reward_model import (
    RewardModel,
    apply_reward_modeling_head,
)
from models.tp import (
    apply_tp,
    apply_reward_head_tp,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_group,
)

from arguments import Arguments
from training_utils.fsdp_utils import (
    fixed_full_optim_state_dict,
    fixed_scatter_full_optim_state_dict,
)


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def get_trainable_state_dict(
    model: Union[torch.nn.Module, FSDP],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    use_fsdp: bool = False,
    trainable_param_names: Optional[list] = None,
    save_only_model: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Returns a dictionary containing the trainable state of the model and optimizer.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.

    Returns:
        dict: A dictionary containing the trainable state of the model and optimizer.
    """
    scheduler_state_dict = scheduler.state_dict()

    if not use_fsdp:
        model: torch.nn.Module = model
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
    else:
        model: FSDP = model
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=cfg,
            optim_state_dict_config=optim_cfg,
        ):
            model_state_dict = model.state_dict()

            if save_only_model:
                # convert model_state_dict to bf16
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to(
                        dtype=compute_dtype
                    )
                optimizer_state_dict = None
            else:
                optimizer_state_dict = fixed_full_optim_state_dict(
                    model=model,
                    optim=optimizer,
                    rank0_only=True,
                    group=get_data_parallel_group(),
                )

        # if get_data_parallel_rank() > 0:
        #     assert model_state_dict is None or len(model_state_dict) == 0
        #     assert optimizer_state_dict is None or len(optimizer_state_dict) == 0

    # we only keep model parameters that are trainable

    if get_data_parallel_rank() == 0:
        if use_fsdp:
            trainable_param_names = trainable_param_names
        else:
            trainable_param_names = [
                name for name, param in model.named_parameters() if param.requires_grad
            ]

        trainable_model_state_dict = {}
        if trainable_param_names is not None:
            trainable_param_names = set(trainable_param_names)

            for name, param in model_state_dict.items():
                if name in trainable_param_names:
                    trainable_model_state_dict[name] = param
                    trainable_param_names.remove(name)

            if len(trainable_param_names) > 0:
                rank0_print(
                    f"Missing {len(trainable_param_names)} parameters in model state dict:"
                )
                for name in trainable_param_names:
                    rank0_print(name)
                raise ValueError("Missing parameters in model state dict")
        else:
            trainable_model_state_dict = model_state_dict

        return {
            "model": trainable_model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }
    else:
        return None


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    rank: int,
    save_only_model: bool,
    save_total_limit: int,
    use_fsdp: bool = False,
    trainable_param_names: Optional[list] = None,
    prefix: str = "",
    metrics: Optional[dict] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Saves a checkpoint.

    Args:
        checkpoint_path (str): The path to save the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save.
        epoch (int): The epoch to save.
        global_step (int): The global step to save.
    """
    state_dict = get_trainable_state_dict(
        model,
        optimizer,
        scheduler,
        use_fsdp,
        trainable_param_names,
        save_only_model,
        compute_dtype,
    )

    if state_dict is not None:
        if save_only_model:
            new_state_dict = {}
            new_state_dict["model"] = state_dict["model"]
            state_dict = new_state_dict

        state_dict["epoch"] = epoch or -1
        state_dict["global_step"] = global_step
        if metrics is not None:
            state_dict["metrics"] = metrics
        torch.save(state_dict, checkpoint_path)

    if dist.is_initialized():
        dist.barrier()

    if state_dict is None:
        return

    # write last_checkpoint
    save_dir = "/".join(checkpoint_path.split("/")[:-1])

    if rank == 0:
        with open(f"{save_dir}/{prefix}last_checkpoint", "w") as f:
            f.write(checkpoint_path.replace("_rank_0", "_rank*").split("/")[-1])

    def extract_step_number(filename):
        # Use a regular expression to find the number after 'step_'
        match = re.search(r"step_(\d+)", filename)
        if match:
            # Return the number as an integer
            return int(match.group(1))
        else:
            raise ValueError(f"Invalid checkpoint filename: {filename}")

    if save_total_limit is not None:
        checkpoints = glob.glob(f"{save_dir}/{prefix}*_rank_{rank}.pt")
        checkpoints = sorted(checkpoints, key=extract_step_number)

        if len(checkpoints) > save_total_limit:
            message = checkpoints[0].replace("_rank_0", "_rank*")
            rank0_print(f"Removing {message}")
            os.remove(checkpoints[0])


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    use_fsdp: bool = False,
) -> None:
    """Loads a checkpoint.

    Args:
        checkpoint_path (str): The path to load the checkpoint.
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load.
    """
    load_rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    pattern = checkpoint_path.replace(f"_rank_{load_rank}", "_rank_*")
    ckpt_world_size = len(glob.glob(pattern))

    if ckpt_world_size != world_size:
        raise ValueError(
            f"The model parallel setting of resuming training "
            "does not match the current setting: {ckpt_world_size} (train) vs {world_size} (resume)"
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)

    if not use_fsdp:
        model.load_state_dict(
            state_dict["model"],
            strict=False,
        )

        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])
    else:
        model: FSDP = model
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=cfg,
        ):
            model.load_state_dict(
                state_dict["model"],
                strict=False,
            )

        if get_data_parallel_rank() == 0:
            full_osd = state_dict["optimizer"]
        else:
            full_osd = None

        sharded_osd = fixed_scatter_full_optim_state_dict(
            full_optim_state_dict=full_osd,
            model=model,
            optim=optimizer,
            group=get_data_parallel_group(),
        )
        optimizer.load_state_dict(sharded_osd)

    if scheduler is not None:
        try:
            scheduler.load_state_dict(state_dict["scheduler"])
        except KeyError:
            scheduler.step(state_dict["global_step"])

    ret_dict = {
        "epoch": state_dict["epoch"],
        "global_step": state_dict["global_step"],
    }

    if "metrics" in state_dict:
        ret_dict["metrics"] = state_dict["metrics"]
    return ret_dict


def load_inference_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
) -> None:
    """Loads a checkpoint.

    Args:
        checkpoint_path (str): The path to load the checkpoint.
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to load.
    """
    if not torch.distributed.is_initialized():
        state_dict = torch.load(checkpoint_path)["model"]
    else:
        load_rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()

        if f"_rank_{load_rank}" in checkpoint_path:
            pattern = checkpoint_path.replace(f"_rank_{load_rank}", "_rank_*")
        else:
            pattern = checkpoint_path

        ckpt_world_size = len(glob.glob(pattern))
        assert ckpt_world_size > 0, f"No checkpoint found: '{pattern}'"

        if ckpt_world_size == world_size:
            state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)[
                "model"
            ]

        elif ckpt_world_size > world_size:
            assert ckpt_world_size % world_size == 0

            dup_factor = ckpt_world_size // world_size

            all_state_dict = []

            for i in range(dup_factor):
                ckpt_file_path = pattern.replace("*", str(load_rank * dup_factor + i))
                model_state_dict = torch.load(
                    ckpt_file_path, map_location="cpu", mmap=True
                )["model"]
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].cpu()
                all_state_dict.append(model_state_dict)

            state_dict = merge_tp_checkpoint(all_state_dict)

        else:
            assert world_size % ckpt_world_size == 0

            split_factor = world_size // ckpt_world_size
            split_idx = load_rank % split_factor
            ckpt_file_path = pattern.replace("*", str(load_rank // split_factor))
            if "model" in torch.load(ckpt_file_path, map_location="cpu", mmap=True):
                full_state_dict = torch.load(ckpt_file_path, map_location="cpu", mmap=True)[
                    "model"
                ]
            else:
                full_state_dict = torch.load(ckpt_file_path, map_location="cpu", mmap=True)
            for key in full_state_dict:
                full_state_dict[key] = full_state_dict[key].cpu()
            state_dict = split_tp_checkpoint(full_state_dict, split_factor, split_idx)

    if list(state_dict.keys())[0].startswith("module."):
        assert all([k.startswith("module.") for k in state_dict.keys()])
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    # remove size un-matching parameters
    model_state_dict = model.state_dict()

    # If any of state in model starts with "backbone_model.", and the layers in checkpoint do not have it.
    # It means we need to add "backbone_model." back to the checkpoint
    if any([k.startswith("backbone_model.") for k in model_state_dict.keys()]) and all([not k.startswith("backbone_model.") for k in state_dict.keys()]):
        state_dict = {f"backbone_model.{k}": v for k, v in state_dict.items()}

    for key in list(state_dict.keys()):
        if ("output.weight" in key and state_dict[key].size(0) != model_state_dict[key].size(0) and model_state_dict[key].size(0) == 2):
            # When loading the reward model, we need to apply the reward modeling head
            # 2 means the reward modeling head is a binary classifier
            print("Applying reward modeling head to the loaded checkpoint ...")
            state_dict[key] = state_dict[key][: model_state_dict[key].size(0)]
        if key not in model_state_dict or model_state_dict[key].size() != state_dict[key].size():
            rank0_print("Warning: removing", key, "from checkpoint.")
            del state_dict[key]

    model.load_state_dict(
        state_dict,
        strict=False,
    )

    return None, None


def merge_tp_checkpoint(
    all_state_dict: list,
):
    state_dict = {}

    all_keys = list(all_state_dict[0].keys())

    for key in all_keys:
        if "wqkv" in key:
            if all_state_dict[0][key].size(1) == 4096:
                n_heads, n_local_heads = 32, 32
            elif all_state_dict[0][key].size(1) == 5120:
                n_heads, n_local_heads = 40, 40
            elif all_state_dict[0][key].size(1) == 6656:
                n_heads, n_local_heads = 52, 52
            elif all_state_dict[0][key].size(1) == 8192:
                n_heads, n_local_heads = 64, 8
            else:
                raise ValueError(
                    f"Invalid size for {key}: {all_state_dict[0][key].size(1)}"
                )

            head_dim = all_state_dict[0][key].size(0) // (n_heads + n_local_heads * 2)

            weight_splits = [
                head_dim * n_heads,
                head_dim * n_local_heads,
                head_dim * n_local_heads,
            ]

            merged_q, merged_k, merged_v = [], [], []
            for x in all_state_dict:
                q, k, v = x[key].split(weight_splits, dim=0)
                merged_q.append(q)
                merged_k.append(k)
                merged_v.append(v)

            merged_q = torch.cat(merged_q, dim=0)
            merged_k = torch.cat(merged_k, dim=0)
            merged_v = torch.cat(merged_v, dim=0)

            state_dict[key] = torch.cat([merged_q, merged_k, merged_v], dim=0)

        elif "wo" in key or "w2" in key:
            state_dict[key] = torch.cat([x[key] for x in all_state_dict], dim=1)
        elif "output.weight" in key:
            state_dict[key] = torch.cat([x[key] for x in all_state_dict], dim=1)
        elif "norm" in key or "tok_embeddings" in key:
            state_dict[key] = all_state_dict[0][key]
        else:
            state_dict[key] = torch.cat([x[key] for x in all_state_dict], dim=0)

        # free the memory
        for x in all_state_dict:
            del x[key]

    return state_dict


def split_tp_checkpoint(
    full_state_dict: dict,
    split_factor: int,
    split_idx: int,
):
    state_dict = {}

    all_keys = list(full_state_dict.keys())

    for key in all_keys:
        if "wqkv" in key:
            if full_state_dict[key].size(1) == 4096:
                n_heads, n_local_heads = 32, 32
            elif full_state_dict[key].size(1) == 5120:
                n_heads, n_local_heads = 40, 40
            elif full_state_dict[key].size(1) == 6656:
                n_heads, n_local_heads = 52, 52
            elif full_state_dict[key].size(1) == 8192:
                n_heads, n_local_heads = 64, 8
            else:
                raise ValueError(
                    f"Invalid size for {key}: {full_state_dict[key].size(1)}"
                )

            head_dim = full_state_dict[key].size(0) // (n_heads + n_local_heads * 2)

            weight_splits = [
                head_dim * n_heads,
                head_dim * n_local_heads,
                head_dim * n_local_heads,
            ]

            q, k, v = full_state_dict[key].split(weight_splits, dim=0)

            q = torch.tensor_split(q, split_factor, dim=0)[split_idx]
            k = torch.tensor_split(k, split_factor, dim=0)[split_idx]
            v = torch.tensor_split(v, split_factor, dim=0)[split_idx]

            state_dict[key] = torch.cat([q, k, v], dim=0)

        elif "wo" in key or "w2" in key:
            state_dict[key] = torch.tensor_split(
                full_state_dict[key], split_factor, dim=1
            )[split_idx]
        elif "output.weight" in key:
            state_dict[key] = torch.tensor_split(
                full_state_dict[key], split_factor, dim=1
            )[split_idx]
        elif "norm" in key or "tok_embeddings" in key:
            state_dict[key] = full_state_dict[key]
        else:
            state_dict[key] = torch.tensor_split(
                full_state_dict[key], split_factor, dim=0
            )[split_idx]

        # free the memory
        del full_state_dict[key]

    return state_dict


def get_checkpoint_path(
    checkpoint_dir: str,
    epoch: Optional[int],
    global_step: int,
    tp_rank: int,
    prefix: str,
) -> str:
    """Returns the path to save the checkpoint.

    Args:
        checkpoint_dir (str): The directory to save the checkpoint.
        epoch (int): The epoch to save.
        global_step (int): The global step to save.

    Returns:
        str: The path to save the checkpoint.
    """
    if epoch is None:
        return f"{checkpoint_dir}/{prefix}step_{global_step}_rank_{tp_rank}.pt"
    else:
        return f"{checkpoint_dir}/{prefix}epoch_{epoch}_step_{global_step}_rank_{tp_rank}.pt"


def get_latest_checkpoint_path(
    checkpoint_dir: str, prefix: Optional[str] = None
) -> str:
    """Search and return the path to the latest checkpoint.

    Args:
        checkpoint_dir (str): The directory to save the checkpoint.
        prefix (str): The prefix of the checkpoint.

    Returns:
        str: The path to the latest checkpoint.
        int: The epoch to save.
        int: The global step to save.
    """
    if prefix is None:
        prefix = ""

    if not os.path.exists(f"{checkpoint_dir}/{prefix}last_checkpoint"):
        return None, 0, 0

    with open(f"{checkpoint_dir}/{prefix}last_checkpoint", "r") as f:
        last_checkpoint_file = f.read()

    last_checkpoint_file = last_checkpoint_file.split("/")[-1]
    last_checkpoint_file = os.path.join(checkpoint_dir, last_checkpoint_file)
    last_checkpoint_file = last_checkpoint_file.replace("_rank*", "_rank_0")

    if dist.is_initialized():
        rank = get_model_parallel_rank()
        last_checkpoint_file = last_checkpoint_file.replace("_rank_0", f"_rank_{rank}")

    epoch = None
    if "epoch_" in last_checkpoint_file:
        match = re.search(r"epoch_(\d+)", last_checkpoint_file)
        if match:
            epoch = int(match.group(1))
        else:
            raise ValueError(f"Invalid checkpoint filename: {last_checkpoint_file}")

    if "step_" in last_checkpoint_file:
        match = re.search(r"step_(\d+)", last_checkpoint_file)
        if match:
            global_step = int(match.group(1))
        else:
            raise ValueError(f"Invalid checkpoint filename: {last_checkpoint_file}")

    last_checkpoint_file = last_checkpoint_file.strip()
    return last_checkpoint_file, epoch, global_step


def checkpoint_hook(
    args: Arguments,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    global_step: int,
    epoch_length: int,
    use_fsdp: bool = False,
    trainable_param_names: Optional[list] = None,
    prefix: str = "",
    metrics: Optional[dict] = None,
):
    if args.save_strategy == "no":
        return

    rank = 0

    if dist.is_initialized():
        rank = get_model_parallel_rank()

    # make sure the checkpoint dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    save_flag = False
    if args.save_strategy == "epoch":
        if global_step % epoch_length == 0 and global_step != 0:
            save_flag = True
    elif args.save_strategy == "steps":
        if global_step % args.save_steps == 0 and global_step != 0:
            save_flag = True
    elif metrics is not None:
        save_flag = True
    else:
        raise ValueError(f"Invalid save strategy: {args.save_strategy}")

    if save_flag:
        checkpoint_path = get_checkpoint_path(
            args.save_dir, epoch, global_step, rank, prefix
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            rank,
            args.save_only_model,
            args.save_total_limit,
            use_fsdp,
            trainable_param_names,
            prefix,
            metrics,
            args.compute_dtype,
        )


def load_model_from_from_ckpt(
    checkpoint_path: Path,
    sft_checkpoint_path: Optional[Path],
    device: torch.device,
    precision: torch.dtype,
    use_tp: bool,
    requires_grad: bool,
    skip_init: bool = False,
    sequence_parallel: bool = False,
    vocab_parallel: bool = False,
):
    with torch.device("meta"):
        model = Transformer.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_output=True,
            freeze_norm=True,
            vocab_parallel=vocab_parallel,
        )

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for finetuning!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for finetuning!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        rank0_print("Applying tensor parallel to model ...")
        apply_tp(
            model, requires_grad=requires_grad, sequence_parallel=sequence_parallel
        )

    if not skip_init:
        if str(sft_checkpoint_path).endswith(".pth"):
            rank0_print(
                f"Loading sft model from {str(sft_checkpoint_path)} ..."
            )
            load_inference_checkpoint(str(sft_checkpoint_path), model)
        else:
            print(sft_checkpoint_path)
            sft_checkpoint_path, _, _ = get_latest_checkpoint_path(sft_checkpoint_path)
            if sft_checkpoint_path is not None:
                rank0_print(
                    f"Loading sft model from {sft_checkpoint_path.replace(f'_rank_0', '_rank_*')} ..."
                )
                load_inference_checkpoint(sft_checkpoint_path, model)
            else:
                rank0_print(
                    "Warning: no sft checkpoint found, using base checkpoint."
                    " (OK for unwrapped policy / resuming training / train from scratch)."
                )

    model.requires_grad_(requires_grad)
    model = model.to(device=device, dtype=precision)
    return model.train(requires_grad)


def load_reward_model_from_ckpt(
    checkpoint_path: Path,
    rm_checkpoint_path: Optional[Path],
    device: torch.device,
    precision: torch.dtype,
    use_tp: bool,
    requires_grad: bool,
    skip_init: bool = False,
    sequence_parallel: bool = False,
    vocab_parallel: bool = False,
):
    with torch.device("meta"):
        model = RewardModel.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_norm=True,
            vocab_parallel=vocab_parallel,
        )

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for finetuning!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for finetuning!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.backbone_model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        rank0_print("Applying tensor parallel to model ...")
        apply_tp(
            model.backbone_model,
            requires_grad=requires_grad,
            sequence_parallel=sequence_parallel,
        )

    apply_reward_modeling_head(model.backbone_model, requires_grad=requires_grad)

    if use_tp:
        rank0_print("Applying tensor parallel to reward head ...")
        apply_reward_head_tp(model.backbone_model, requires_grad=requires_grad)

    if not skip_init:
        if str(rm_checkpoint_path).endswith(".pth"):
            rank0_print(
                f"Loading reward model from {str(rm_checkpoint_path)} ..."
            )
            load_inference_checkpoint(str(rm_checkpoint_path), model)
        else:
            rm_checkpoint_path, _, _ = get_latest_checkpoint_path(rm_checkpoint_path)
            if rm_checkpoint_path is not None:
                rank0_print(
                    f"Loading reward model from {rm_checkpoint_path.replace(f'_rank_0', '_rank_*')} ..."
                )
                load_inference_checkpoint(rm_checkpoint_path, model)
                rank0_print("Reward head", model.backbone_model.output.weight)
            else:
                rank0_print("Warning: no rm checkpoint found, using base checkpoint.")

    model.requires_grad_(requires_grad)
    model = model.to(device=device, dtype=precision)
    return model.train(requires_grad)


def load_reward_model_from_sft_ckpt(
    checkpoint_path: Path,
    sft_checkpoint_path: Optional[Path],
    device: torch.device,
    precision: torch.dtype,
    use_tp: bool,
    requires_grad: bool,
    reward_head_init_scheme: str = "zeros",
    sequence_parallel: bool = False,
    vocab_parallel: bool = False,
):
    with torch.device("meta"):
        model = RewardModel.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_norm=True,
            vocab_parallel=vocab_parallel,
        )

    if "int8" in str(checkpoint_path):
        raise NotImplementedError("int8 quantization cannot be used for finetuning!")

    if "int4" in str(checkpoint_path):
        raise NotImplementedError("int4 quantization cannot be used for finetuning!")

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.backbone_model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(
            model.backbone_model,
            requires_grad=True,
            sequence_parallel=sequence_parallel,
        )

    sft_checkpoint_path, _, _ = get_latest_checkpoint_path(sft_checkpoint_path)
    if sft_checkpoint_path is not None:
        print("Loading sft model ...")
        load_inference_checkpoint(sft_checkpoint_path, model.backbone_model)

    apply_reward_modeling_head(
        model.backbone_model,
        requires_grad=requires_grad,
        init_sceheme=reward_head_init_scheme,
    )

    if use_tp:
        print("Applying tensor parallel to reward head ...")
        apply_reward_head_tp(model.backbone_model, requires_grad=requires_grad)

    model.requires_grad_(requires_grad)
    model = model.to(device=device, dtype=precision)
    return model.train(requires_grad)
