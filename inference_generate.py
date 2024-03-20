# Copyright 2024 The GPT-Accelera Team
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import gc
import os
import sys
from copy import deepcopy
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import OrderedDict
import itertools
import fcntl

import torch
from torch.distributed import _functional_collectives as funcol

import torch._inductor.config
import torch._dynamo.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.model import Transformer
from models.tp import maybe_init_dist, initialize_model_parallel, apply_tp
from models.tp import (
    get_model_parallel_rank,
    get_model_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from models.tokenizer_utils import (
    AcceleraTokenizer,
    batch_encode_tokens,
)
from checkpoint_utils import (
    get_latest_checkpoint_path,
    load_inference_checkpoint,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)
    # return torch.argmax(probs_sort, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        pivot = v.select(-1, -1).view(-1, 1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits, vocab_parallel, temperature: float = 1.0, top_k: Optional[int] = None
):
    with torch.autocast(device_type="cuda", enabled=False):
        logits = logits[:, -1].float()

        if vocab_parallel:
            logits = funcol.all_gather_tensor(
                logits, gather_dim=-1, group=get_model_parallel_group()
            )

        probs = logits_to_probs(logits, temperature, top_k)
        idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def remove_all_backward_hooks(model: torch.nn.Module) -> Dict[str, OrderedDict]:
    all_backward_hooks = {}

    for name, module in model.named_modules():
        all_backward_hooks[name] = module._backward_hooks
        module._backward_hooks = OrderedDict()

    return all_backward_hooks


def prefill(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos, left_pad_mask_pos)
    return sample(logits, model.vocab_parallel, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    num_new_tokens: int,
    eos_id: Optional[int] = None,
    **sampling_kwargs,
):
    eos_flag = None
    if eos_id is not None:
        eos_flag = torch.zeros_like(
            cur_token, dtype=torch.bool, device=cur_token.device
        )

    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, left_pad_mask_pos, **sampling_kwargs
            )
        input_pos += 1
        new_tokens.append(next_token.clone().view(-1, 1))
        new_probs.append(next_prob.clone().view(-1, 1))
        cur_token = next_token.view(-1, 1)

        if eos_flag is not None:
            eos_flag = eos_flag | (next_token == eos_id)

        if eos_flag is not None and eos_flag.all():
            break

    return new_tokens, new_probs, i


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    left_pad_mask_pos: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    B = prompt.size(0)
    T = prompt.size(1)
    T_new = T + max_new_tokens
    # max_seq_length = min(T_new, model.config.block_size)
    # max_seq_length = max_seq_len or model.config.block_size
    max_seq_length = model.config.block_size

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=B, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((B, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        next_token = prefill(
            model, prompt, input_pos, left_pad_mask_pos, **sampling_kwargs
        )

    seq[:, T] = next_token.view(B)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    generated_tokens, _, num_decoded_tokens = decode_n_tokens(
        model,
        next_token.view(B, -1),
        input_pos,
        left_pad_mask_pos,
        max_new_tokens - 1,
        eos_id,
        **sampling_kwargs,
    )

    generated_tokens = torch.cat(generated_tokens, dim=-1).view(B, -1)

    seq[:, T + 1 : T + 1 + generated_tokens.size(1)] = generated_tokens

    return seq, num_decoded_tokens


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        model = Transformer.from_name(
            checkpoint_path.parent.name,
            freeze_tok_embeddings=True,
            freeze_output=True,
            freeze_norm=True,
            vocab_parallel=True,
        )

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from models.quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def main(
    seed: int,
    prompt_file: Path,
    output_file: Path,
    batch_size: int = 4,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    default_compile: bool = False,
    finetune_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_prefix: Optional[str] = None,
    resume_generation: bool = False,
    tensor_parallel_size: Optional[int] = None,
    on_the_fly_8bit_quantization: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    tp_size = 1
    if use_tp:
        tp_size = tensor_parallel_size or torch.distributed.get_world_size()
        initialize_model_parallel(tp_size)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if finetune_checkpoint_path is not None:
        finetune_checkpoint_path, _, _ = get_latest_checkpoint_path(
            finetune_checkpoint_path,
            prefix=finetune_checkpoint_prefix,
        )

        if finetune_checkpoint_path is not None:
            print(f"Loading finetune model from {finetune_checkpoint_path} ...")
            load_inference_checkpoint(finetune_checkpoint_path, model)
        model = model.to(device=device)
        model = model.eval()

    if on_the_fly_8bit_quantization:
        print("Quantizing model ...")
        from models.quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime_on_the_fly()
        model = model.to(device=device)
        model = model.eval()

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AcceleraTokenizer(tokenizer_path)

    torch.manual_seed(seed)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )

    assert not (compile and default_compile), "Cannot compile with both modes"

    if compile or default_compile:
        global decode_one_token

    if compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    if default_compile:
        decode_one_token = torch.compile(
            decode_one_token, mode="default", fullgraph=True
        )

    with open(prompt_file, "r") as f:
        prompts = json.load(f)

    # sort prompts by length to minimize padding

    prompt_idx = list(range(len(prompts)))

    assert "idx" not in prompts[0], "Prompts already have idx field"

    if "prompt" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["prompt"]}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    elif "input" in prompts[0]:
        prompts = [
            {"idx": idx, "prompt": prompt["input"]}
            for idx, prompt in zip(prompt_idx, prompts)
        ]
    else:
        raise ValueError("Prompts must have either prompt or input field")

    print("Tokenizing prompts ...")
    all_prompts = [prompt["prompt"] for prompt in prompts]
    tokenized_full_seq = tokenizer.batch_encode(
        all_prompts, bos=[False] * len(all_prompts), eos=[False] * len(all_prompts)
    )

    for prompt, tokenized in zip(prompts, tokenized_full_seq):
        prompt["prompt_len"] = len(tokenized)

    prompts = sorted(prompts, key=lambda x: x["prompt_len"])

    num_sample_prompts = []
    for prompt in prompts:
        for i in range(num_samples):
            sample_prompt = deepcopy(prompt)
            sample_prompt["sample_idx"] = i
            num_sample_prompts.append(sample_prompt)
    prompts = num_sample_prompts

    skipped_prompt_ids = dict()

    if rank == 0 or not use_tp:
        output_parent = output_file.parent
        if not output_parent.is_dir():
            output_parent.mkdir(exist_ok=True, parents=True)

    if use_tp:
        torch.distributed.barrier()

    if resume_generation and os.path.isfile(output_file):
        with open(output_file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["idx"] not in skipped_prompt_ids:
                    skipped_prompt_ids[sample["idx"]] = 0
                skipped_prompt_ids[sample["idx"]] += 1

    # prompts = [prompt for prompt in prompts if prompt["idx"] not in skipped_prompt_ids]
    new_prompts = []
    for prompt in prompts:
        if prompt["idx"] not in skipped_prompt_ids:
            new_prompts.append(prompt)
        else:
            skipped_prompt_ids[prompt["idx"]] -= 1
            if skipped_prompt_ids[prompt["idx"]] == 0:
                del skipped_prompt_ids[prompt["idx"]]
    prompts = new_prompts

    while len(prompts) % batch_size != 0:
        prompts.insert(0, prompts[0])

    dp_rank = get_data_parallel_rank()
    tp_rank = get_model_parallel_rank()

    dp_size = get_data_parallel_world_size()

    if tp_rank == 0:
        output_writer = open(output_file, "a")

    batch_idx = 0

    gc.collect()
    torch.cuda.empty_cache()

    max_seq_len = prompts[-1]["prompt_len"] + max_new_tokens
    max_seq_len = min(max_seq_len, model.config.block_size)

    if compile:
        remove_all_backward_hooks(model)

    for batched_prompt_idx in range(0, len(prompts), batch_size):
        batch_idx += 1
        if batch_idx % dp_size != dp_rank:
            continue

        batched_prompts = prompts[batched_prompt_idx : batched_prompt_idx + batch_size]

        encoded, left_pad_mask_pos = batch_encode_tokens(
            tokenizer, [_["prompt"] for _ in batched_prompts], bos=True, device=device
        )
        prompt_length = encoded.size(1)

        # torch.cuda.synchronize()
        t0 = time.perf_counter()

        model_max_length = model.config.block_size

        if max_new_tokens + prompt_length >= model_max_length:
            max_new_tokens = model_max_length - prompt_length - 1

        y, num_decoded_tokens = generate(
            model,
            encoded,
            left_pad_mask_pos,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            max_seq_len=max_seq_len,
        )

        full_y_list = y.tolist()
        print(post_process(full_y_list[0], tokenizer))
        print()

        # torch.cuda.synchronize()
        t = time.perf_counter() - t0
        tokens_generated = num_decoded_tokens * y.size(0)
        tokens_sec = tokens_generated / t
        print(f"Prompt length: {prompt_length}")
        print(
            f"Time for inference {batched_prompt_idx + batch_size} / {len(prompts)}"
            f": {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        outputs = []

        for y_list in full_y_list:
            output = post_process(y_list[prompt_length:], tokenizer)
            outputs.append(output)

        if tp_rank == 0:
            fcntl.flock(output_writer, fcntl.LOCK_EX)
            try:
                for prompt, output in zip(batched_prompts, outputs):
                    output_writer.write(
                        json.dumps(
                            {
                                "idx": prompt["idx"],
                                "sample_idx": prompt["sample_idx"],
                                "prompt": prompt["prompt"],
                                "output": output,
                            }
                        )
                        + "\n"
                    )
                output_writer.flush()
            finally:
                fcntl.flock(output_writer, fcntl.LOCK_UN)


def post_process(y_list, tokenizer):
    y_list = y_list[:]
    if tokenizer.eos_id in y_list:
        y_list = y_list[: y_list.index(tokenizer.eos_id)]

    if tokenizer.pad_id in y_list:
        y_list = y_list[::-1]
        y_list = y_list[: y_list.index(tokenizer.pad_id)]
        y_list = y_list[::-1]

    return tokenizer.decode(y_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        required=True,
        help="File containing prompts, one per line.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="File to write generated samples to.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--default_compile",
        action="store_true",
        help="Whether to compile the model with default settings.",
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        type=Path,
        default=None,
        help="Finetune checkpoint path.",
    )

    parser.add_argument(
        "--finetune_checkpoint_prefix",
        type=str,
        default=None,
        help="Finetune checkpoint prefix.",
    )

    parser.add_argument(
        "--resume_generation", action="store_true", help="Whether to resume generation."
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Size of tensor parallelism.",
    )

    parser.add_argument(
        "--on_the_fly_8bit_quantization",
        action="store_true",
        help="Whether to quantize after loading the model.",
    )

    args = parser.parse_args()
    main(
        args.seed,
        args.prompt_file,
        args.output_file,
        args.batch_size,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.default_compile,
        args.finetune_checkpoint_path,
        args.finetune_checkpoint_prefix,
        args.resume_generation,
        args.tensor_parallel_size,
        args.on_the_fly_8bit_quantization,
    )
