from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import torch
import re
import argparse
import os
import glob

# we need to check that we have login the HF account
# !huggingface-cli whoami
# !huggingface-cli login


def load_and_merge_models(
    tp_ckpt_name, pretrain_name, tokenizer_name, save_name_hf, push_to_hf_hub_name
):
    assert (
        save_name_hf or push_to_hf_hub_name
    ), "Please provide a save path or push to HF hub name"

    tp_model_list = []

    last_checkpoint_file = os.path.join(tp_ckpt_name, "last_checkpoint")
    with open(last_checkpoint_file, "r") as f:
        last_checkpoint_file = f.readline().strip()

    last_checkpoint_file = last_checkpoint_file.split("/")[-1]
    last_checkpoint_file = os.path.join(tp_ckpt_name, last_checkpoint_file)

    print("Loading checkpoint files:", last_checkpoint_file)
    for file in sorted(glob.glob(last_checkpoint_file)):
        tp_model_list.append(
            torch.load(
                file,
                mmap=True,
            )["model"]
        )

    print("Loading HF model...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrain_name,
        # device_map="cpu",
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
    )
    cpu_state_dict = model.cpu().state_dict()

    replaced_keys = set()

    print("Convert to HF model...")
    num_tp = len(tp_model_list)

    state_dict = {}

    for key in tp_model_list[0].keys():
        if "wo" in key or "w2" in key:
            state_dict[key] = torch.cat(
                [tp_model_list[i][key].cpu() for i in range(num_tp)], dim=1
            )
        elif "wqkv" in key:
            state_dict[key] = torch.stack(
                [tp_model_list[i][key].cpu() for i in range(num_tp)], dim=0
            )
        elif "output" in key:
            state_dict[key] = torch.cat(
                [tp_model_list[i][key].cpu() for i in range(num_tp)], dim=1
            )
        else:
            state_dict[key] = torch.cat(
                [tp_model_list[i][key].cpu() for i in range(num_tp)], dim=0
            )

    pattern = r"layers\.(\d+)\."

    for key in state_dict.keys():
        layer = None
        match = re.search(pattern, key)
        # layer number except for:
        #   lm_head.weight
        if match:
            layer = match.group(1)
        elif "output.weight" in key:
            name = f"lm_head.weight"
            print(cpu_state_dict[name].size(), state_dict[key].size())
            # repeat on dim 0 to match the size
            repeat_size = cpu_state_dict[name].size(0) // state_dict[key].size(0)
            new_state_dict = state_dict[key].repeat(repeat_size, 1)
            cpu_state_dict[name] = 0.0 * cpu_state_dict[name] + new_state_dict
            replaced_keys.add(name)
        else:
            raise ValueError(f"Invalid key: {key}")

        print("Converting layer", key)
        if "wqkv" in key:
            merged_q, merged_k, merged_v = [], [], []
            reconstruct_q, reconstruct_k = [], []

            if state_dict[key].size(2) == 4096:
                n_heads, n_local_heads = 32, 32
            elif state_dict[key].size(2) == 5120:
                n_heads, n_local_heads = 40, 40
            elif state_dict[key].size(2) == 6656:
                n_heads, n_local_heads = 52, 52
            elif state_dict[key].size(2) == 8192:
                n_heads, n_local_heads = 64, 8
            else:
                raise ValueError(f"Invalid size for {key}: {state_dict[key].size()}")

            head_dim = state_dict[key].size(1) // (n_heads + n_local_heads * 2)

            weight_splits = [
                head_dim * n_heads,
                head_dim * n_local_heads,
                head_dim * n_local_heads,
            ]

            for split_idx in range(state_dict[key].size(0)):
                chunk = state_dict[key][split_idx]
                q, k, v = chunk.split(weight_splits, dim=0)
                merged_q.append(q)
                merged_k.append(k)
                merged_v.append(v)
            merged_q = torch.cat(merged_q, dim=0)
            merged_k = torch.cat(merged_k, dim=0)
            merged_v = torch.cat(merged_v, dim=0)

            #### qk need reconstruction ####
            split_qs = torch.split(merged_q, split_size_or_sections=128, dim=0)
            split_ks = torch.split(merged_k, split_size_or_sections=128, dim=0)
            for split in split_qs:
                matrix0 = split[::2, :]
                matrix1 = split[1::2, :]
                reconstruct_q.append(matrix0)
                reconstruct_q.append(matrix1)
            reconstruct_q = torch.cat(reconstruct_q, dim=0)
            for split in split_ks:
                matrix0 = split[::2, :]
                matrix1 = split[1::2, :]
                reconstruct_k.append(matrix0)
                reconstruct_k.append(matrix1)
            reconstruct_k = torch.cat(reconstruct_k, dim=0)
            #### qk need reconstruction ####

            name = f"model.layers.{layer}.self_attn.q_proj.weight"
            cpu_state_dict[name] = reconstruct_q
            replaced_keys.add(name)

            name = f"model.layers.{layer}.self_attn.k_proj.weight"
            cpu_state_dict[name] = reconstruct_k
            replaced_keys.add(name)

            name = f"model.layers.{layer}.self_attn.v_proj.weight"
            cpu_state_dict[name] = merged_v
            replaced_keys.add(name)

        if "wo" in key:
            name = f"model.layers.{layer}.self_attn.o_proj.weight"
            cpu_state_dict[name] = state_dict[key]
            replaced_keys.add(name)
        if "w1" in key:
            name = f"model.layers.{layer}.mlp.gate_proj.weight"
            cpu_state_dict[name] = state_dict[key]
            replaced_keys.add(name)
        if "w3" in key:
            name = f"model.layers.{layer}.mlp.up_proj.weight"
            cpu_state_dict[name] = state_dict[key]
            replaced_keys.add(name)
        if "w2" in key:
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            cpu_state_dict[name] = state_dict[key]
            replaced_keys.add(name)

    unreplaced_keys = set(cpu_state_dict.keys()) - replaced_keys
    print("Unreplaced keys:", unreplaced_keys)

    print("Loading state dict...")

    model.load_state_dict(cpu_state_dict, strict=False)

    print("Saving HF model...")

    if save_name_hf is not None:
        model.save_pretrained(save_name_hf)
        config = AutoConfig.from_pretrained(pretrain_name)
        tokenizer.save_pretrained(save_name_hf)
        config.save_pretrained(save_name_hf)
    else:
        model.push_to_hub(push_to_hf_hub_name, private=True, safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--tp_ckpt_name", type=str, help="Path to the TP checkpoint name", required=True
    )
    parser.add_argument(
        "--tokenizer_name", type=str, help="Path to the tokenizer name", required=True
    )
    parser.add_argument(
        "--pretrain_name", type=str, help="Path to the pretrain name", required=True
    )
    parser.add_argument(
        "--save_name_hf", type=str, default=None, help="Path to save the HF model"
    )
    parser.add_argument(
        "--push_to_hf_hub_name", type=str, default=None, help="Push to HF hub"
    )

    args = parser.parse_args()
    load_and_merge_models(
        args.tp_ckpt_name,
        args.pretrain_name,
        args.tokenizer_name,
        args.save_name_hf,
        args.push_to_hf_hub_name,
    )
