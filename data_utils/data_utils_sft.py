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

import os
from dataclasses import dataclass
import logging
from typing import Dict, Sequence, Union

import torch

from datasets import load_dataset

from arguments import Arguments
import trainers.common_utils as utils
from models.tokenizer_utils import AcceleraTokenizer

logger = logging.getLogger(__name__)

DROMEDARY_PROMPT_DICT = {
    "prompt_input": (
        "{meta_prompt}\n" "{instruction}\n\n" "{input}\n\n" "### Dromedary"
    ),
    "prompt_no_input": ("{meta_prompt}\n" "{instruction}\n\n" "### Dromedary"),
}

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def preprocess_for_sft(
    instances: Sequence[Dict],
    tokenizer: AcceleraTokenizer,
    source_max_len: int,
    target_max_len: int,
    total_max_len: int,
    train_on_source: bool,
    add_eos_to_target: bool,
    add_eos_to_marked_target: bool,
    return_win_rate: bool = False,
) -> Dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    # Extract elements
    sources = [example["input"] for example in instances]
    targets = [f"\n{example['output']}" for example in instances]

    begin_padding_len = tokenizer(
        ["\n"], return_tensors="pt", add_bos=False, add_eos=False
    ).input_ids.shape[1]

    # Tokenize
    tokenized_sources_with_prompt = tokenizer(
        sources,
        max_length=source_max_len,
        padding="max_length",
        truncation=True,
        add_bos=True,
        add_eos=False,
        padding_side="left",
        truncation_side="left",
    )

    marked_eos = None
    if "is_eos" in instances[0] and add_eos_to_marked_target:
        marked_eos = [example["is_eos"] for example in instances]

    win_rate = None
    if return_win_rate:
        if "win_rate" in instances[0]:
            win_rate = [example["win_rate"] for example in instances]
        else:
            win_rate = [0.5 for _ in instances]

    # logger.warning(f"Tokenizing {len(targets)} pairs...")
    tokenized_targets = tokenizer(
        targets,
        max_length=target_max_len + begin_padding_len,
        padding="max_length",
        truncation=True,
        add_bos=False,
        add_eos=add_eos_to_target,
        marked_eos=marked_eos,
        padding_side="right",
        truncation_side="right",
    )
    # Build the input and labels for causal LM
    input_ids = []
    labels = []
    for source_length, tokenized_source, tokenized_target in zip(
        tokenized_sources_with_prompt["length"],
        tokenized_sources_with_prompt["input_ids"],
        tokenized_targets["input_ids"],
    ):
        tokenized_target = tokenized_target[begin_padding_len:]
        full_seq = tokenized_source + tokenized_target

        # move the beginning padding to the end of the full_seq
        num_begin_padding = len(tokenized_source) - source_length
        full_seq = full_seq[num_begin_padding:] + full_seq[:num_begin_padding]

        if total_max_len is not None:
            full_seq = full_seq[:total_max_len]

        # input_ids.append(torch.tensor(full_seq))
        input_ids.append(full_seq)
        if not train_on_source:
            full_seq_label = (
                [tokenizer.pad_id for _ in range(source_length)]
                + tokenized_target
                + [tokenizer.pad_id for _ in range(num_begin_padding)]
            )
            if total_max_len is not None:
                full_seq_label = full_seq_label[:total_max_len]
            # labels.append(torch.tensor(full_seq_label))
            labels.append(full_seq_label)
        else:
            # labels.append(torch.tensor(copy.deepcopy(full_seq)))
            labels.append(full_seq)
    # Apply padding
    # input_ids = pad_sequence(
    #     input_ids, batch_first=True, padding_value=tokenizer.pad_id
    # )
    # labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_id)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    data_dict = {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(tokenizer.pad_id),
    }
    if labels is not None:
        data_dict["labels"] = labels
    if return_win_rate:
        data_dict["win_rate"] = torch.tensor(win_rate).view(-1, 1)
    return data_dict


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: AcceleraTokenizer
    source_max_len: int
    target_max_len: int
    total_max_len: int
    train_on_source: bool
    add_eos_to_target: bool
    add_eos_to_marked_target: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return preprocess_for_sft(
            instances=instances,
            tokenizer=self.tokenizer,
            source_max_len=self.source_max_len,
            target_max_len=self.target_max_len,
            total_max_len=self.total_max_len,
            train_on_source=self.train_on_source,
            add_eos_to_target=self.add_eos_to_target,
            add_eos_to_marked_target=self.add_eos_to_marked_target,
        )


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def extract_dromedary_dataset(example, meta_prompts):
    assert "example_id" in example
    total_meta_prompt = len(meta_prompts)
    meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]

    if example.get("input", "") != "":
        prompt_format = DROMEDARY_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = DROMEDARY_PROMPT_DICT["prompt_no_input"]

    return {
        "input": prompt_format.format(meta_prompt=meta_prompt, **example),
        "output": "\n" + example["output"],
    }


def extract_prm_dataset(example):
    if example["output_prefix"] == "":
        ret = {
            "input": "Question: " + example["input"],
            "output": "\n\nAnswer: " + example["output"],
        }
    else:
        ret = {
            "input": "Question: "
            + example["input"]
            + "\n\nAnswer: "
            + example["output_prefix"],
            "output": example["output"],
        }

    if "is_eos" in example:
        ret["is_eos"] = example["is_eos"]

    return ret


def extract_prm_v2_dataset(example):
    if example["output_prefix"] == "":
        ret = {
            "input": "# Question\n\n" + example["input"] + "\n\n# Solution",
            "output": "\n\n" + example["output"],
        }
    else:
        ret = {
            "input": "# Question\n\n"
            + example["input"]
            + "\n\n# Solution\n\n"
            + example["output_prefix"],
            "output": example["output"],
        }

    if "is_eos" in example:
        ret["is_eos"] = example["is_eos"]

    return ret


def extract_metamath_dataset(example):
    ret = {
        "input": "# Question\n\n" + example["query"] + "\n\n# Solution",
        "output": "\n\n" + example["output"],
        "is_eos": True,
    }

    return ret


def make_sft_data_module(
    tokenizer: AcceleraTokenizer,
    args: Arguments,
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """

    def load_data(dataset_name):
        if dataset_name == "alpaca":
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == "alpaca-clean":
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == "chip2":
            return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
        elif dataset_name == "self-instruct":
            return load_dataset("yizhongw/self_instruct", name="self_instruct")
        elif dataset_name == "hh-rlhf":
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == "longform":
            return load_dataset("akoksal/LongForm")
        elif dataset_name == "oasst1":
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "vicuna":
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = (
                        args.dataset_format if args.dataset_format else "alpaca"
                    )
                    full_dataset = utils.local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(
                    f"Dataset {dataset_name} not implemented yet."
                )

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == "alpaca"
            or dataset_format == "alpaca-clean"
            or (dataset_format is None and args.dataset in ["alpaca", "alpaca-clean"])
        ):
            dataset = dataset.map(
                extract_alpaca_dataset, remove_columns=["instruction"]
            )
        elif dataset_format == "hh-rlhf" or (
            dataset_format is None and args.dataset == "hh-rlhf"
        ):
            dataset = dataset.map(lambda x: {"input": "", "output": x["chosen"]})
        elif dataset_format == "prm":
            dataset = dataset.map(extract_prm_dataset)
        elif dataset_format == "prm-v2":
            dataset = dataset.map(extract_prm_v2_dataset)
        elif dataset_format == "metamath":
            dataset = dataset.map(extract_metamath_dataset)
        elif dataset_format == "mapped":
            dataset = dataset
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output", "is_eos"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        total_max_len=args.total_max_len,
        train_on_source=args.train_on_source,
        add_eos_to_target=args.add_eos_to_target,
        add_eos_to_marked_target=args.add_eos_to_marked_target,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
    )
