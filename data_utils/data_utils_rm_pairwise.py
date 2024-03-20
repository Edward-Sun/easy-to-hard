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

import logging
from typing import Optional, Dict, Sequence

import torch
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset

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


def format_prompt(
    example: Dict[str, str],
    prompt_dict: Dict[str, str],
) -> str:
    if prompt_dict is not None:
        assert (
            "instruction" in example
        ), "Internal error: example missing required keys."

        if example.get("input", "") != "":
            prompt_format = prompt_dict["prompt_input"]
        else:
            prompt_format = prompt_dict["prompt_no_input"]
    else:
        prompt_format = "{input}"

    format_prompt = prompt_format.format(**example)
    return format_prompt


def format_output(
    example: dict,
    output_key="output",
) -> str:
    return example[output_key]


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: AcceleraTokenizer,
    max_length: int,
    end_sequence_with_eos: bool,
    use_data_frame: bool = False,
) -> dict:
    """Tokenize a list of strings."""
    if use_data_frame:
        raise NotImplementedError
    strings_ds = strings

    tokenized_strings = tokenizer(
        strings_ds,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_bos=True,
        add_eos=True if end_sequence_with_eos else False,
        padding_side="right",
        truncation_side="right",
    )

    input_ids = torch.stack(
        [torch.tensor(tokenized) for tokenized in tokenized_strings["input_ids"]],
        dim=0,
    )

    return input_ids


def preprocess_for_reward_modeling(
    data: HFDataset,
    tokenizer: AcceleraTokenizer,
    end_sequence_with_eos: bool = False,
    max_length: Optional[int] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    prompt_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    list_dict_data = data.to_pandas().to_dict("records")

    def _get_numeric_preference(example: dict):
        # 1 vs 2 is stored in table, but for modeling we use 0 vs 1; remap here.
        return {1: 0, 2: 1}[example["preference"]]

    choice = torch.tensor(
        [[_get_numeric_preference(dict_data)] for dict_data in list_dict_data]
    )

    def _get_text(example: dict, output_key: str):
        full_prompt = format_prompt(example, prompt_dict) + format_output(
            example, output_key
        )
        return full_prompt

    text_list_0, text_list_1 = tuple(
        [_get_text(dict_data, key) for dict_data in list_dict_data]
        for key in ("output_1", "output_2")
    )

    if max_length is None:
        max_length = query_len + response_len

    logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")
    tokenized_0, tokenized_1 = tuple(
        _tokenize_fn(text_list, tokenizer, max_length, end_sequence_with_eos)
        for text_list in (text_list_0, text_list_1)
    )
    # "size" (bsz, 2, seq_len)
    input_ids = torch.stack(
        [tokenized_0, tokenized_1],
        dim=1,
    )

    packaged_data = dict(
        input_ids=input_ids,
        choice=choice,
        metadata=dict(mean_choice=choice.float().mean().item()),
    )

    return packaged_data


class PairwiseRewardModelingDataset(Dataset):
    def __init__(
        self,
        data: HFDataset,
        tokenizer: AcceleraTokenizer,
        end_sequence_with_eos: bool = False,
        max_length: Optional[int] = None,
        query_len: Optional[int] = None,
        response_len: Optional[int] = None,
        prompt_dict: Optional[Dict[str, str]] = None,
    ):
        super(PairwiseRewardModelingDataset, self).__init__()
        data_dict = preprocess_for_reward_modeling(
            data=data,
            tokenizer=tokenizer,
            end_sequence_with_eos=end_sequence_with_eos,
            max_length=max_length,
            query_len=query_len,
            response_len=response_len,
            prompt_dict=prompt_dict,
        )
        self.input_ids = data_dict["input_ids"]
        self.choice = data_dict["choice"]
        self.metadata = data_dict["metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            choice=self.choice[i],
        )


def make_pairwise_reward_modeling_data_module(
    tokenizer: AcceleraTokenizer,
    args: Arguments,
):
    preference_dataset = utils.local_dataset(args.dataset)
    train_preference = preference_dataset["train"]

    if args.dataset_format == "alpaca":
        prompt_dict = ALPACA_PROMPT_DICT
    elif args.dataset_format is None:
        prompt_dict = None
    else:
        raise ValueError(
            f"Unsupported dataset_format: {args.dataset_format}."
            "Only alpaca and None are supported."
        )

    train_dataset = PairwiseRewardModelingDataset(
        data=train_preference,
        tokenizer=tokenizer,
        end_sequence_with_eos=args.add_eos_to_target,
        max_length=args.total_max_len,
        query_len=args.source_max_len,
        response_len=args.target_max_len,
        prompt_dict=prompt_dict,
    )

    eval_dataset = None
    if args.eval_size > 0:
        train_dataset, eval_dataset = utils.split_train_into_train_and_eval(
            train_dataset=train_dataset,
            eval_size=args.eval_size,
            seed=args.seed,
        )

    data_collator = utils.DataCollatorForStackableDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
