# Copyright 2024 The GPT-Accelera Team
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
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from arguments import Arguments
import trainers.common_utils as utils
from models.tokenizer_utils import AcceleraTokenizer

SPLITTER = " ;;; "


@dataclass
class DataCollatorForPointwiseRewardModeling(object):
    tokenizer: AcceleraTokenizer
    source_max_len: int
    target_max_len: int
    total_max_len: int
    train_on_every_token: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        targets = [f"\n{example['output']}" for example in instances]
        labels = [example["label"] for example in instances]

        begin_padding_len = self.tokenizer(
            ["\n"], return_tensors="pt", add_bos=False, add_eos=False
        ).input_ids.shape[1]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            add_bos=True,
            add_eos=False,
            padding_side="left",
            truncation_side="left",
        )

        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len + begin_padding_len,
            padding="max_length",
            truncation=True,
            add_bos=False,
            add_eos=False,
            padding_side="right",
            truncation_side="right",
        )
        # Build the input and labels for causal LM
        input_ids = []
        weights = []
        for (
            source_length,
            target_length,
            tokenized_source,
            tokenized_target,
        ) in zip(
            tokenized_sources_with_prompt["length"],
            tokenized_targets["length"],
            tokenized_sources_with_prompt["input_ids"],
            tokenized_targets["input_ids"],
        ):
            real_target_length = target_length - begin_padding_len
            tokenized_target = tokenized_target[begin_padding_len:]
            full_seq = tokenized_source + tokenized_target

            # move the beginning padding to the end of the full_seq
            num_begin_padding = len(tokenized_source) - source_length
            full_seq = full_seq[num_begin_padding:] + full_seq[:num_begin_padding]

            if self.total_max_len is not None:
                full_seq = full_seq[: self.total_max_len]

            weight = (
                [0 for _ in range(source_length)]
                + [1 for _ in range(real_target_length)]
                + [0 for _ in range(len(tokenized_target) - real_target_length)]
                + [0 for _ in range(num_begin_padding)]
            )

            if not self.train_on_every_token:
                # we only train on the last three tokens of the target
                if real_target_length > 3:
                    weight = (
                        [0 for _ in range(source_length)]
                        + [0 for _ in range(real_target_length - 3)]
                        + [1 for _ in range(3)]
                        + [0 for _ in range(len(tokenized_target) - real_target_length)]
                        + [0 for _ in range(num_begin_padding)]
                    )

            if self.total_max_len is not None:
                weight = weight[: self.total_max_len]

            input_ids.append(torch.tensor(full_seq))
            weights.append(torch.tensor(weight))

        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        weights = pad_sequence(weights, batch_first=True, padding_value=0)
        weights = weights.float()
        labels = (
            torch.tensor(labels).view(-1, 1).repeat(1, input_ids.shape[1]).contiguous()
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_id),
            "weights": weights,
            "labels": labels,
        }
        return data_dict


@dataclass
class DataCollatorForPointwiseRewardModelingV2(object):
    tokenizer: AcceleraTokenizer
    source_max_len: int
    target_max_len: int
    total_max_len: int
    train_on_every_token: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        batch_size = len(sources)
        targets = []
        target_batch_sizes = []
        for example in instances:
            target_steps = [
                f"\n{output}" for output in example["output"].split(SPLITTER)
            ]
            targets.extend(target_steps)
            target_batch_sizes.append(len(target_steps))
        step_labels = [
            [int(_) for _ in example["label"].split(SPLITTER)] for example in instances
        ]

        begin_padding_len = self.tokenizer(
            ["\n"], return_tensors="pt", add_bos=False, add_eos=False
        ).input_ids.shape[1]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            add_bos=True,
            add_eos=False,
            padding_side="left",
            truncation_side="left",
        )

        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len + begin_padding_len,
            padding="max_length",
            truncation=True,
            add_bos=False,
            add_eos=False,
            padding_side="right",
            truncation_side="right",
        )
        # Build the input and labels for causal LM
        input_ids = []
        weights = []
        labels = []

        batched_tokenized_targets = {}
        batched_tokenized_targets["input_ids"] = []
        batched_tokenized_targets["length"] = []
        start_idx = 0
        for i in range(0, batch_size):
            end_idx = start_idx + target_batch_sizes[i]
            batched_tokenized_targets["input_ids"].append(
                tokenized_targets["input_ids"][start_idx:end_idx]
            )
            batched_tokenized_targets["length"].append(
                tokenized_targets["length"][start_idx:end_idx]
            )
            start_idx = end_idx

        assert len(batched_tokenized_targets["input_ids"]) == len(
            tokenized_sources_with_prompt["input_ids"]
        ), f"{len(batched_tokenized_targets['input_ids'])} != {len(tokenized_sources_with_prompt['input_ids'])}"
        assert len(batched_tokenized_targets["length"]) == len(
            tokenized_sources_with_prompt["length"]
        ), f"{len(batched_tokenized_targets['length'])} != {len(tokenized_sources_with_prompt['length'])}"

        for (
            source_length,
            batched_target_length,
            tokenized_source,
            batched_tokenized_target,
            batched_step_label,
        ) in zip(
            tokenized_sources_with_prompt["length"],
            batched_tokenized_targets["length"],
            tokenized_sources_with_prompt["input_ids"],
            batched_tokenized_targets["input_ids"],
            step_labels,
        ):
            weight = []
            full_seq = []
            label = []

            # add source
            num_begin_padding = len(tokenized_source) - source_length
            full_seq = full_seq + tokenized_source[num_begin_padding:]
            weight = weight + [0 for _ in range(source_length)]
            label = label + [0 for _ in range(source_length)]

            # add target one by one
            for target_length, tokenized_target, step_label in zip(
                batched_target_length, batched_tokenized_target, batched_step_label
            ):
                real_target_length = target_length - begin_padding_len
                tokenized_target = tokenized_target[begin_padding_len:target_length]
                full_seq = full_seq + tokenized_target

                if not self.train_on_every_token and real_target_length > 3:
                    weight = (
                        weight
                        + [0 for _ in range(real_target_length - 3)]
                        + [1 for _ in range(3)]
                    )
                else:
                    weight = weight + [1 for _ in range(real_target_length)]
                label = label + [step_label for _ in range(real_target_length)]

            # add padding
            if self.total_max_len is not None:
                full_seq = full_seq[: self.total_max_len]
                weight = weight[: self.total_max_len]
                label = label[: self.total_max_len]

                if self.total_max_len > len(full_seq):
                    padding_length = self.total_max_len - len(full_seq)
                    weight = weight + [0 for _ in range(padding_length)]
                    full_seq = full_seq + [
                        self.tokenizer.pad_id for _ in range(padding_length)
                    ]
                    label = label + [0 for _ in range(padding_length)]

            assert len(full_seq) == len(weight)
            assert len(full_seq) == len(label)
            input_ids.append(torch.tensor(full_seq))
            weights.append(torch.tensor(weight))
            labels.append(torch.tensor(label))

        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        weights = pad_sequence(weights, batch_first=True, padding_value=0)
        weights = weights.float()
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        labels = labels.long()
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_id),
            "weights": weights,
            "labels": labels,
        }
        return data_dict


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

    ret["label"] = example["label"]

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

    ret["label"] = example["label"]

    return ret


def extract_prm_v3_dataset(example):
    if example["output_prefix"] == "":
        ret = {
            "input": "# Question\n\n" + example["input"] + "\n\n# Solution\n\n",
            "output": example["output"],
        }
    else:
        ret = {
            "input": "# Question\n\n"
            + example["input"]
            + "\n\n# Solution\n\n"
            + example["output_prefix"],
            "output": example["output"],
        }

    ret["label"] = example["label"]

    return ret


def extract_prm_v4_dataset(example):
    output = [_ + "\n\n" for _ in example["output"][:-1]] + [example["output"][-1]]
    assert len(output) == len(example["label"])
    assert all([SPLITTER not in _ for _ in output])

    _input = "# Question\n\n" + example["input"] + "\n\n# Solution\n\n"
    if "output_prefix" in example and example["output_prefix"] is not None:
        _input = _input + example["output_prefix"]

    ret = {
        "input": _input,
        "output": SPLITTER.join(output),
        "label": SPLITTER.join([str(_) for _ in example["label"]]),
    }
    return ret


def make_pointwise_reward_modeling_data_module(
    tokenizer: AcceleraTokenizer,
    args: Arguments,
) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """

    def load_data(dataset_name):
        if os.path.exists(dataset_name):
            try:
                full_dataset = utils.local_dataset(dataset_name)
                return full_dataset
            except:
                raise ValueError(f"Error loading dataset from {dataset_name}")
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    multiple_output_dataset = False
    if args.dataset_format == "prm-v4":
        multiple_output_dataset = True

    def format_dataset(dataset, dataset_format):
        if dataset_format == "prm":
            dataset = dataset.map(extract_prm_dataset)
        elif dataset_format == "prm-v2":
            dataset = dataset.map(extract_prm_v2_dataset)
        elif dataset_format == "prm-v3":
            dataset = dataset.map(extract_prm_v3_dataset)
        elif dataset_format == "prm-v4":
            dataset = dataset.map(extract_prm_v4_dataset)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output", "label"]
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

    if multiple_output_dataset:
        data_collator = DataCollatorForPointwiseRewardModelingV2(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            total_max_len=args.total_max_len,
            train_on_every_token=args.train_on_every_token,
        )
    else:
        data_collator = DataCollatorForPointwiseRewardModeling(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            total_max_len=args.total_max_len,
            train_on_every_token=args.train_on_every_token,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
    )
