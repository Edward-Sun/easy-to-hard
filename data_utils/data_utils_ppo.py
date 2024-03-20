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
from typing import Dict
import logging

import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from arguments import Arguments
import trainers.common_utils as utils
from models.tokenizer_utils import AcceleraTokenizer

logger = logging.getLogger(__name__)


class QueryDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: AcceleraTokenizer,
        query_len: int,
    ):
        super(QueryDataset, self).__init__()

        list_dict_data = dataset.to_pandas().to_dict("records")

        # prompts are strings; queries are tensors.
        queries = [dict_data["input"] for dict_data in list_dict_data]
        answers = [
            f"{dict_data['answer']} ;;; {dict_data['gt_answer']} ;;; {dict_data['level']}"
            for dict_data in list_dict_data
        ]

        logger.warning(f"Debugging: {answers[:10]}")
        queries = [
            tokenizer(query, return_tensors="pt", truncation=False).input_ids.squeeze(
                dim=0
            )
            for query in queries
        ]

        answers = [
            tokenizer(answer, return_tensors="pt", truncation=False).input_ids.squeeze(
                dim=0
            )
            for answer in answers
        ]

        filtered_queries = []
        filtered_answers = []

        for query, answer in zip(queries, answers):
            if len(query) <= query_len:
                filtered_queries.append(query)
                filtered_answers.append(answer)

        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )

        queries = torch.stack(
            [
                utils.left_pad(query, target_size=(query_len,), value=tokenizer.pad_id)
                for query in filtered_queries
            ]
        )

        max_answer_len = max([len(answer) for answer in filtered_answers])
        answers = torch.stack(
            [
                utils.left_pad(
                    answer,
                    target_size=(max_answer_len,),
                    value=tokenizer.pad_id,
                )
                for answer in filtered_answers
            ]
        )

        assert queries.shape[0] == answers.shape[0]

        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_id).long()
        self.answers = answers
        # Auxiliary data.
        self.list_dict_data = list_dict_data

    def __getitem__(self, i):
        return dict(
            queries=self.queries[i],
            query_attn_masks=self.query_attn_masks[i],
            answers=self.answers[i],
        )

    def __len__(self):
        return len(self.queries)


def make_rl_data_module(
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

    def format_dataset(dataset):
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "answer", "gt_answer", "level"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset)

    # Split train/eval, reduce size
    eval_dataset = None
    if args.do_eval:
        if args.eval_dataset is not None:
            eval_dataset = load_data(args.eval_dataset)
            eval_dataset = format_dataset(eval_dataset)
            eval_dataset = eval_dataset["train"]
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

    test_dataset = None
    if args.do_test:
        if args.test_dataset is not None:
            test_dataset = load_data(args.test_dataset)
            test_dataset = format_dataset(test_dataset)
            test_dataset = test_dataset["train"]
        else:
            raise NotImplementedError("Must specify test dataset if `do_test` is True.")

    train_dataset = dataset["train"]
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))

    train_dataset = QueryDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        query_len=args.source_max_len,
    )

    if eval_dataset is not None:
        eval_dataset = QueryDataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            query_len=args.source_max_len,
        )

    if test_dataset is not None:
        test_dataset = QueryDataset(
            dataset=test_dataset,
            tokenizer=tokenizer,
            query_len=args.source_max_len,
        )

    data_collator = utils.DataCollatorForStackableDataset()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
    )
