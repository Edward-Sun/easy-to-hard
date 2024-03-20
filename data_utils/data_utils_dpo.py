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

from typing import Dict, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset

from arguments import Arguments
import trainers.common_utils as utils
from models.tokenizer_utils import AcceleraTokenizer
from data_utils.data_utils_sft import preprocess_for_sft, extract_alpaca_dataset


class DPODataset(Dataset):
    def __init__(
        self,
        args: Arguments,
        dataset: HFDataset,
        tokenizer: AcceleraTokenizer,
    ):
        super(DPODataset, self).__init__()
        self.tensors = preprocess_for_dpo(
            args=args,
            dataset=dataset,
            tokenizer=tokenizer,
        )

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: value[i] for key, value in self.tensors.items()}


def preprocess_for_dpo(
    args: Arguments,
    dataset: HFDataset,
    tokenizer: AcceleraTokenizer,
    reorder_wl: bool = True,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    df = dataset.to_pandas()
    output_1, output_2, preference = df["output_1"], df["output_2"], df["preference"]

    assign_w_kwargs = dict(
        output=np.where(preference == 1, output_1, output_2),
    )
    assign_l_kwargs = dict(
        output=np.where(preference == 2, output_1, output_2),
    )
    assign_keys = ["instruction", "input", "output"]

    if "is_eos_1" in df.columns:
        is_eos_1, is_eos_2 = df["is_eos_1"], df["is_eos_2"]
        assign_w_kwargs.update(
            is_eos=np.where(preference == 1, is_eos_1, is_eos_2),
        )
        assign_l_kwargs.update(
            is_eos=np.where(preference == 2, is_eos_1, is_eos_2),
        )
        assign_keys.extend(["is_eos"])

    if "win_rate_1" in df.columns:
        win_rate_1, win_rate_2 = df["win_rate_1"], df["win_rate_2"]
        assign_w_kwargs.update(
            win_rate=np.where(preference == 1, win_rate_1, win_rate_2),
        )
        assign_l_kwargs.update(
            win_rate=np.where(preference == 2, win_rate_1, win_rate_2),
        )
        assign_keys.extend(["win_rate"])

    if reorder_wl:
        df_w = df.assign(**assign_w_kwargs)[assign_keys]
        df_l = df.assign(**assign_l_kwargs)[assign_keys]
    else:
        df_w = df.assign(output=output_1)[assign_w_kwargs]
        df_l = df.assign(output=output_2)[assign_l_kwargs]

    df_w_list = df_w.to_dict("records")
    df_l_list = df_l.to_dict("records")

    assert len(df_w_list) == len(df_l_list)

    if args.dataset_format == "alpaca":
        for i in range(len(df_w_list)):
            df_w_list[i].update(extract_alpaca_dataset(df_w_list[i]))
            df_l_list[i].update(extract_alpaca_dataset(df_l_list[i]))
    elif args.dataset_format is None:
        pass
    else:
        raise ValueError(f"Unknown dataset format: {args.dataset_format}")

    tensors_w = preprocess_for_sft(
        instances=df_w_list,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        total_max_len=args.total_max_len,
        train_on_source=args.train_on_source,
        add_eos_to_target=args.add_eos_to_target,
        add_eos_to_marked_target=args.add_eos_to_marked_target,
        return_win_rate=True,
    )
    tensors_l = preprocess_for_sft(
        instances=df_l_list,
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        total_max_len=args.total_max_len,
        train_on_source=args.train_on_source,
        add_eos_to_target=args.add_eos_to_target,
        add_eos_to_marked_target=args.add_eos_to_marked_target,
        return_win_rate=True,
    )
    return dict(
        input_ids_w=tensors_w["input_ids"],
        labels_w=tensors_w["labels"],
        win_rate_w=tensors_w["win_rate"],
        input_ids_l=tensors_l["input_ids"],
        labels_l=tensors_l["labels"],
        win_rate_l=tensors_l["win_rate"],
    )


def make_dpo_data_module(
    tokenizer: AcceleraTokenizer,
    args: Arguments,
) -> dict:
    preference_dataset = utils.local_dataset(args.dataset)
    train_preference = preference_dataset["train"]

    train_dataset = DPODataset(
        args=args,
        dataset=train_preference,
        tokenizer=tokenizer,
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
