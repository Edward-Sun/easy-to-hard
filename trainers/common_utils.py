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

import argparse
from dataclasses import dataclass
import os
import tempfile
import random
from typing import Callable, Dict, Optional, Sequence, Union, Mapping, Any, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from datasets import load_dataset

Numeric = Union[int, float]


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def flatten_dict(nested, sep=".", postprocess_fn=lambda *args: args):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):  # collections.Mapping fails in py3.10.
                rec(v, prefix + k + sep, into)
            else:
                v = postprocess_fn(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def merge_dict(dicts: Sequence[dict], merge_fn: Callable = lambda *args: args) -> dict:
    """Merge a sequence of dicts (with the same set of keys) into a single dict."""
    if len(dicts) == 0:
        return dict()
    return {key: merge_fn([dict_[key] for dict_ in dicts]) for key in dicts[0].keys()}


def prepare_inputs(
    data: Union[torch.Tensor, Any], device: Union[str, int, torch.device]
) -> Union[torch.Tensor, Any]:
    if isinstance(data, Mapping):
        return type(data)(
            {k: prepare_inputs(v, device) for k, v in data.items()}
        )  # noqa
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_inputs(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)  # This can break with deepspeed.
    return data


def pad_inputs_on_batch(
    data: Sequence[torch.Tensor], per_device_batch_size: int
) -> Sequence[torch.Tensor]:
    batch_size = None
    output_tensors = []
    for tensor in data:
        if batch_size is None:
            batch_size = tensor.size(0)
        assert tensor.size(0) == batch_size

        if batch_size % per_device_batch_size != 0:
            filled_size = per_device_batch_size - (batch_size % per_device_batch_size)
            tensor = torch.cat(
                [
                    tensor,
                    tensor[0:1].expand(filled_size, *tensor.size()[1:]),
                ],
                dim=0,
            )
        output_tensors.append(tensor)
    return output_tensors


def pad(
    inputs: torch.Tensor,
    target_size: Union[torch.Size, Sequence[int]],
    value=0.0,
    left=True,
):
    current_size = inputs.size()
    diffs = tuple(ti - ci for ti, ci in zip_(target_size, current_size))
    pad_params = []
    for diff in diffs:
        pad_params = ([diff, 0] if left else [0, diff]) + pad_params
    res = F.pad(inputs, pad=pad_params, value=value)
    return res


def left_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=True)


def right_pad(
    inputs: torch.Tensor, target_size: Union[torch.Size, Sequence[int]], value=0.0
):
    return pad(inputs=inputs, target_size=target_size, value=value, left=False)


def manual_seed(args_or_seed: Union[int, argparse.Namespace], fix_cudnn=False):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


class InfiniteLoader(object):
    """Wraps an existing DataLoader so that it outputs stuff indefinitely; useful for semi-supervised learning and DDP."""

    def __init__(self, loader: DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.data_iterator = iter(loader)
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iterator)
        except StopIteration:
            # Increment the epoch count
            self.epoch += 1

            # If using Distributed Data Parallel, set the epoch for the sampler
            if dist.is_initialized():
                self.loader.sampler.set_epoch(self.epoch)

            # Create a new iterator for the next epoch
            self.data_iterator = iter(self.loader)
            data = next(self.data_iterator)

        return data


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class DataCollatorForStackableDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([instance[key] for instance in instances])
            for key in instances[0].keys()
        }


def local_dataset(dataset_name):
    if dataset_name.endswith(".json"):
        full_dataset = load_dataset(
            "json",
            data_files=dataset_name,
            cache_dir=os.path.join(
                tempfile.gettempdir(), f"{os.getuid()}_cache", "huggingface", "datasets"
            ),
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    return full_dataset


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset
