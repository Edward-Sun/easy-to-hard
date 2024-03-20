# Copyright 2024 The GPT-Accelera Team
# Copyright 2023 The Transformers Team
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

from typing import Dict, Optional, Union, Any, List, Tuple
import logging
from collections import UserDict
from enum import Enum
from pathlib import Path

import torch
import numpy as np

from sentencepiece import SentencePieceProcessor

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
    print("transformers is not installed. Please install it to use AutoTokenizer.")

logger = logging.getLogger(__name__)


# Define type aliases and NamedTuples
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return isinstance(x, np.ndarray)


def is_torch_device(x):
    return isinstance(x, torch.device)


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    NUMPY = "np"


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class BatchEncoding(UserDict):
    """
    Holds the output of the [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] and
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] methods (tokens, attention_masks, etc).

    This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes
    utility methods to map from word/character space to token space.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
    ):
        super().__init__(data)
        self.convert_to_tensors(
            tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis
        )

    def __getitem__(self, item: Union[int, str]) -> Any:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        if isinstance(item, str):
            return self.data[item]
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def convert_to_tensors(
        self,
        tensor_type: Optional[Union[str, TensorType]] = None,
        prepend_batch_axis: bool = False,
    ):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
            prepend_batch_axis (`int`, *optional*, defaults to `False`):
                Whether or not to add the batch dimension during the conversion.
        """
        if tensor_type is None:
            return self

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.PYTORCH:
            is_tensor = torch.is_tensor

            def as_tensor(value, dtype=None):
                if isinstance(value, list) and isinstance(value[0], np.ndarray):
                    return torch.tensor(np.array(value))
                return torch.tensor(value)

        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(
                    value[0], (list, tuple, np.ndarray)
                ):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor(
                            [np.asarray(val) for val in value], dtype=object
                        )
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if prepend_batch_axis:
                    value = [value]

                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except Exception as e:
                if key == "overflowing_tokens":
                    raise ValueError(
                        "Unable to create tensor returning overflowing tokens of different lengths. "
                        "Please see if a fast version of this tokenizer is available to have this feature available."
                    ) from e
                raise ValueError(
                    "Unable to create tensor, you should probably activate truncation and/or padding with"
                    " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                    f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                    " expected)."
                ) from e

        return self

    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """
        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if (
            isinstance(device, str)
            or is_torch_device(device)
            or isinstance(device, int)
        ):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            raise ValueError(f"Attempting to cast a BatchEncoding to {device}.")
        return self


class AcceleraTokenizer:
    """A customized tokenizer similar to the one in transformers."""

    def __init__(
        self,
        tokenizer: Union[str, Path, SentencePieceProcessor] = None,
        model_vocab_size: int = 32000,  # sometimes model's vocab size is not the same as tokenizer's vocab size
    ):
        if str(tokenizer).endswith(".model"):
            if isinstance(tokenizer, str):
                tokenizer = SentencePieceProcessor(model_file=tokenizer)
            elif isinstance(tokenizer, Path):
                tokenizer = SentencePieceProcessor(model_file=str(tokenizer))

            self.hf_model = None
            self.sp_model = tokenizer
            self.model_vocab_size = model_vocab_size

            # BOS / EOS token IDs
            self.n_words: int = self.sp_model.vocab_size()
            self.bos_id: int = self.sp_model.bos_id()  # 1
            self.eos_id: int = self.sp_model.eos_id()  # 2
            self.pad_id: int = self.sp_model.pad_id()  # -1
            self.unk_id: int = self.sp_model.unk_id()  # 0
            assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
            self.bos_token = "<s>"
            self.eos_token = "</s>"
        else:
            if AutoTokenizer is None:
                raise ImportError(
                    "transformers is not installed. Please install it to use AutoTokenizer."
                )
            self.hf_model = AutoTokenizer.from_pretrained(tokenizer)
            self.sp_model = None

            # check if it's deepseek-math tokenizer
            assert self.hf_model.vocab_size == 100000
            assert self.hf_model.bos_token_id == 100000
            assert self.hf_model.eos_token_id == 100001
            self.hf_model.add_special_tokens(
                {"pad_token": "<pad>", "unk_token": "<unk>"}
            )

            self.model_vocab_size = self.hf_model.vocab_size + 4
            self.n_words: int = self.model_vocab_size
            self.bos_id: int = self.hf_model.bos_token_id
            self.eos_id: int = self.hf_model.eos_token_id
            self.pad_id: int = self.hf_model.pad_token_id
            self.unk_id: int = self.hf_model.unk_token_id
            self.bos_token = self.hf_model.bos_token
            self.eos_token = self.hf_model.eos_token

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        add_bos: bool = True,
        add_eos: bool = False,
        marked_eos: List[bool] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s).
        Return input_ids, attention_mask and length (length is the length before padding).
        """
        is_batched = isinstance(text, (list, tuple))

        if not is_batched:
            tokenized_text = self.encode(text, bos=add_bos, eos=add_eos)

            return BatchEncoding(
                {
                    "input_ids": [tokenized_text],
                    "attention_mask": [[1] * len(tokenized_text)],
                    "length": [len(tokenized_text)],
                },
                tensor_type=return_tensors,
            )

        if marked_eos is None:
            tokenized_text = self.batch_encode(
                text, bos=[add_bos] * len(text), eos=[add_eos] * len(text)
            )
        else:
            assert len(text) == len(marked_eos)
            tokenized_text = self.batch_encode(
                text, bos=[add_bos] * len(text), eos=marked_eos
            )

        if truncation:
            if truncation_side == "left":
                tokenized_text = [t[-max_length:] for t in tokenized_text]
            elif truncation_side == "right":
                tokenized_text = [t[:max_length] for t in tokenized_text]
            else:
                raise ValueError(
                    f"Invalid truncation side: {truncation_side}. Should be 'left' or 'right'"
                )

        if padding == "longest":
            padded_length = max(len(t) for t in tokenized_text)
        elif padding == "max_length":
            assert max_length is not None
            padded_length = max_length
        else:
            padded_length = None

        attention_mask = [[1] * len(t) for t in tokenized_text]
        length = [len(t) for t in tokenized_text]

        if padded_length is not None:
            if padding_side == "right":
                tokenized_text = [
                    t + [self.pad_id] * (padded_length - len(t)) for t in tokenized_text
                ]
                attention_mask = [
                    m + [0] * (padded_length - len(m)) for m in attention_mask
                ]
            elif padding_side == "left":
                tokenized_text = [
                    [self.pad_id] * (padded_length - len(t)) + t for t in tokenized_text
                ]
                attention_mask = [
                    [0] * (padded_length - len(m)) + m for m in attention_mask
                ]
            else:
                raise ValueError(
                    f"Invalid padding side: {padding_side}. Should be 'left' or 'right'"
                )

        return BatchEncoding(
            {
                "input_ids": tokenized_text,
                "attention_mask": attention_mask,
                "length": length,
            },
            tensor_type=return_tensors,
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str

        if self.sp_model is not None:
            t = self.sp_model.encode(s)
            if bos:
                t = [self.bos_id] + t
            if eos:
                t = t + [self.eos_id]
        else:
            t = self.hf_model.encode(s, add_special_tokens=False)
            if bos:
                t = [self.bos_id] + t
            if eos:
                t = t + [self.eos_id]

        t = [x if x < self.model_vocab_size else self.model_vocab_size - 1 for x in t]
        return t

    def decode(self, t: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        if isinstance(t, torch.Tensor):
            t = t.tolist()

        t = [x for x in t if x != self.pad_id]
        if self.sp_model is not None:
            ret = self.sp_convert_tokens_to_string(
                t, skip_special_tokens=skip_special_tokens
            )
        else:
            ret = self.hf_model.decode(t, skip_special_tokens=skip_special_tokens)
        return ret

    def sp_convert_tokens_to_string(self, tokens, skip_special_tokens=False):
        """Converts a sequence of tokens (string) in a single string."""
        if self.unk_id != self.pad_id:
            self.map_special_tokens = {
                self.bos_id: self.bos_token,
                self.eos_id: self.eos_token,
                self.pad_id: self.pad_token,
                self.unk_id: self.unk_token,
            }
        else:
            self.map_special_tokens = {
                self.bos_id: self.bos_token,
                self.eos_id: self.eos_token,
                self.pad_id: self.pad_token,
            }
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in [self.bos_id, self.eos_id, self.pad_id, self.unk_id]:
                if not skip_special_tokens:
                    out_string += (
                        self.sp_model.decode(current_sub_tokens)
                        + self.map_special_tokens[token]
                    )
                    current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        if len(current_sub_tokens) > 0:
            out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def batch_encode(
        self,
        batch_s: List[str],
        bos: List[bool],
        eos: List[bool],
    ):
        """
        Encodes a list of string into a list of token IDs.

        Args:
            batch_s (List[str]): The input strings to be encoded.
            bos (List[bool]): Whether to prepend the beginning-of-sequence token.
            eos (List[bool]): Whether to append the end-of-sequence token.

        Returns:
            List[List[int]]: A list of token IDs.
        """
        assert type(batch_s[0]) is str
        assert len(batch_s) == len(bos)
        assert len(batch_s) == len(eos)

        if self.sp_model is not None:
            batch_t = self.sp_model.encode(batch_s)
        else:
            batch_t = self.hf_model(batch_s, add_special_tokens=False)["input_ids"]

        for i in range(len(batch_t)):
            if bos[i]:
                batch_t[i] = [self.bos_id] + batch_t[i]
        for i in range(len(batch_t)):
            if eos[i]:
                batch_t[i] = batch_t[i] + [self.eos_id]
        for i in range(len(batch_t)):
            batch_t[i] = [
                x if x < self.model_vocab_size else self.model_vocab_size - 1
                for x in batch_t[i]
            ]
        return batch_t

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()

        if self.sp_model is not None:
            return [
                self.decode(
                    seq,
                    skip_special_tokens=skip_special_tokens,
                    **kwargs,
                )
                for seq in sequences
            ]
        else:
            return self.hf_model.batch_decode(
                sequences,
                skip_special_tokens=skip_special_tokens,
                **kwargs,
            )


def batch_encode_tokens(
    tokenizer: AcceleraTokenizer,
    strings: List[str],
    bos=True,
    eos=False,
    device="cuda",
    padding_side="left",
):
    assert padding_side in ["left", "right"]
    batched_tokens = []
    max_len = 0
    for string in strings:
        tokens = tokenizer.encode(string, bos=bos, eos=eos)
        batched_tokens.append(tokens)
        max_len = max(max_len, len(tokens))

    if tokenizer.pad_id >= 0:
        pad_id = tokenizer.pad_id
    else:
        pad_id = tokenizer.unk_id

    if padding_side == "left":
        left_pad_mask_pos = torch.zeros(
            (len(batched_tokens),), dtype=torch.int, device=device
        )
    is_padded = False

    for i in range(len(batched_tokens)):
        if len(batched_tokens[i]) < max_len:
            pad_len = max_len - len(batched_tokens[i])

            if padding_side == "left":
                batched_tokens[i] = [pad_id] * pad_len + batched_tokens[i]
                left_pad_mask_pos[i] = pad_len
            else:
                batched_tokens[i] = batched_tokens[i] + [pad_id] * pad_len

            is_padded = True

    if padding_side == "left":
        if not is_padded:
            left_pad_mask_pos = None

        return (
            torch.tensor(batched_tokens, dtype=torch.int, device=device),
            left_pad_mask_pos,
        )
    else:
        return torch.tensor(batched_tokens, dtype=torch.int, device=device)
