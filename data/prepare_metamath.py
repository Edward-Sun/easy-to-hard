import glob
import json
import os
import random
import re
import tqdm
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")

import re


MMIQC_ANSWER_PREFIX = 'Please solve the following problem and put your answer at the end with "# Answer\n\n{answer}".'


def main(
    output_path: str,
    pruned_output_path: str,
    pruned_numbers: int = 8,
    epoch: int = 1,
    levels: str = "Level 1, Level 2, Level 3",
):
    levels = levels.split(", ")

    metamath = load_dataset("meta-math/MetaMathQA")["train"]
    ori_math = load_dataset("hendrycks/competition_math")["train"]

    select_list = []
    question_set = set()
    for item in ori_math:
        if item["level"] in levels:
            question_set.add(item["problem"])

    for item in metamath:
        if "MATH" in item["type"]:
            if item["original_question"] in question_set:
                select_list.append(item)
        else:
            select_list.append(item)

    print("Original: ", len(metamath))
    print("Before pruning: ", len(select_list), f"x {epoch} epoches")

    random.shuffle(select_list)

    for item in select_list:
        split_ans = item["response"].split("The answer is: ")
        item["output"] = (
            split_ans[0].split("####")[0].strip() + "\n\n# Answer\n\n" + split_ans[1]
        )

        item["output"] = item["output"].replace("\\[ ", "\\[")
        item["output"] = item["output"].replace(" \\]", "\\]")

        item["output"] = item["output"].replace("\n", "\n\n")
        item["output"] = item["output"].replace(".\n", ".\n\n")
        item["output"] = item["output"].replace(". ", ".\n\n")
        item["output"] = item["output"].replace(".$", ".$\n\n")
        item["output"] = item["output"].replace("\\]", "\\]\n\n")
        for _ in range(10):
            item["output"] = item["output"].replace("\n\n\n", "\n\n")
            item["output"] = item["output"].replace("\n ", "\n")

        item["output"] = item["output"].replace(",\n\n", ", ")
        item["output"] = item["output"].replace("\n\n$", " $")
        item["output"] = item["output"].replace("\n\n\\]", " \\]")
        item["output"] = item["output"].replace("\\[\n\n", "\\[ ")
        item["output"] = item["output"].replace("$ $", "$$")
        item["output"] = item["output"].strip()

        # for i in "abcdefghijklmnopqrstuvwxyz:":
        #     item["output"] = item["output"].replace(f"{i}\n\n\\[", f"{i} \\[")

        for common_abbreviation in [
            "Mr.",
            "Dr.",
            "Ms.",
            "Mrs.",
            "St.",
            "Prof.",
        ]:
            item["output"] = item["output"].replace(
                common_abbreviation + "\n\n", common_abbreviation + " "
            )

    pruned_metamath = []
    num_dict = {}
    for item in select_list:
        num_dict[item["query"]] = []

    for item in select_list:
        num_dict[item["query"]].append(item)

    for key in num_dict:
        training_examples = []
        for _ in range(epoch):
            training_examples.extend(num_dict[key])
        training_examples = training_examples[: pruned_numbers * epoch]
        pruned_metamath.extend(training_examples)

    random.shuffle(pruned_metamath)

    with open(output_path, "w") as f:
        json.dump(select_list, f, indent=2)

    with open(pruned_output_path, "w") as f:
        json.dump(pruned_metamath, f, indent=2)

    print("After pruning", len(pruned_metamath), f"x 1 epoch")
    print("Saving to {}".format(output_path))
    print("Saving pruned to {}".format(pruned_output_path))


if __name__ == "__main__":
    fire.Fire(main)
