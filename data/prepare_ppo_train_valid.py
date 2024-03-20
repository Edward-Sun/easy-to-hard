import json
from typing import List

import fire
import os
import tempfile

import random
from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")


def main(
    train_output_path: str,
    valid_output_path: str,
    train_math_path: str,
    test_math_path: str,
    valid_size: int = 500,
    skip_unavailable: bool = False,
    question_format: str = "prm800k",
    levels: str = "Level 1, Level 2, Level 3",
    remove_metamath_problems: bool = False,
    seed: int = 42,
):
    metamath_dataset = load_dataset(
        "meta-math/MetaMathQA",
        cache_dir=os.path.join(
            tempfile.gettempdir(), f"{os.getuid()}_cache", "huggingface", "datasets"
        )
    )

    metamath_questions = set()
    for ex in metamath_dataset["train"]:
        metamath_questions.add(question_format.format(question=ex["query"]))

    original_math_dataset = load_dataset(
        "hendrycks/competition_math",
        cache_dir=os.path.join(
            tempfile.gettempdir(), f"{os.getuid()}_cache", "huggingface", "datasets"
        ),
    )

    if question_format == "prm800k":
        question_format = "# Question\n\n{question}\n\n# Solution\n\n"
    elif question_format == "deepseek":
        question_format = "User: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
    else:
        raise ValueError("Unknown question format")

    original_test = original_math_dataset["test"]
    original_test_set = set()
    for data in original_test:
        original_test_set.add(question_format.format(question=data["problem"]))

    levels = levels.split(", ")
    math_dataset = {}

    math_dataset["train"] = []
    math_dataset["test"] = []

    with open(train_math_path) as f:
        for line in f:
            data = json.loads(line)
            math_dataset["train"].append(data)

    with open(test_math_path) as f:
        for line in f:
            data = json.loads(line)
            math_dataset["test"].append(data)

    print("Train:", len(math_dataset["train"]))
    print("Test:", len(math_dataset["test"]))

    outputs = []

    for data in math_dataset["train"]:
        gt_answer = data["answer"]
        subject = data["subject"]
        level = data["level"]

        if f"Level {level}" in levels:
            answer = gt_answer
        else:
            answer = "Unavailable"

        outputs.append(
            {
                "input": question_format.format(question=data["problem"]),
                "answer": answer,
                "gt_answer": gt_answer,
                "subject": subject,
                "level": level,
            }
        )

    print("Total:", len(outputs))

    # num_list = [43, 90, 105, 128, 134]
    # assert sum(num_list) == valid_size
    count_subject_level = {}
    for output in math_dataset["test"]:
        subject = output["subject"]
        level = output["level"]
        if (subject, level) not in count_subject_level:
            count_subject_level[(subject, level)] = 0
        count_subject_level[(subject, level)] += 1
    print("Count subject level:", count_subject_level)
    print("Num of subject level:", len(count_subject_level))

    train_outputs = []
    valid_outputs = []
    max_max_len = 512 - 16

    random.Random(seed).shuffle(outputs)
    over_lengthed_examples = 0

    metamat_skipped = 0

    for output in outputs:
        input_length = len(tokenizer(output["input"])["input_ids"])
        if remove_metamath_problems and output["input"] in metamath_questions:
            metamat_skipped += 1
            continue

        if input_length > max_max_len:
            over_lengthed_examples += 1
            train_outputs.append(output)
        else:
            subject = output["subject"]
            level = output["level"]
            # if num_list[level - 1] > 0 and output["input"] in original_test_set:
            if count_subject_level[(subject, level)] > 0 and output["input"] in original_test_set:
                valid_outputs.append(output)
                # num_list[level - 1] -= 1
                count_subject_level[(subject, level)] -= 1

            else:
                train_outputs.append(output)

    if skip_unavailable:
        new_train_outputs = []
        for output in train_outputs:
            if output["answer"] != "Unavailable":
                new_train_outputs.append(output)
        train_outputs = new_train_outputs

    with open(train_output_path, "w") as f:
        json.dump(train_outputs, f, indent=2)

    print("Train:", len(train_outputs))
    print("MetaMath skipped:", metamat_skipped)
    print(f"Saving train to {train_output_path}")

    # count levels in train:
    count_levels = {}
    for output in train_outputs:
        level = output["level"]
        if level not in count_levels:
            count_levels[level] = 0
        count_levels[level] += 1
    print("Count levels in train:", count_levels)

    with open(valid_output_path, "w") as f:
        json.dump(valid_outputs, f, indent=2)
    print("Valid:", len(valid_outputs))
    print(f"Saving valid to {valid_output_path}")

    print("Over lengthed examples:", over_lengthed_examples)

if __name__ == "__main__":
    fire.Fire(main)
