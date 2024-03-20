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


def main(
    train_math_path: str,
    test_math_path: str,
    prm_data_pattern: str,
    output_path: str,
    levels: str = "Level 1, Level 2, Level 3",
    splitter: str = "\n\n",
):
    # math_dataset = load_dataset("hendrycks/competition_math")
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

    all_promblems = set()
    train_promblems = set()
    asy_promblems = set()

    for split in ["train", "test"]:
        for ex in math_dataset[split]:
            problem = ex["problem"]
            all_promblems.add(problem)
            if "[asy]" in problem:
                asy_promblems.add(problem)

    levels = levels.split(", ")

    for ex in math_dataset["train"]:
        if f'Level {ex["level"]}' in levels:
            problem = ex["problem"]
            train_promblems.add(problem)

    print(f"Number of MATH-easy train problems: {len(train_promblems)}")
    print(f"Number of [asy] problems: {len(asy_promblems)}")

    additional_promblems = set()

    prm_data_files = glob.glob(prm_data_pattern)

    outputs = []
    skipped = 0

    prm_train_promblems = set()

    for data_file in prm_data_files:
        with open(data_file) as f:
            for line in f:
                data = json.loads(line)

                if data["question"]["problem"] in train_promblems:
                    outputs.append(data)
                    prm_train_promblems.add(data["question"]["problem"])
                elif data["question"]["problem"] not in all_promblems:
                    outputs.append(data)
                    prm_train_promblems.add(data["question"]["problem"])
                else:
                    skipped += 1

                # # What is the smallest positive integer such that the product of its digits is $2016$?
                if data["question"]["problem"] not in all_promblems:
                    additional_promblems.add(data["question"]["problem"])

    print(f"Number of PRM train problems: {len(prm_train_promblems)}")
    print(f"Number of additional problems in PRM800k: {len(additional_promblems)}")

    print(f"Skipped {skipped} problems")
    print(f"Number of PRM annotations: {len(outputs)}")

    answer_only_correct_cnt = 0
    answer_only_wrong_cnt = 0

    extended_output = []
    all_labels = []

    generated_answer_set = set()

    for ex in outputs:
        question = ex["question"]["problem"]

        chosen_prefix = ""

        for step in ex["label"]["steps"]:
            if step["completions"] is None:
                print(step)
                break

            for completion in step["completions"]:
                text = completion["text"]

                rating = completion["rating"]

                if rating is None:
                    continue

                if "# Answer\n\n" in text and "\n# Answer\n\n" not in text:
                    text = text.replace("# Answer\n\n", "\n# Answer\n\n")
                if "\n# Answer\n\n" in text and "\n\n# Answer\n\n" not in text:
                    text = text.replace("\n# Answer\n\n", "\n\n# Answer\n\n")

                if "\n\n# Answer\n\n" in text:
                    if rating >= 0:
                        answer_only_correct_cnt += 1
                    else:
                        answer_only_wrong_cnt += 1
                    text = text.strip()

                else:
                    # recheck this
                    # text = text.split("\n\n# Answer\n\n")[0]
                    if len(text) == 0:
                        break

                if chosen_prefix == "":
                    full_text = text
                else:
                    full_text = chosen_prefix + text

                if full_text not in generated_answer_set:
                    new_splitter = splitter
                    if text.endswith("Mr.") or text.endswith("Mrs."):
                        new_splitter = ""
                    elif "# Answer\n\n" in text:
                        new_splitter = ""
                    extended_output.append(
                        {
                            "input": question,
                            "output_prefix": chosen_prefix,
                            "output": text + new_splitter,
                            # merge 0 and 1 ratings
                            "label": 1 if rating >= 0 else 0,
                        }
                    )
                    generated_answer_set.add(full_text)
                    all_labels.append(1 if rating >= 0 else 0)

            if step["chosen_completion"] is None:
                if step["human_completion"] is not None:
                    next_sentence = step["human_completion"]["text"]
                elif len(step["completions"]) == 1:
                    next_sentence = step["completions"][0]["text"]
                else:
                    break
            else:
                next_sentence = step["completions"][step["chosen_completion"]]["text"]

            if chosen_prefix == "":
                chosen_prefix = next_sentence + splitter
            else:
                chosen_prefix = chosen_prefix + next_sentence + splitter

    print(f"Number of extended PRM annotations: {len(extended_output)}")
    print(f"Number of 0 labels: {all_labels.count(0)}")
    print(f"Number of 1 labels: {all_labels.count(1)}")

    print(f"Number of answer only correct: {answer_only_correct_cnt}")
    print(f"Number of answer only wrong: {answer_only_wrong_cnt}")

    random.shuffle(extended_output)

    # Check max input (input + output_prefix) length and max output length

    # max_input_length = 768
    # max_output_length = 256
    max_total_length = 768 - 32

    filtered_output = []

    for ex in tqdm.tqdm(extended_output):
        input_length = len(tokenizer(ex["input"] + ex["output_prefix"])["input_ids"])
        output_length = len(tokenizer(ex["output"])["input_ids"])

        # if input_length <= max_input_length and output_length <= max_output_length:
        #     filtered_output.append(ex)

        if input_length + output_length <= max_total_length:
            filtered_output.append(ex)

    print(f"Number of filtered PRM annotations: {len(filtered_output)}")

    with open(output_path, "w") as f:
        json.dump(filtered_output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
