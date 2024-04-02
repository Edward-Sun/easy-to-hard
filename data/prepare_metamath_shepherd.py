import json
import random
import re
from tqdm import tqdm
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")

import re


def main(
    output_1to5_path: str,
    output_1to3_path: str,
):

    shepherd = load_dataset("peiyi9979/Math-Shepherd")["train"]
    metamath = load_dataset("meta-math/MetaMathQA")["train"]
    ori_math = load_dataset("hendrycks/competition_math")["train"]

    easy_metamath = []
    hard_metamath = []

    level1_3_set = set()
    for item in ori_math:
        if item["level"] in ["Level 1", "Level 2", "Level 3"]:
            # if item["level"] in ["Level 4", "Level 5"]:
            level1_3_set.add(item["problem"])
    for item in tqdm(metamath, desc="Processing"):
        if "MATH" in item["type"]:

            if item["original_question"] in level1_3_set:
                easy_metamath.append(item)
            else:
                hard_metamath.append(item)
    #### remove gsm8k ####
    new_shepherd = []
    for item in shepherd:
        if item["task"] == "MATH":
            new_shepherd.append(item)

    print(len(easy_metamath), len(new_shepherd))
    level1to3_metamath = set()
    level4to5_metamath = set()

    match_tokens = range(5, 50)
    for item in hard_metamath:
        formatted_question = item["original_question"].replace(" ", "")
        level4to5_metamath.add(formatted_question)
        for match_token in match_tokens:
            level4to5_metamath.add(formatted_question[:match_token])

    for item in easy_metamath:
        formatted_question = item["original_question"].replace(" ", "")
        level1to3_metamath.add(formatted_question)
        for match_token in match_tokens:
            if formatted_question[:match_token] not in level4to5_metamath:
                level1to3_metamath.add(formatted_question[:match_token])

    processed_shepherd = []
    for item in tqdm(new_shepherd):
        splitter = "\u043a\u0438"

        if " Step 1: " not in item["input"]:
            # print("Wrong format")
            continue

        question, solution = item["input"].split(" Step 1: ")
        solution = "Step 1: " + solution

        _, labeled_solution = item["label"].split(" Step 1: ")
        labeled_solution = "Step 1: " + labeled_solution

        formatted_question = question.replace(" ", "")
        if formatted_question in level1to3_metamath:
            item["level"] = "easy"
        else:
            match_flag = False
            for match_token in match_tokens:
                if formatted_question[:match_token] in level1to3_metamath:
                    match_flag = True
                    break
            if match_flag:
                item["level"] = "easy"
            else:
                item["level"] = "hard"

        processed_shepherd_item = dict()
        processed_shepherd_item["input"] = question
        processed_shepherd_item["output"] = []
        processed_shepherd_item["label"] = []
        processed_shepherd_item["level"] = item["level"]
        for step in solution.split(splitter):
            if len(step) > 0:
                clean_step = re.sub(r"Step \d+: ", "\n\n", step).strip()
                processed_shepherd_item["output"].append(clean_step)
                labeled_solution = labeled_solution[len(step) :]

                if len(labeled_solution) == 0:
                    del processed_shepherd_item["output"][-1]
                    break

                if labeled_solution[0] == "+":
                    processed_shepherd_item["label"].append(1)
                elif labeled_solution[0] == "-":
                    processed_shepherd_item["label"].append(0)
                else:
                    raise ValueError("label error")
                labeled_solution = labeled_solution[1:]
        processed_shepherd.append(processed_shepherd_item)

    for item in processed_shepherd:
        new_output = []
        for output in item["output"]:
            gold_answer = output.split("The answer is: ")
            if len(gold_answer) == 2:
                output = (
                    gold_answer[0].strip() + "\n\n# Answer\n\n" + gold_answer[1].strip()
                )
            new_output.append(output)
        item["output"] = new_output

        if len(item["output"]) != len(item["label"]):
            print(len(item["output"]), len(item["label"]))
    print("all data:", len(processed_shepherd))

    random.shuffle(processed_shepherd)

    with open(output_1to5_path, "w") as f:
        json.dump(processed_shepherd, f)

    level1to3_processed_shepherd = []
    for item in processed_shepherd:
        if item["level"] == "easy":
            level1to3_processed_shepherd.append(item)
    with open(output_1to3_path, "w") as f:
        json.dump(level1to3_processed_shepherd, f, indent=2)
    print("level 1-3:", len(level1to3_processed_shepherd))
    print("Saved to", output_1to3_path)


if __name__ == "__main__":
    fire.Fire(main)
