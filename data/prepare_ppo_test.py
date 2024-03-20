import json
from typing import List

import fire

from datasets import load_dataset

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_34b")


def main(
    output_path: str,
    train_math_path: str,
    test_math_path: str,
    skip_unavailable: bool = False,
    question_format: str = "prm800k",
    levels: str = "Level 1, Level 2, Level 3",
):
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

    if question_format == "prm800k":
        question_format = "# Question\n\n{question}\n\n# Solution\n\n"
    elif question_format == "deepseek":
        question_format = "User: {question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
    else:
        raise ValueError("Unknown question format")

    outputs = []

    levels = levels.split(", ")

    for data in math_dataset["test"]:
        gt_answer = data["answer"]
        subject = data["subject"]
        level = data["level"]

        if f"Level {level}" in levels:
            answer = gt_answer
        else:
            answer = "Unavailable"

        if skip_unavailable and answer == "Unavailable":
            continue

        outputs.append(
            {
                "input": question_format.format(question=data["problem"]),
                "answer": answer,
                "gt_answer": gt_answer,
                "subject": subject,
                "level": level,
            }
        )

    print("Test:", len(outputs))

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
