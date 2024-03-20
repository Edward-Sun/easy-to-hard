import glob
import json
import os
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
    metamath_path: str,
    shepherd_path: str,
    math_path: str,
):
    with open(shepherd_path, 'r') as f:
        lines = f.readlines()
    shepherd = [json.loads(line.strip()) for line in lines]

    with open(metamath_path, 'r') as f:
        metamath = json.load(f)
    easy_metamath = []
    hard_metamath = []

    with open(math_path, 'r') as f:
        ori_math = json.load(f)

    level1_3_set = set()
    for item in ori_math:
        if item['level'] in ['Level 1', 'Level 2', 'Level 3']:
            level1_3_set.add(item['question'])
    for item in tqdm(metamath, desc="Processing"):
        if 'MATH' in item['type']:
            if item['original_question'] in level1_3_set:
                easy_metamath.append(item)
            else:
                hard_metamath.append(item)
    #### remove gsm8k ####
    new_shepherd = []
    for item in shepherd:
        if item['task'] == 'MATH':
            new_shepherd.append(item)

    print(len(easy_metamath), len(new_shepherd))
    level1to3_metamath = set()
    level4to5_metamath = set()
    for item in easy_metamath:
        level1to3_metamath.add(item['original_question'][:30])

    for item in hard_metamath:
        level4to5_metamath.add(item['original_question'][:30])

    processed_shepherd = []
    for item in tqdm(new_shepherd):
        question = item['input'].split(' Step 1')[0]
        if question[:30] in level1to3_metamath:
            item['level'] = 'easy'
            # if question[:30] not in level4to5_metamath:
            #     item['level'] = 'easy'
            # else:
            #     item['level'] = 'hard'
        else:
            item['level'] = 'hard'
        matches = [match.start() for match in re.finditer(r'[+-]\nStep', item['label'])]
        processed_shepherd_item = dict()
        processed_shepherd_item['input'] = question
        processed_shepherd_item['output'] = []
        processed_shepherd_item['label'] = []
        processed_shepherd_item['level'] = item['level']
        for idx in matches:
            processed_shepherd_item['output'].append(item['label'][len(question):idx])
            if item['label'][idx] == '+':
                processed_shepherd_item['label'].append(1)
            elif item['label'][idx] == '-':
                processed_shepherd_item['label'].append(0)
            else:
                print('error')

        if item['label'].strip()[-1] == '+':
            processed_shepherd_item['output'].append(item['label'][len(question):])
            processed_shepherd_item['label'].append(1)
        elif item['label'].strip()[-1] == '-':
            processed_shepherd_item['output'].append(item['label'][len(question):])
            processed_shepherd_item['label'].append(0)
        processed_shepherd.append(processed_shepherd_item)

    for item in processed_shepherd:
        new_output = []
        for idx, output in enumerate(item['output']):
            output = output.replace(' Step 1: ', '')
            output = re.sub(r'[+-]\nStep \d+: ', '', output)
            gold_answer = output.split('The answer is: ')
            if len(gold_answer)==2:
                output = gold_answer[0].strip() + '\n\n# Answer\n\n' + gold_answer[1][:-2]
            if idx >=1:
                new_output.append(output[len(past_output):].strip())
            else:
                new_output.append(output.strip())
            past_output = output
        item['output'] = new_output
        # print(item['output'])
        if len(item['output']) != len(item['label']):
            print(len(item['output']), len(item['label']))
    print("all data:", len(processed_shepherd))

    with open(output_1to5_path, 'w') as f:
        json.dump(processed_shepherd, f)
    level1to3_processed_shepherd = []
    for item in processed_shepherd:
        if item['level'] == 'easy':
            level1to3_processed_shepherd.append(item)
    with open(output_1to3_path, 'w') as f:
        json.dump(level1to3_processed_shepherd, f)
    print("level 1-3:", len(level1to3_processed_shepherd))

if __name__ == "__main__":
    fire.Fire(main)
