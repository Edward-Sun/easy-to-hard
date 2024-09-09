# Copyright 2024 The GPT-Accelera Team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch

from math_utils import grader

from models.tokenizer_utils import AcceleraTokenizer

import trainers.common_utils as common_utils


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def _calculate_outcome_accuracy(
    predicted_answers: List[str],
    gt_answers: List[str],
    answers: List[str],
    levels: List[int],
    outcome_reward: bool,
    easy_outcome_reward: bool,
    device: torch.device,
):
    assert len(predicted_answers) == len(answers)

    assert not (
        outcome_reward and easy_outcome_reward
    ), "Cannot use both outcome_reward and easy_outcome_reward."

    with common_utils.DisableLogger():
        outcome_accuracy = [
            1.0 if grader.grade_answer(predicted_answer, gt_answer) else 0.0
            for predicted_answer, gt_answer in zip(predicted_answers, gt_answers)
        ]

        # TODO (zhiqings): 0.25 is a magic number.
        unavailable_reward = 0.25
        if outcome_reward:
            symbolic_rewards = outcome_accuracy
        elif easy_outcome_reward:
            symbolic_rewards = []
            for predicted_answer, answer in zip(predicted_answers, answers):
                if answer == "Unavailable":
                    score = unavailable_reward
                elif grader.grade_answer(predicted_answer, answer):
                    score = 1.0
                else:
                    score = 0.0
                symbolic_rewards.append(score)
        else:
            symbolic_rewards = [
                unavailable_reward for _ in range(len(predicted_answers))
            ]

    assert len(symbolic_rewards) == len(predicted_answers)

    per_level_counts = {}
    per_level_accuracy = {}

    all_unique_levels = list(range(1, 6))

    for level in all_unique_levels:
        per_level_counts[level] = []
        per_level_accuracy[level] = []

    for level, accuracy in zip(levels, outcome_accuracy):
        for unique_level in all_unique_levels:
            if level == unique_level:
                per_level_counts[unique_level].append(1.0)
                per_level_accuracy[unique_level].append(accuracy)
            else:
                per_level_counts[unique_level].append(0.0)
                per_level_accuracy[unique_level].append(0.0)

    for level in all_unique_levels:
        assert len(per_level_counts[level]) == len(outcome_accuracy)
        assert len(per_level_accuracy[level]) == len(outcome_accuracy)
        per_level_counts[level] = torch.tensor(per_level_counts[level], device=device)
        per_level_accuracy[level] = torch.tensor(
            per_level_accuracy[level], device=device
        )

    original_symbolic_rewards = symbolic_rewards

    symbolic_rewards = torch.tensor(symbolic_rewards, device=device)
    outcome_accuracy = torch.tensor(outcome_accuracy, device=device)

    ret_dict = {
        "symbolic_rewards": symbolic_rewards,
        "outcome_accuracy": outcome_accuracy,
    }

    for level in sorted(list(all_unique_levels)):
        ret_dict[f"level_{level}_counts"] = per_level_counts[level]
        ret_dict[f"level_{level}_accuracy"] = per_level_accuracy[level]

    return ret_dict, original_symbolic_rewards


def post_process_math_rollouts(
    text_responses: List[str],
    answer_gt_levels: List[str],
    tokenizer: AcceleraTokenizer,
    stop_token: Optional[str],
    outcome_reward: bool,
    easy_outcome_reward: bool,
    device: torch.device,
):
    if stop_token is not None:
        parsed_stop_token = stop_token
        parsed_stop_token = parsed_stop_token.replace(r"\n", "\n")
        parsed_stop_token = parsed_stop_token.replace(r"\\", "\\")
    else:
        parsed_stop_token = tokenizer.eos_token

    predicted_answers = []
    for text_response in text_responses:
        predicted_answer = "No answer found."
        if "\n\n" in parsed_stop_token:
            if parsed_stop_token in text_response:
                predicted_answer = text_response.split(parsed_stop_token)[1]
                predicted_answer = predicted_answer.split(tokenizer.eos_token)[0]
        elif "\\boxed{}" == parsed_stop_token:
            boxed_predicted_answer = text_response.split(tokenizer.eos_token)[0]
            boxed_predicted_answer = remove_boxed(
                last_boxed_only_string(boxed_predicted_answer)
            )
            if boxed_predicted_answer is not None:
                predicted_answer = boxed_predicted_answer
        else:
            raise ValueError(f"Unknown stop token: {parsed_stop_token}")
        predicted_answers.append(predicted_answer)

    text_answers_gt_levels = tokenizer.batch_decode(
        answer_gt_levels,
        skip_special_tokens=True,
    )

    answers, gt_answers, levels = [], [], []
    for text_answers_gt_level in text_answers_gt_levels:
        assert len(text_answers_gt_level.split(";;;")) == 3, text_answers_gt_level
        answer, gt_answer, level = text_answers_gt_level.split(";;;")
        answers.append(answer.strip())
        gt_answers.append(gt_answer.strip())
        levels.append(int(level.strip()))

    outcome_metrics, symbolic_rewards = _calculate_outcome_accuracy(
        predicted_answers,
        gt_answers,
        answers,
        levels,
        outcome_reward,
        easy_outcome_reward,
        device,
    )
    return outcome_metrics


def _post_process_newline_scores(
    encoded_full_seq,
    scores,
    prompt,
    output,
    str_splitter="\n\n",
    newline_id=13,
):
    if str_splitter == "\n\n":
        token_splitter = f" {newline_id}, {newline_id},"
    else:
        raise ValueError(f"Unknown str_splitter: {str_splitter}")
    encoded_full_seq_string = " " + ", ".join(str(_) for _ in encoded_full_seq) + ","

    prompt_newlines = prompt.count(str_splitter)
    output_newlines = output.count(str_splitter)
    full_seq_newlines = encoded_full_seq_string.count(token_splitter)

    # we get scores at the position from the end of the prompt
    # to the end of the output

    assert prompt_newlines + output_newlines <= full_seq_newlines, (
        f"Something wrong: {prompt_newlines} + {output_newlines} v.s. {full_seq_newlines}\n\n"
        f"===\n{prompt}\n===\n{output}\n===\n{encoded_full_seq_string}\n==="
    )

    encoded_prompt_string = token_splitter.join(
        encoded_full_seq_string.split(token_splitter, prompt_newlines)[:-1]
    )
    prompt_seq_len = (
        len([_.strip() for _ in encoded_prompt_string.split(",") if _.strip()]) + 2
    )
    encoded_output_string = encoded_full_seq_string.split(
        token_splitter, prompt_newlines
    )[-1]

    splitted_encoded_full_seq_string = [
        _.strip() for _ in encoded_full_seq_string.split(",") if _.strip()
    ]

    newline_scores = []
    beautiful_output = [prompt.strip()]
    pos = prompt_seq_len
    assert (
        splitted_encoded_full_seq_string[pos - 1] == f"{newline_id}"
    ), f"Something wrong: {splitted_encoded_full_seq_string[pos - 3: pos + 2]} v.s. {newline_id}"
    assert (
        splitted_encoded_full_seq_string[pos - 2] == f"{newline_id}"
    ), f"Something wrong: {splitted_encoded_full_seq_string[pos - 4: pos + 1]} v.s. {newline_id}"

    if len(encoded_output_string.split(token_splitter)) != len(
        output.split(str_splitter)
    ):
        newline_scores = []
        beautiful_output = "Parsing error"
        return newline_scores, beautiful_output

    for i, encoded_segment, segment in zip(
        range(output_newlines + 1),
        encoded_output_string.split(token_splitter),
        output.split(str_splitter),
    ):
        pos += len([_.strip() for _ in encoded_segment.split(",") if _.strip()]) + 2
        if i < output_newlines - 1:
            assert splitted_encoded_full_seq_string[pos - 1] == f"{newline_id}", (
                f"Something wrong: {splitted_encoded_full_seq_string[pos - 3: pos + 2]}"
                f" v.s. {newline_id}"
            )
            assert splitted_encoded_full_seq_string[pos - 2] == f"{newline_id}", (
                f"Something wrong: {splitted_encoded_full_seq_string[pos - 4: pos + 1]}"
                f" v.s. {newline_id}"
            )
            if pos < prompt_seq_len + 4:  # 4 is magic number
                newline_scores.append(
                    (
                        pos - 2 - prompt_seq_len,
                        -4.44,
                    )
                )  # noqa
                beautiful_output.append(f"{segment} (-4.44)")
            else:
                newline_scores.append(
                    (
                        pos - 2 - prompt_seq_len,
                        min(
                            scores[pos - 3],
                            scores[pos - 2],
                            scores[pos - 1],
                        ),
                    )
                )  # noqa
                beautiful_output.append(
                    f"{segment} ({scores[pos - 3]},{scores[pos - 2]},{scores[pos - 1]})"
                )
        else:
            beautiful_output.append(segment)

    return newline_scores, str_splitter.join(beautiful_output)


def post_process_math_prm_scores(
    sequences: torch.Tensor,
    rewards: torch.Tensor,
    text_queries: List[str],
    text_responses: List[str],
    tokenizer: AcceleraTokenizer,
):
    encoded_full_seqs = sequences.tolist()
    scores = rewards.tolist()
    assert len(encoded_full_seqs) == len(scores)

    newline_id = tokenizer.encode("\n\n\n", bos=False, eos=False)[-1]

    packed_post_process = [
        _post_process_newline_scores(
            encoded_full_seqs[i],
            scores[i],
            text_queries[i],
            text_responses[i],
            newline_id=newline_id,
        )
        for i in range(len(encoded_full_seqs))
    ]

    new_line_scores = [_[0] for _ in packed_post_process]
    beautiful_outputs = [_[1] for _ in packed_post_process]

    rank0_print("=" * 20)
    rank0_print("Reward:", beautiful_outputs[0])
    rank0_print("=" * 20)
    return new_line_scores


def math_stop_token_penalty(
    text_responses: List[str],
    stop_token: str,
    eos_token: str,
):
    if stop_token is not None:
        parsed_stop_token = stop_token
        parsed_stop_token = parsed_stop_token.replace(r"\n", "\n")
        parsed_stop_token = parsed_stop_token.replace(r"\\", "\\")
    else:
        parsed_stop_token = eos_token

    stop_token_penalty = []
    for text_response in text_responses:
        if parsed_stop_token not in text_response:
            stop_token_penalty.append(1.0)
        else:
            if "\n\n" in parsed_stop_token:
                main_solution, rest_tokens = text_response.split(parsed_stop_token, 1)
                main_solution = main_solution.strip()
                rest_tokens = rest_tokens.split(eos_token)[0]

                if rest_tokens.strip() == "":
                    stop_token_penalty.append(1.0)
                    continue

                if ("\\boxed{" + rest_tokens + "}") in main_solution:
                    # The best cast: The answer should be highlighted with \boxed{}.
                    stop_token_penalty.append(0.0)
                elif (
                    f"${rest_tokens}$" in main_solution.split("\n\n")[-1]
                    or f"${rest_tokens}.$" in main_solution.split("\n\n")[-1]
                ):
                    # The second best cast: The answer should be highlighted with math mode.
                    stop_token_penalty.append(0.5)
                elif rest_tokens in main_solution.split("\n\n")[-1]:
                    # The third best cast: The answer should show up in the last line.
                    stop_token_penalty.append(0.75)
                else:
                    stop_token_penalty.append(1.0)
            elif "\\boxed{}" == parsed_stop_token:
                if "\\boxed" in text_response:
                    stop_token_penalty.append(0.0)
                else:
                    stop_token_penalty.append(1.0)
            else:
                raise ValueError(f"Unknown stop token: {parsed_stop_token}")
    return stop_token_penalty


def _post_terminating_reward(
    reward_outputs: Dict[str, torch.Tensor],
    responses: torch.Tensor,
    penalize_no_stop_token: bool,
    relative_stop_token_penalty: bool,
    stop_token_penalty: List[float],
    penalty_reward_value: float,
) -> Dict[str, torch.Tensor]:
    """Assign bad reward values to sequences which didn't stop properly."""
    if penalize_no_stop_token:
        stop_token_penalty = torch.tensor(stop_token_penalty, device=responses.device)
        rewards = reward_outputs["rewards"]
        if relative_stop_token_penalty:
            rewards = rewards + stop_token_penalty * penalty_reward_value
        else:
            rewards[stop_token_penalty > 0.0] = penalty_reward_value
        reward_outputs["rewards"] = rewards
        return reward_outputs
    else:
        return reward_outputs


def shape_math_process_rewards(
    symbolic_rewards: torch.Tensor,
    queries: torch.Tensor,
    responses: torch.Tensor,
    tokenizer: AcceleraTokenizer,
    device: torch.device,
    new_line_scores: List[List[Tuple[int, float]]],
    stop_token_penalty: List[bool],
    penalty_reward_value: float,
    penalize_no_stop_token: bool,
    relative_stop_token_penalty: bool,
    process_reward_scheme: str,
    process_reward_upper_bound: float,
    apply_process_reward: bool,
    apply_terminal_process_reward: bool,
    process_reward_scale: float,
):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    for idx, per_solution_new_line_scores in enumerate(new_line_scores):
        for pos, score in per_solution_new_line_scores:
            assert pos >= 0, f"Not non-negative: {pos}"
            # # TODO (zhiqings): 13 is the id of newline token in llama
            # assert responses_np[idx, pos] == 13, f"Not 13: {responses_np[idx, pos]}"
            # assert (
            #     responses_np[idx, pos + 1] == 13
            # ), f"Not 13: {responses_np[idx, pos + 1]}"

    shaped_rewards = np.zeros((responses.size(0), responses.size(1)), dtype=float)

    for idx, per_solution_new_line_scores in enumerate(new_line_scores):
        scores = [sigmoid(_[1]) for _ in per_solution_new_line_scores]
        effetive_negative_step = -1
        if process_reward_scheme == "min":
            min_score = process_reward_upper_bound
            normalized_score = []
            for score in scores:
                if score < min_score:
                    normalized_score.append(score - min_score)
                    min_score = score
                    effetive_negative_step = len(normalized_score)
                else:
                    normalized_score.append(0.0)
        elif process_reward_scheme == "prod":
            prod_score = 1.0
            min_score = process_reward_upper_bound
            normalized_score = []
            for score in scores:
                prod_score = prod_score * score
                if prod_score < min_score:
                    normalized_score.append(prod_score - min_score)
                    min_score = prod_score
                    effetive_negative_step = len(normalized_score)
                else:
                    normalized_score.append(0.0)
        else:
            raise ValueError(f"Unknown process_reward_scheme: {process_reward_scheme}")

        # TODO (zhiqings): 0.25 is a magic number.
        positive_score = 0.25
        per_step_positive_score = positive_score / (len(scores) + 1e-6)
        normalized_score = [_ + per_step_positive_score for _ in normalized_score]

        assert len(per_solution_new_line_scores) == len(normalized_score)
        for pos, score in zip(
            [_[0] for _ in per_solution_new_line_scores], normalized_score
        ):
            shaped_rewards[idx, pos] = score / 2.0
            shaped_rewards[idx, pos + 1] = score / 2.0

        if idx == 0:
            rank0_print("=" * 20)
            rank0_print("Normalized Scores:", normalized_score)
            rank0_print("=" * 20)

    terminal_rewards = symbolic_rewards.clone()

    terminal_rewards = _post_terminating_reward(
        {"rewards": terminal_rewards},
        responses,
        penalize_no_stop_token=penalize_no_stop_token,
        relative_stop_token_penalty=relative_stop_token_penalty,
        stop_token_penalty=stop_token_penalty,
        penalty_reward_value=penalty_reward_value,
    )["rewards"]

    if apply_process_reward or apply_terminal_process_reward:
        shaped_rewards = torch.tensor(shaped_rewards, device=device)
        terminal_positions = (responses != tokenizer.pad_id).sum(dim=1) - 1
        shaped_rewards = shaped_rewards * process_reward_scale
        shaped_rewards[
            torch.arange(queries.size(0), device=responses.device),
            terminal_positions,
        ] += terminal_rewards

        if apply_process_reward:
            return shaped_rewards, shaped_rewards.sum(dim=1), responses
        else:
            return None, shaped_rewards.sum(dim=1), responses
    else:
        return None, terminal_rewards, responses


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None
