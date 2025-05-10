import re
import os
import copy
import datasets
import argparse

from typing import List, Dict
from verl.utils.hdfs_io import copy as hdfs_copy, makedirs

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. The reasoning process and answer are enclosed within "
    "<think></think> and <answer></answer> tags, respectively, i.e., "
    "<think> reasoning process here</think><answer> answer here</answer>."
)


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(1).replace(',', '')
    return final_solution


def get_process_fn(split: str, model_family: str, max_length: int = None):
    system_prompt = SYSTEM_PROMPT
    if max_length is not None:
        system_prompt += f" The output of the assistant should be within {max_length} tokens."

    def process_fn(example, idx):
        question_raw = example.pop('question')
        answer_raw = example.pop('answer')
        solution = extract_solution(answer_raw)

        if model_family == "deepseek":
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
        elif model_family == "qwen":
            instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        else:
            raise NotImplementedError()

        question = question_raw + ' ' + instruction

        return {
            "data_source": "openai/gsm8k",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer_raw,
                'question': question_raw
            },
            "level": 6
        }

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--model_family', default='qwen', choices=["qwen", "deepseek"])
    parser.add_argument('--max_length', type=int, default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset('openai/gsm8k', 'main')
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(
        function=get_process_fn('train', model_family=args.model_family, max_length=args.max_length),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        function=get_process_fn('test', model_family=args.model_family, max_length=args.max_length),
        with_indices=True
    )

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        hdfs_copy(src=local_dir, dst=args.hdfs_dir)
