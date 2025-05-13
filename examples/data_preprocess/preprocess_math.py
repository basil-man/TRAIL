import re
import os
import datasets
import argparse

from typing import List, Dict
from verl.utils.hdfs_io import copy as hdfs_copy, makedirs

SYSTEM_PROMPT = (
"You are a helpful assistant."
)

def extract_solution(solution_str):
    boxed = re.search(r'\\boxed{([^}]*)}', solution_str)
    if boxed:
        return boxed.group(1).strip()
    fallback = re.findall(r'-?\d+\.?\d*', solution_str)
    assert fallback, "No solution found in: " + solution_str
    return fallback[-1]

def get_process_fn(split: str, model_family: str):
    system_prompt = SYSTEM_PROMPT
    def process_fn(example, idx):
        question_raw = example.pop('problem')
        answer_raw = example.pop('solution')
        solution = extract_solution(answer_raw)

        if model_family == "deepseek":
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
        elif model_family == "qwen":
            instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        else:
            raise NotImplementedError()

        question =instruction + ' ' + question_raw

        return {
            "data_source": "EleutherAI/hendrycks_math",
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
    parser.add_argument('--local_dir', default='./data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--model_family', default='qwen', choices=["qwen", "deepseek"])
    args = parser.parse_args()

    subset_names = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    splits = ['train', 'test']

    dataset = {split: {} for split in splits}
    for subset in subset_names:
        for split in splits:
            print(f"Loading {subset} - {split}")
            dataset[split][subset] = datasets.load_dataset('EleutherAI/hendrycks_math', subset, split=split)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    for split in splits:
        all_data = []
        for subset in subset_names:
            print(f"Processing {split} - {subset}")
            ds = dataset[split][subset]
            processed = ds.map(
                function=get_process_fn(split, model_family=args.model_family),
                with_indices=True
            )
            all_data.extend(processed)

        full_dataset = datasets.Dataset.from_list(all_data)
        full_dataset.to_parquet(os.path.join(local_dir, f'{split}.parquet'))
        print(f"Saved {split} set to {os.path.join(local_dir, f'{split}.parquet')}")

    if args.hdfs_dir is not None:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        hdfs_copy(src=local_dir, dst=args.hdfs_dir)
