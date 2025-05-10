import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from data_processing.process_utils import *  # noqa: F401, F403
from data_processing.answer_extraction import *  # noqa: F401, F403
from eval.eval_script import *  # noqa: F401, F403

# ------------------------- Utility -------------------------

def set_random_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path: str):
    if path.endswith("json"):
        return json.load(open(path, "r"))
    if path.endswith("jsonl"):
        with open(path, "r") as file:
            return [json.loads(line) for line in file]
    raise NotImplementedError(f"Unsupported file type: {path}")


# ------------------------- Inference -------------------------

def infer(args, test_data, answer_extraction_fn):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    prompts = []
    for example in test_data:
        prompt = ""
        for mess in example["messages"]:
            if mess["role"] == "user":
                if args.model_type == "llama3":
                    prompt += (
                        f"{tokenizer.bos_token}"  # BOS
                        + "<|start_header_id|>user<|end_header_id|>\n\n"
                        + "Please reason step by step, and put your final answer within \\boxed{}.\n"
                        + f"{mess['content']}\n{tokenizer.eos_token}<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                elif args.model_type == "qwen":
                    prompt += (
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
                        + f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n"
                    )
                elif args.model_type == "deepseek":
                    prompt += (
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        "<|im_start|>user\nLet's think step by step and output the final answer within \\boxed{}.\n"
                        + f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n"
                    )
                else:
                    raise NotImplementedError(f"Unknown model_type: {args.model_type}")
            elif mess["role"] == "assistant":
                prompt += mess["content"].rstrip()
            prompt = prompt.lstrip()
        example["prompt"] = prompt
        prompts.append(prompt)

    print("Loading model with vLLM…")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path, dtype="float16")

    torch.cuda.synchronize()
    start_time = time()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    total_time = time() - start_time

    model_outputs = [output.outputs[0].text for output in outputs]

    cot_lengths = []
    for model_completion in model_outputs:
        cot = model_completion.split("\n\nThe final answer is:")[0]
        cot_length = tokenizer(cot, return_tensors="pt")["input_ids"].shape[1]
        cot_lengths.append(cot_length)

    predictions = [
        eval(answer_extraction_fn)(item["messages"][-2]["content"], output, task="cot")
        for item, output in tqdm(
            zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs)
        )
    ]
    assert len(model_outputs) > 0, "Model generated empty output list!"

    results = []
    for example, output, pred, cot_length in zip(test_data, model_outputs, predictions, cot_lengths):
        item = deepcopy(example)
        item.update({
            "model_output": output,
            "prediction": pred,
            "cot_length": cot_length,
        })
        results.append(item)
    return results, total_time


# ------------------------- Main -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- Paths & model conf ---
    parser.add_argument("--output-dir", type=str, default="outputs/Qwen2.5-7B-Instruct/gsm8k/")
    parser.add_argument("--model-path", type=str, default="/your_model_path/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-path", type=str, default="/your_model_path/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5b", "1.5b", "3b", "7b", "13b", "33b", "34b", "70b"],
        default="7b",
    )
    parser.add_argument(
        "--model-type", type=str, choices=["llama3", "qwen", "deepseek"], default="qwen"
    )

    # --- Benchmarks ---
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "gsm8k",
            "math",
            "aime2024",          # Maxwell-Jia/AIME_2024
            "amc",    # AI-MO/aimo-validation-amc
        ],
        default="gsm8k",
    )
    parser.add_argument("--data-type", type=str, choices=["train", "test"], default="test")

    # --- Generation settings ---
    parser.add_argument("--max_num_examples", type=int, default=10**14)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)  # Not used by vLLM
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()

    print(f"Evaluating {args.model_path}")
    print(
        f"Max new tokens: {args.max_new_tokens}, Min new tokens: {args.min_new_tokens}, "
        f"temperature: {args.temperature}, seed: {args.seed}\n"
    )

    # Prepare output directory (bench-specific nesting)
    args.output_dir = os.path.join(
        args.output_dir, f"{args.model_size}/", f"Original/{args.data_type}/"
    )

    # Load test configuration (path, processing, evaluation functions)
    test_conf = read_data(f"configs/{args.benchmark}_{args.data_type}.json")

    for src, info in test_conf.items():
        # ---------------- Prepare dataset files ----------------
        fname = os.path.join(args.output_dir, "test_data", "test.jsonl")
        input_dir = os.path.dirname(fname)
        os.makedirs(input_dir, exist_ok=True)
        metric_path = os.path.join(args.output_dir, "samples", "metrics.json")
        if os.path.exists(metric_path) and read_data(metric_path).get("n_samples", 0) > 0:
            continue  # Skip if already done

        # Preprocess & dump dataset
        with open(fname, "w") as file:
            data = read_data(info["test_path"])
            for i, sample in enumerate(tqdm(data, desc=f"processing {src}")):
                fn = eval(info["process_fn"])
                sample["id"] = sample.get("id", f"{src}-{i}")
                for j, item in enumerate(fn(sample)):
                    item["dataset"] = src
                    item["id"] = f"{src}-test-{i}-{j}"
                    assert "answer" in item, "Processed sample missing 'answer'!"
                    print(json.dumps(item), file=file, flush=True)

        output_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(output_dir, exist_ok=True)

        # ---------------- Inference ----------------
        set_random_seed(args.seed)

        print("Loading data…")
        test_data = []
        with open(os.path.join(input_dir, "test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                messages = example["messages"]
                assert messages[-1]["role"] == "assistant"
                example["reference"] = example.get("reference", "") or [
                    mess["content"] for mess in messages if mess["role"] == "assistant"
                ]
                for mess in messages:
                    if mess["role"] == "assistant":
                        mess["content"] = ""  # hide reference answer during inference
                example["messages"] = messages
                test_data.append(example)

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        results, total_time = infer(args, test_data, info["answer_extraction_fn"])

        print("Finished inference…")

        # ---------------- Evaluation ----------------
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        eval_fn = eval(info.get("eval_fn", "eval_math"))  # default to eval_math if missing
        invalid_outputs, labels = [], []
        for item in results:
            if len(item["prediction"]) == 0:
                invalid_outputs.append(
                    {
                        "prompt": item["prompt"],
                        "output": item["model_output"],
                        "answer": item["prediction"],
                    }
                )
                res = False
            else:
                res = eval_fn(item)
            labels.append(res)

        for item, label in zip(results, labels):
            item["accuracy"] = label

        print("Calculating accuracy…")
        acc = sum(item["accuracy"] for item in results) / len(results)
        print("output acc = {:.5f}".format(acc * 100))

        avg_cot_length = sum(item["cot_length"] for item in results) / len(results)
        print("output avg_cot_length = {:.5f}".format(avg_cot_length))
        print("number of invalid outputs: {}".format(len(invalid_outputs)))

        # ---------------- Save predictions ----------------
        pred_fname = "predictions.jsonl"
        with open(os.path.join(output_dir, pred_fname), "a+", encoding="utf-8") as fout:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        # ---------------- Save metrics ----------------
        metric_fname = "metrics.json"
        with open(os.path.join(output_dir, metric_fname), "w") as fout:
            json.dump(
                {
                    "n_samples": len(results),
                    "accuracy": acc,
                    "avg_cot_length": avg_cot_length,
                    "sample_latency": total_time / len(test_data),
                },
                fout,
                indent=4,
            )
