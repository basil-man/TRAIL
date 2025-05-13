import os
import json
import random
import argparse
from copy import deepcopy
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ------------------------- Project imports -------------------------
from data_processing.process_utils import *  # noqa: F401, F403
from data_processing.answer_extraction import *  # noqa: F401, F403
from eval.eval_script import *  # noqa: F401, F403

"""===============================================================
Core evaluation script (updated: CoT length from final output).
==============================================================="""

# ================================================================
# Utility helpers (MUST be top‑level for multiprocessing pickling)
# ================================================================

def set_random_seed(seed: int):
    """Seed python, NumPy and torch RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path: str):
    if path.endswith("json"):
        return json.load(open(path, "r"))
    if path.endswith("jsonl"):
        with open(path, "r") as file:
            return [json.loads(line) for line in file]
    raise NotImplementedError(f"Unsupported file type: {path}")

# -----------------------------------------------------------------------------
# CoT length calculation (now uses final output text)
# -----------------------------------------------------------------------------
def process_completion(model_completion: str, tokenizer):
    """Compute total token length of the final output text."""
    # Count all tokens in the model's final output (reasoning + answer)
    return len(tokenizer(model_completion, add_special_tokens=False)["input_ids"])

# ---------------------- Answer extraction worker --------------------------

def extract_answer_worker(args, answer_extraction_fn):
    item, output = args
    return eval(answer_extraction_fn)(
        item["messages"][-2]["content"], output, task="cot"
    )

# ================================================================
# Inference with budget forcing (returns combined reasoning+answer)
# ================================================================

def inference_with_budget_forcing(
    llm,
    tokenizer,
    prompts,
    budget_tokens,
    num_ignores,
    max_final_tokens,
    temperature=0.0,
):
    """Batch inference with budget forcing."""
    thinking_prompts = []
    is_chat_model = []
    for prompt in prompts:
        chat_style = "<|im_start|>" in prompt
        is_chat_model.append(chat_style)
        thinking_prompts.append(prompt + ("<|im_start|>think\n" if chat_style else ""))

    stop_qwen = tokenizer("<|im_end|>")["input_ids"] if any(is_chat_model) else None
    stop_llama = [tokenizer.eos_token_id] if tokenizer.eos_token_id else None

    base_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=budget_tokens,
    )
    batch_reqs = []
    for chat, tp in zip(is_chat_model, thinking_prompts):
        p = deepcopy(base_params)
        p.stop_token_ids = stop_qwen if chat else stop_llama
        batch_reqs.append((tp, p))
    prompts_list, params_list = zip(*batch_reqs) if batch_reqs else ([], [])

    thinking_outputs = llm.generate(prompts_list, params_list) if prompts_list else []
    thinking_texts = [o.outputs[0].text for o in thinking_outputs]

    cont_texts = []
    if num_ignores > 0:
        ext_prompts, ext_params = [], []
        for chat, base, out_text, out in zip(
            is_chat_model, thinking_prompts, thinking_texts, thinking_outputs
        ):
            rem = max(1, budget_tokens - len(out.outputs[0].token_ids))
            ext_prompts.append(base + out_text + "Wait")
            p = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=rem,
                min_tokens=1,
            )
            p.stop_token_ids = stop_qwen if chat else stop_llama
            ext_params.append(p)
        cont_outputs = llm.generate(ext_prompts, ext_params) if ext_prompts else []
        cont_texts = [o.outputs[0].text for o in cont_outputs]

    final_prompts = []
    for i, base in enumerate(thinking_prompts):
        reasoning = thinking_texts[i] + (cont_texts[i] if cont_texts else "")
        if is_chat_model[i]:
            final_prompts.append(reasoning + "\nFinal Answer:")
        else:
            final_prompts.append(reasoning)

    final_params = []
    for chat in is_chat_model:
        p = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_final_tokens,
        )
        p.stop_token_ids = stop_qwen if chat else stop_llama
        final_params.append(p)
    final_outputs = llm.generate(final_prompts, final_params) if final_prompts else []

    combined_texts = []
    for i, fo in enumerate(final_outputs):
        reasoning = thinking_texts[i] + (cont_texts[i] if cont_texts else "")
        ans = fo.outputs[0].text
        combined_texts.append(reasoning + ans)

    class MockOut:
        def __init__(self, text): self.text = text
    class MockReq:
        def __init__(self, text): self.outputs = [MockOut(text)]

    return [MockReq(t) for t in combined_texts]

# ================================================================
# Main inference routine
# ================================================================

def infer(args, test_data, answer_extraction_fn):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )

    # Build prompts
    prompts = []
    for example in test_data:
        prompt = ""
        for mess in example["messages"]:
            if mess["role"] == "user":
                if args.model_type == "llama3":
                    prompt += (
                        f"{tokenizer.bos_token}"
                        + "<|start_header_id|>user<|end_header_id|>\n\n"
                        + "Please reason step by step, and put your final answer within \\boxed{}\n"
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

    # Load model
    print("Loading model with vLLM…")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    torch.cuda.synchronize()
    start_time = time()

    # Inference loop
    batch_size = args.eval_batch_size
    all_outputs = []
    print(f"Processing {len(prompts)} prompts in batches of {batch_size}…")
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch inference"):
        batch_prompts = prompts[i : i + batch_size]
        if args.budget_forcing and args.budget_forcing_tokens > 0:
            batch_outputs = inference_with_budget_forcing(
                llm,
                tokenizer,
                batch_prompts,
                args.budget_forcing_tokens,
                args.budget_forcing_ignores,
                args.max_new_tokens,
                args.temperature,
            )
        else:
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=1.0,
                max_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id]
                if tokenizer.eos_token_id is not None
                else None,
            )
            batch_outputs = llm.generate(batch_prompts, sampling_params)
        all_outputs.extend(batch_outputs)

    torch.cuda.synchronize()
    total_time = time() - start_time
    print(
        f"Total inference time: {total_time:.2f}s, Avg per prompt: {total_time/len(prompts):.2f}s"
    )

    model_outputs = [out.outputs[0].text for out in all_outputs]

    # CoT token length calculation
    print("Calculating CoT lengths…")
    with Pool(processes=min(cpu_count(), 8)) as pool:
        cot_lengths = pool.map(
            partial(process_completion, tokenizer=tokenizer), model_outputs
        )

    # Answer extraction
    print("Extracting answers…")
    with Pool(processes=min(cpu_count(), 8)) as pool:
        predictions = pool.map(
            partial(extract_answer_worker, answer_extraction_fn=answer_extraction_fn),
            zip(test_data, model_outputs),
        )

    assert model_outputs, "Model generated empty output list!"

    # Build results
    results = []
    for example, output, pred, cot_len in zip(
        test_data, model_outputs, predictions, cot_lengths
    ):
        item = deepcopy(example)
        item.update(
            {
                "model_output": output,
                "prediction": pred,
                "cot_length": cot_len,
            }
        )
        results.append(item)

    return results, total_time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths & conf
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
        "--model-type",
        type=str,
        choices=["llama3", "qwen", "deepseek"],
        default="qwen",
    )

    # Benchmarks
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["gsm8k", "math", "aime2024", "amc"],
        default="gsm8k",
    )
    parser.add_argument("--data-type", type=str, choices=["train", "test"], default="test")

    # Generation settings
    parser.add_argument("--max_num_examples", type=int, default=10 ** 14)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Budget forcing
    parser.add_argument("--budget_forcing", action="store_true")
    parser.add_argument("--budget_forcing_tokens", type=int, default=32000)
    parser.add_argument("--budget_forcing_ignores", type=int, default=1)

    # Performance
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_parallel_loading_workers", type=int, default=4)
    parser.add_argument("--skip_tokenization_timing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args, _ = parser.parse_known_args()

    print(f"Evaluating {args.model_path}")
    print(
        f"Max new tokens: {args.max_new_tokens}, Min new tokens: {args.min_new_tokens}, "
        f"temperature: {args.temperature}, seed: {args.seed}\n"
    )

    if args.budget_forcing:
        print(
            f"Budget forcing enabled with {args.budget_forcing_tokens} tokens and {args.budget_forcing_ignores} ignores\n"
        )

    # Prepare output directory
    args.output_dir = os.path.join(
        args.output_dir, f"{args.model_size}/", f"Original/{args.data_type}/"
    )

    # Load test configuration
    test_conf = read_data(f"configs/{args.benchmark}_{args.data_type}.json")

    for src, info in test_conf.items():
        fname = os.path.join(args.output_dir, "test_data", "test.jsonl")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        metric_path = os.path.join(args.output_dir, "samples", "metrics.json")
        if os.path.exists(metric_path) and read_data(metric_path).get("n_samples", 0) > 0:
            continue


        # Preprocess dataset --------------------------------------------------
        with open(fname, "w") as f_out:
            data = read_data(info["test_path"])
            for i, sample in enumerate(tqdm(data, desc=f"processing {src}")):
                fn = eval(info["process_fn"])
                sample["id"] = sample.get("id", f"{src}-{i}")
                for j, item in enumerate(fn(sample)):
                    item["dataset"] = src
                    item["id"] = f"{src}-test-{i}-{j}"
                    assert "answer" in item, "Processed sample missing 'answer'!"
                    print(json.dumps(item), file=f_out, flush=True)

        output_dir = os.path.join(args.output_dir, "samples")
        os.makedirs(output_dir, exist_ok=True)

        # Inference -----------------------------------------------------------
        set_random_seed(args.seed)

        print("Loading data…")
        test_data = []
        with open(fname) as fin:
            for line in fin:
                example = json.loads(line)
                messages = example["messages"]
                assert messages[-1]["role"] == "assistant"
                example["reference"] = example.get("reference", "") or [
                    mess["content"] for mess in messages if mess["role"] == "assistant"
                ]
                for mess in messages:
                    if mess["role"] == "assistant":
                        mess["content"] = ""  # hide reference answer
                example["messages"] = messages
                test_data.append(example)

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        results, total_time = infer(args, test_data, info["answer_extraction_fn"])

        print("Finished inference…")

        # Evaluation ----------------------------------------------------------
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        eval_fn = eval(info.get("eval_fn", "eval_math"))
        invalid, labels = [], []
        for item in results:
            if len(item["prediction"]) == 0:
                invalid.append(
                    {
                        "prompt": item["prompt"],
                        "output": item["model_output"],
                        "answer": item["prediction"],
                    }
                )
                labels.append(False)
            else:
                labels.append(eval_fn(item))

        for item, label in zip(results, labels):
            item["accuracy"] = label

        acc = sum(labels) / len(labels)
        avg_len = sum(item["cot_length"] for item in results) / len(results)
        print(f"output acc = {acc * 100:.5f}")
        print(f"output avg_cot_length = {avg_len:.5f}")
        print(f"number of invalid outputs: {len(invalid)}")

        # Save predictions ----------------------------------------------------
        with open(os.path.join(output_dir, "predictions.jsonl"), "a+", encoding="utf-8") as fout:
            for item in results:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Save metrics ---------------------------------------------------------
        with open(os.path.join(output_dir, "metrics.json"), "w") as fout:
            json.dump(
                {
                    "n_samples": len(results),
                    "accuracy": acc,
                    "avg_cot_length": avg_len,
                    "sample_latency": total_time / len(test_data),
                    "budget_forcing": args.budget_forcing,
                    "budget_forcing_tokens": args.budget_forcing_tokens if args.budget_forcing else None,
                    "budget_forcing_ignores": args.budget_forcing_ignores if args.budget_forcing else None,
                },
                fout,
                indent=4,
            )
