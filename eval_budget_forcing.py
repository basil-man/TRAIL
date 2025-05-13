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

from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data


def infer_with_budget_forcing(args, test_data, answer_extraction_fn):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print("Loading model with vLLM...")
    llm = LLM(model=args.model_path, tokenizer=args.tokenizer_path, dtype="float16")

    results = []
    torch.cuda.synchronize()
    start_time = time()
    
    for example in tqdm(test_data, desc="Processing examples"):
        # Prepare prompt
        prompt = ""
        for mess in example['messages']:
            if mess['role'] == 'user':
                prompt += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{mess['content']}<|im_end|>\n<|im_start|>assistant\n"
            elif mess['role'] == 'assistant':
                prompt += mess['content'].rstrip()
            prompt = prompt.lstrip()
        
        example['prompt'] = prompt
        
        # Add thinking start tag
        think_prompt = prompt + "开始思考"
        
        # Initial thinking phase
        stop_token_ids = tokenizer("<|im_start|><|im_end|>")["input_ids"]
        thinking_params = SamplingParams(
            max_tokens=args.max_tokens_thinking,
            min_tokens=0,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        
        thinking_output = llm.generate(think_prompt, sampling_params=thinking_params)
        
        # Continue thinking with ignore mechanism
        max_tokens_thinking_tmp = args.max_tokens_thinking
        full_prompt = think_prompt + thinking_output[0].outputs[0].text
        
        # Handle ignoring stop tokens if needed
        for i in range(args.num_ignore):
            max_tokens_thinking_tmp -= len(thinking_output[0].outputs[0].token_ids)
            full_prompt += args.ignore_str
            
            thinking_params = SamplingParams(
                max_tokens=max_tokens_thinking_tmp,
                min_tokens=1,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                temperature=0.0,
            )
            
            thinking_output = llm.generate(full_prompt, sampling_params=thinking_params)
            full_prompt += thinking_output[0].outputs[0].text
        
        # Final answer phase
        stop_token_ids = tokenizer("<|im_end|>")["input_ids"]
        answer_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
        )
        
        final_output = llm.generate(full_prompt, sampling_params=answer_params)
        model_output = final_output[0].outputs[0].text
        
        # Calculate thinking length
        thinking_text = full_prompt[len(prompt + "<|im_start|>think"):]
        thinking_length = len(tokenizer(thinking_text, return_tensors="pt")['input_ids'][0])
        
        # Extract the answer
        prediction = eval(answer_extraction_fn)(example['messages'][-2]['content'], model_output, task='cot')
        
        item = deepcopy(example)
        item.update({
            'model_output': model_output,
            'thinking': thinking_text,
            'prediction': prediction,
            'thinking_length': thinking_length,
        })
        results.append(item)
    
    torch.cuda.synchronize()
    total_time = time() - start_time
    
    return results, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/s1.1-3B/math/")
    parser.add_argument("--model-path", type=str, default="simplescaling/s1.1-3B")
    parser.add_argument("--tokenizer-path", type=str, default="simplescaling/s1.1-3B")
    parser.add_argument("--model-size", type=str, default="3b")
    parser.add_argument("--model-type", type=str, default="s1")
    parser.add_argument("--benchmark", type=str, default="math")
    parser.add_argument("--data-type", type=str, default="test")

    parser.add_argument("--max_num_examples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # Budget forcing parameters
    parser.add_argument("--max_tokens_thinking", type=int, default=32000)
    parser.add_argument("--num_ignore", type=int, default=1)
    parser.add_argument("--ignore_str", type=str, default="Wait")
    
    args, _ = parser.parse_known_args()

    print(f"Evaluating {args.model_path} with budget forcing")
    print(f"Max thinking tokens: {args.max_tokens_thinking}, Ignore times: {args.num_ignore}")
    print(f"Max new tokens: {args.max_new_tokens}, temperature: {args.temperature}, seed: {args.seed}\n")

    # 修改输出目录名称以反映当前的tokens设置
    args.output_dir = os.path.join(args.output_dir, f"{args.model_size}/", f"BudgetForcing_tokens_{args.max_tokens_thinking}/{args.data_type}/")
    
    test_conf = read_data(f"configs/{args.benchmark}_{args.data_type}.json")

    for src, info in test_conf.items():
        fname = os.path.join(args.output_dir, "test_data", "test.jsonl")
        input_dir = os.path.dirname(fname)
        os.makedirs(input_dir, exist_ok=True)
        metric_path = os.path.join(args.output_dir, "samples", "metrics.json")
        if os.path.exists(metric_path) and read_data(metric_path)['n_samples'] > 0:
            continue

        with open(fname, "w") as file:
            data = read_data(info['test_path'])
            for i, sample in enumerate(tqdm(data, desc=f'processing {src}')):
                fn = eval(info['process_fn'])
                sample['id'] = sample.get('id', f"{src}-{i}")
                for j, item in enumerate(fn(sample)):
                    item['dataset'] = src
                    item['id'] = f"{src}-test-{i}-{j}"
                    assert 'answer' in item
                    print(json.dumps(item), file=file, flush=True)

            output_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(output_dir, exist_ok=True)

        set_random_seed(args.seed)

        print("Loading data...")
        test_data = []
        with open(os.path.join(input_dir, f"test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                messages = example['messages']
                assert messages[-1]['role'] == 'assistant'
                example['reference'] = example.get('reference', '') or [mess['content'] for mess in messages if mess['role'] == 'assistant']
                for mess in messages:
                    if mess['role'] == 'assistant':
                        mess['content'] = ''
                example['messages'] = messages
                test_data.append(example)

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results, total_time = infer_with_budget_forcing(args, test_data, info['answer_extraction_fn'])

        print("Finished inference...")

        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        invalid_outputs = []
        labels = []
        for item in results:
            if len(item['prediction']) == 0:
                invalid_outputs.append({'prompt': item['prompt'], 'output': item['model_output'], 'answer': item['prediction']})
                res = False
            else:
                res = eval_math(item)
            labels.append(res)

        for item, label in zip(results, labels):
            item['accuracy'] = label

        print("Calculating accuracy...")
        acc = sum(item['accuracy'] for item in results) / len(results)
        print("output acc = {:.5f}".format(acc * 100))

        avg_thinking_length = sum(item['thinking_length'] for item in results) / len(results)
        print("output avg_thinking_length = {:.5f}".format(avg_thinking_length))
        print("number of invalid outputs: {}".format(len(invalid_outputs)))

        pred_fname = "predictions.jsonl"
        for item in results:
            with open(os.path.join(output_dir, pred_fname), 'a+', encoding='utf-8') as fout:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')

        metric_fname = "metrics.json"
        with open(os.path.join(output_dir, metric_fname), "w") as fout:
            json.dump({
                "n_samples": len(results),
                "accuracy": acc,
                "avg_thinking_length": avg_thinking_length,
                "sample_latency": total_time / len(test_data),
            }, fout, indent=4)
