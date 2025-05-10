# This script is a simple math evaluation script using VLLM with one GPU.

import os
import re
import json
import copy
import click
import shutil
import logging
import importlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
from functools import partial
from typing import Callable, List, Any, Dict, Union

from omegaconf import OmegaConf
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI

from verl.evaluation import REWARD_REGISTRY, pass_at_k
from verl.evaluation.data_utils import DATA_REGISTRY
from verl.utils.logging_utils import set_basic_config

set_basic_config(logging.INFO)

LOGGER = logging.getLogger(__name__)


def call_api(messages, temperature=1.0, top_p=1.0, n=1, stop=None, seed=None, max_tokens=None, port=8000):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        seed=seed,
    )
    return result


def build_dataset(data_path):
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
        if data_path in DATA_REGISTRY:
            dataset = DATA_REGISTRY.get(data_path)(dataset)
    return dataset


def build_vllm(model_name: str):
    llm = LLM(model=model_name, dtype="bfloat16", max_model_len=10240)
    return llm


def clean_model_name(model_name):
    def find_step_num(name):
        match = re.search(r"step_(\d+)", name)
        return int(match.group(1)) if match else None

    if os.path.exists(model_name):
        stepnum = find_step_num(model_name.split("/")[-1])
        config = json.load(open(os.path.join(model_name, "config.json")))
        return f"{config['_name_or_path'].split('/')[-1]},step@{stepnum}"
    else:
        return model_name.split("/")[-1]


def clean_prompt_name(prompt_file):
    return prompt_file.split("/")[-1].rstrip(".txt").rstrip(".py")


def samplingParam2json(sampling_param):
    param_str = str(sampling_param)
    # Extract the parameters into a JSON dictionary
    pattern = r"(\w+)=([\[\]{}0-9\.\-]+|True|False|None|'[^']*'|\"[^\"]*\")"
    matches = re.findall(pattern, param_str)
    params_dict = {
        key: eval(value)
        if value not in ["None", "True", "False"]
        else (None if value == "None" else (True if value == "True" else False))
        for key, value in matches
    }
    return params_dict


def batch_iterator(dataset: Dataset, preprocess_fn: Callable, batch_size: int):
    all_batches = [
        [
            preprocess_fn(x)
            for x in dataset.select(range(i, min(i + batch_size, len(dataset))))
        ]
        for i in range(0, len(dataset), batch_size)
    ]

    for batch in all_batches:
        yield batch


def load_constants(template_file):
    assert template_file.endswith(".py"), "The template file should be a python file"
    module_name = os.path.splitext(os.path.basename(template_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, template_file)
    if spec is None:
        raise ImportError(f"Cannot import {module_name} from {template_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)["MESSAGE_TEMPLATE"]


class CustomDefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep the placeholder as-is


def replace_prompt(
    prompt_list: List[Dict[str, str]], replace_dict: Dict[str, str]
) -> List[Dict[str, str]]:
    prompt_list = copy.deepcopy(prompt_list)
    prompt_list = [
        {
            k: v.format_map(CustomDefaultDict(**replace_dict))
            if isinstance(v, str)
            else v
            for k, v in prompt.items()
        }
        for prompt in prompt_list
    ]
    return prompt_list


def process_sample(
    sample: Dict[str, Any],
    input_key: List[str],
    template: Union[List[str], str],
    apply_template_fn: Callable = None,
):
    # needed_inputs = {k: sample[k] for k in input_key}
    # if apply_template_fn is not None:
    #     chat = replace_prompt(template, needed_inputs)
    #     prompt = apply_template_fn(chat, tokenize=False, add_generation_prompt=True)
    # else:
    #     prompt = template.format_map(CustomDefaultDict(**needed_inputs))
    prompt = apply_template_fn(sample["prompt"], tokenize=False, add_generation_prompt=True)
    sample["input"] = prompt
    return sample


def dummy_process(
    sample: Dict[str, Any],
    input_key: List[str],
    template: Union[List[str], str],
    apply_template_fn: Callable = None,
):
    return sample


@click.command()
# Experiment configs
@click.option("--outdir", prompt="The output directory")
@click.option("--run_name", prompt="The experiment name")
@click.option("--model_name", prompt="The model name")
# @click.option("--use_server", is_flag=True, default=False, prompt="Whether to use vLLM server")
# @click.option("--port", default=8000, prompt="Port number for vLLM server")
# Data configs
@click.option("--data_path", prompt="The path to data")
@click.option("--prompt_file", prompt="The path to prompt template")
@click.option("--start_index", default=-1, prompt="Start index for data point")
@click.option("--end_index", default=-1, prompt="End index for data point")
# Evlauation configs
@click.option(
    "--eval_fn",
    default="deepseek_math",
    prompt="The name of reward function, find it in math_utils.py",
)
# Sampling configs
@click.option("--temperature", default=1.0, prompt="The temperature for sampling")
@click.option("--top_p", default=1.0, prompt="The ratio for TopP sampling")
@click.option("--n", default=8, prompt="The number of responses to sample per prompt")
@click.option(
    "--max_tokens", default=2048, prompt="The number of responses to sample per prompt"
)
@click.option(
    "--input_key",
    nargs=1,
    default={"problem"},
    prompt="Maximum nubmer of samples",
    multiple=True,
)

def main(**configs):
    configs["stop"] = None
    configs["use_server"] = False
    if configs["stop"] is not None:
        configs["stop"] = list(configs["stop"])
    if configs["input_key"] is not None:
        configs["input_key"] = list(configs["input_key"])

    assert os.path.exists(configs["prompt_file"]), "The prompt file does not exist"

    # set up out dir
    now = datetime.now()
    now = now.strftime("%m-%d-%H-%M-%S")
    basedir = Path(configs["outdir"]) / configs["run_name"]
    basedir = (
        basedir
        / f"model@{clean_model_name(configs['model_name'])},prompt@{clean_prompt_name(configs['prompt_file'])},time@{now}"
    )
    basedir.mkdir(parents=True, exist_ok=True)
    # set up configs
    sampling_params = SamplingParams(
        temperature=configs["temperature"],
        top_p=configs["top_p"],
        n=configs["n"],
        max_tokens=configs["max_tokens"],
        stop=configs["stop"],
    )
    # save configs
    out_sampling_params = samplingParam2json(sampling_params)
    configs["sampling_params"] = out_sampling_params
    OmegaConf.save(OmegaConf.create(configs), basedir / "configs.yaml")
    shutil.copy(configs["prompt_file"], basedir / "prompt_template.txt")

    # setup model and data
    dataset = build_dataset(configs["data_path"])
    if configs["start_index"] != -1:
        dataset = dataset.select(range(configs["start_index"], min(configs["end_index"], len(dataset))))

    if configs["prompt_file"].endswith(".py"):
        MESSAGE_TEMPLATE = load_constants(configs["prompt_file"])
    elif configs["prompt_file"].endswith(".txt"):
        MESSAGE_TEMPLATE = open(configs["prompt_file"], "r").read().strip()
    else:
        raise ValueError("The prompt file should be either a .py or .txt file")
    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])
    process_fn = dummy_process if configs["use_server"] else process_sample
    DATA_PREPROCESS_FN = partial(
        process_fn,
        template=MESSAGE_TEMPLATE,
        input_key=configs["input_key"],
        apply_template_fn=tokenizer.apply_chat_template,
    )
    EVAL_FN = REWARD_REGISTRY.get(configs["eval_fn"])
    if not configs["use_server"]:
        llm = build_vllm(configs["model_name"])
    exist_uid = json.load(open("local_scripts/exist_uid.json", "r"))
    # exist_uid = []

    # evaluation
    all_results = []
    # import pdb; pdb.set_trace()
    for batch in tqdm(batch_iterator(dataset, DATA_PREPROCESS_FN, 1)):
        if batch[0]["uid"] in exist_uid:
            continue
        if configs["use_server"]:
            messages = [x['prompt'] for x in batch][0]
            outputs = call_api(messages, configs["temperature"], configs["top_p"], configs["n"], port=configs["port"])
        else:
            inputs = [x["prompt"] for x in batch]
            outputs = llm.chat(inputs, sampling_params=sampling_params, add_generation_prompt=True, use_tqdm=False)
        for sample, output in zip(batch, outputs):
            for response in output.outputs:
                tmp_sample = copy.deepcopy(sample)
                response_str = response.text
                tmp_sample["response"] = response_str
                tmp_sample["uid"] = sample["uid"]
                tmp_sample["match"] = float(EVAL_FN(response_str, sample["gt_answer"]))
                tmp_sample["response_length"] = len(response.token_ids)
                all_results.append(tmp_sample)
        
        if len(all_results) % 100 == 0:
            result_dataset = Dataset.from_list(all_results)
            result_dataset.save_to_disk(basedir / "results_data")

    if len(all_results) == 0:
        return
    # save results
    result_dataset = Dataset.from_list(all_results)
    result_dataset.save_to_disk(basedir / "results_data")

    pass_at_ks = {}
    for k in [1, 2, 4, 8]:
        res = pass_at_k(all_results, k)
        if res is not None:
            pass_at_ks[f"pass@{k}"] = res
    all_lengths = [x["response_length"] for x in all_results]
    pass_at_ks["length_mean"] = np.mean(all_lengths).item()
    pass_at_ks["length_std"] = np.std(all_lengths).item()
    pass_at_ks["length_min"] = np.min(all_lengths).item()
    pass_at_ks["length_max"] = np.max(all_lengths).item()
    json.dump(
        {
            "model": clean_model_name(configs["model_name"]),
            "prompt": clean_prompt_name(configs["prompt_file"]),
            "metrics": pass_at_ks,
        },
        open(basedir / "results.json", "w"),
    )


if __name__ == "__main__":
    main()
