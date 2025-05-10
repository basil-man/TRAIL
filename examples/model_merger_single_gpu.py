import os
import shutil
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', required=True, type=str, help="The path for your saved model")
    parser.add_argument('--save_dir', required=True, type=str, help="The path to save the HF model")
    args = parser.parse_args()

    # 从 huggingface 文件夹读取 config
    hf_config_path = os.path.join(args.local_dir, "huggingface")
    config = AutoConfig.from_pretrained(hf_config_path)

    # 载入单卡训练权重
    model_path = os.path.join(args.local_dir, "model_world_size_1_rank_0.pt")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    state_dict = torch.load(model_path, map_location='cpu')

    # 识别模型架构
    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    # 构造模型并加载参数
    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')
    model.load_state_dict(state_dict)

    # 保存为 HuggingFace 格式
    print(f"Saving model to {args.save_dir}")
    model.save_pretrained(args.save_dir)

    # 复制 tokenizer 文件
    tokenizer_files = [
        "added_tokens.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt"
    ]
    for file_name in tokenizer_files:
        src = os.path.join(args.local_dir, file_name)
        if os.path.exists(src):
            shutil.copy(src, args.save_dir)

    print("Model and tokenizer saved successfully.")
