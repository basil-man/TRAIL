# TRAIL: Token-level Reinforcement And Imitation Learning

**TRAIL** is a reinforcement + imitation learning framework for **adaptive Chain-of-Thought (CoT) compression** in large language models (LLMs). It enables models to dynamically adjust reasoning length — elaborating when necessary, and skipping redundancy when possible.

## 🔍 Motivation

Modern LLMs like GPT and DeepSeek achieve strong performance by generating long reasoning traces, but suffer from high inference costs due to quadratic attention ($O(L^2)$). TRAIL addresses this by:
- ✅ Dynamically adjusting CoT length per problem  
- 🚀 Achieving 1.5–2.5× inference speedups  
- 🧠 Minimizing or improving accuracy  

## ⚙️ Key Components

- **GRPO (Group Relative Policy Optimization):** learns from competing trajectories to balance accuracy and brevity.  
- **Replay-buffer distillation:** periodically distills strong traces to stabilize learning.  
- **Token-level reward shaping:** assigns fine-grained rewards based on log-probabilities using LLMLingua.  

## 📊 Experimental Results

| Dataset   | Token Reduction | Accuracy Impact |
|-----------|------------------|-----------------|
| GSM8K     | 60.2%            | −0.5%           |
| MATH-500  | 35.8%            | ±0.0%           |
| AMC       | 15.8%            | **+4.8%**       |

All results are based on a 3B parameter model.

## 🧩 Environment Setup

We recommend using `conda` to manage dependencies:

```bash
conda create -n trail python=3.10 -y
conda activate trail
pip install -e .
pip install flash_attn
```

## 📚 Data Preprocessing

We use the [EleutherAI MATH dataset](https://huggingface.co/datasets/EleutherAI/hendrycks_math). To preprocess the data:

```bash
python examples/data_preprocess/preprocess_math.py \
  --model_family=qwen \
  --save_dir=data/math_qwen
```

This script formats questions and adds system prompts in the style required by Verl-compatible training.

## 🏋️ Training & Evaluation

To train and evaluate TRAIL using predefined hyperparameters:

```bash
bash run-ntimes-qw.sh
```

After training, evaluation results will be saved as:

```
all_results_${RUN_NAME}.csv
```

Adjust `RUN_NAME` in the script to distinguish between experimental runs.

## 💾 Model Weights

You can download checkpoints from Hugging Face:

- 🔗 [TRAIL-1.01-Qwen3B](https://huggingface.co/BSL1/TRAIL-1.01-Qwen3B)
- 🔗 [TRAIL-1.05-Qwen3B](https://huggingface.co/BSL1/TRAIL-1.05-Qwen3B)
- 🔗 [TRAIL-1.1-Qwen3B](https://huggingface.co/BSL1/TRAIL-1.1-Qwen3B)
- 🔗 [TRAIL-1.2-Qwen3B](https://huggingface.co/BSL1/TRAIL-1.2-Qwen3B)
- 🔗 [TRAIL-1.4-Qwen3B](https://huggingface.co/BSL1/TRAIL-1.4-Qwen3B)


## 📁 Project Structure

```
TRAIL/
    ├── LICENSE
    ├── README.md
    ├── analyzer_host.py
    ├── analyzer_utils.py
    ├── configs/
    ├── data_processing/
    ├── datasets/
    ├── eval/
    ├── eval_rl.py
    ├── eval_rl.sh
    ├── examples/
    ├── prompt_templates/
    ├── pyproject.toml
    ├── requirements.txt
    ├── run-ntimes-ds.sh
    ├── run-ntimes-qw.sh
    ├── setup.py
    └── verl/
```

## 🙏 Acknowledgements

This project builds on:
- [TokenSkip](https://github.com/hemingkx/TokenSkip)
- [ThinkPrune](https://github.com/UCSB-NLP-Chang/ThinkPrune)
- [Verl](https://github.com/volcengine/verl)
