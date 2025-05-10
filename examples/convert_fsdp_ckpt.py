"""
Convert the sharded FSDP checkpoint into a hf format.
Reference: https://github.com/volcengine/verl/issues/298

python examples/convert_fsdp_ckpt.py \
    --fsdp_checkpoint_path=checkpoints/past_aime_qwen-1.5B-2000-from3k-from4k/actor/global_step_20 \
    --world_size=16 \
    --huggingface_model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --output_path=checkpoints/past_aime_qwen-1.5B-2000-from3k-from4k-hf/actor/global_step_20

"""
from collections import defaultdict

import torch
from absl import app, flags
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

flags.DEFINE_string(
    "fsdp_checkpoint_path",
    None,
    help="The directory where the fsdp checkpoint is saved."
)
flags.DEFINE_integer(
    "world_size",
    None,
    help="The world size during training."
)
flags.DEFINE_string(
    "huggingface_model_path",
    None,
    help="The huggingface model name."
)
flags.DEFINE_string(
    "output_path",
    None,
    required=False,
    help="The output path. Will be the same as the fsdp checkpoint path if not specified."
)

FLAGS = flags.FLAGS

def main(argv):
    fsdp_checkpoint_path = FLAGS.fsdp_checkpoint_path
    huggingface_model_path = FLAGS.huggingface_model_path
    world_size = FLAGS.world_size
    if FLAGS.output_path is None:
        output_path = fsdp_checkpoint_path
    else:
        output_path = FLAGS.output_path

    state_dict = defaultdict(list)

    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    app.run(main)
