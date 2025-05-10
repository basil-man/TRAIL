# This file contains utility functions for the PPO trainer.
import os
import re
import glob
import torch
import shutil
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, Any

CKPT_NAME_PATTERN = r"global_step_(\d+)"
# OPTIMIZER_STATE_NAME = "optimizer_state.pth"
OPTIMIZER_STATE_NAME = "optimizer.pt"


def rotate_clean_checkpoint(
    ckpt_dir: str,
    keep_num: int = 1,
    clean_optimizer: bool = True,
):
    # This function deletes the optimizer state for older checkpoints
    # all_checkpoints = [glob.glob(os.path.join(ckpt_dir, ckpt)) for ckpt in os.listdir(ckpt_dir)]
    pattern = re.compile(CKPT_NAME_PATTERN)
    ckpt_dirs = glob.glob(os.path.join(ckpt_dir, "global_step_*"))

    matched_folders = []
    for folder_name in ckpt_dirs:
        match = pattern.search(os.path.basename(folder_name))
        if match:
            step_number = int(match.group(1))
            matched_folders.append((step_number, folder_name))

    # Sort by step number
    matched_folders.sort(key=lambda x: x[0])

    # Delete the optimizer state for older checkpoints
    temp_dirs = [x[1] for x in matched_folders]
    if len(temp_dirs) > keep_num:
        for temp_dir in temp_dirs[:-keep_num]:
            print("Deleting optimizer state for", temp_dir)
            shutil.rmtree(temp_dir)

    for temp_dir in temp_dirs[-keep_num:-1]:
        if clean_optimizer:
            # if os.path.exists(os.path.join(temp_dir, OPTIMIZER_STATE_NAME)):
            #     os.remove(os.path.join(temp_dir, OPTIMIZER_STATE_NAME))
            for file_path in glob.glob(os.path.join(temp_dir, "optim*")):
                os.remove(file_path)


def find_latest_checkpoint(ckpt_dir):
    checkpoints = glob.glob(os.path.join(ckpt_dir, "global_step_*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    return checkpoints[-1]


# TODO jiabao: consider make it more general
def save_checkpoint(
    local_path: str,
    global_states: Dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    remote_path: str = None,
):
    # Save the model and optimizer state
    model.save_checkpoint(local_path, remote_path)
    # Save global information
    torch.save(
        global_states,
        open(
            os.path.join(
                local_path,
                "global_states.pth",
            ),
            "wb",
        ),
    )
