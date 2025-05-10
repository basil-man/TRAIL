# This file contains functions to obtain all randomness states and reset.

import torch
import torch.distributed as dist
import random
import numpy as np
from typing import Dict, Any


# Get states
def get_rng_states():
    state_random = random.getstate()
    state_numpy = np.random.get_state()
    state_torch = torch.get_rng_state()
    if dist.is_initialized():
        state_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
    else:
        state_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    return {
        "python": state_random,
        "numpy": state_numpy,
        "torch": state_torch,
        "cuda": state_cuda,
    }


# Function to restore states
def restore_rng_states(state_json: Dict[str, Any]):
    state_random = state_json["python"]
    state_numpy = state_json["numpy"]
    state_torch = state_json["torch"]
    state_cuda = state_json["cuda"]

    random.setstate(state_random)
    np.random.set_state(state_numpy)
    torch.set_rng_state(state_torch)
    if state_cuda:
        if dist.is_available():
            torch.cuda.set_rng_state_all(state_cuda)
        else:
            torch.cuda.set_rng_state(state_cuda)
