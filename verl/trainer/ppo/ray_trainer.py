# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import json
import ray
import random
import asyncio
import aiohttp  # 导入用于异步HTTP请求的模块
import time

from functools import reduce
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
import torch.distributed
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoFuture
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.trainer_utils import (
    rotate_clean_checkpoint,
    find_latest_checkpoint,
    save_checkpoint,
)
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.random_utils import get_rng_states, restore_rng_states
from verl.utils.dataset.sampler import DistributedSampler

WorkerType = Type[Worker]

# 全局客户端会话和锁
_global_http_session: Optional[aiohttp.ClientSession] = None
_analyzer_lock = asyncio.Lock()


# 创建或获取全局HTTP会话
async def get_http_session():
    global _global_http_session
    if (_global_http_session is None) or _global_http_session.closed:
        _global_http_session = aiohttp.ClientSession()
    return _global_http_session


# 确保会话在退出时关闭
@asynccontextmanager
async def managed_http_session():
    session = await get_http_session()
    try:
        yield session
    except Exception as e:
        print(f"HTTP session operation error: {str(e)}")
        # 异常情况下不关闭会话，让它继续存在供后续使用


# 关闭全局会话
async def close_global_session():
    global _global_http_session
    if _global_http_session and not _global_http_session.closed:
        await _global_http_session.close()
        _global_http_session = None
        print("Global HTTP session closed")


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def reset_pools(self):
        """尝试重置所有资源池的状态"""
        for name, pool in self.resource_pool_dict.items():
            try:
                print(f"Trying to reset resource pool: {name}")
                # 这里不是标准API，但尝试调用可能存在的方法
                if hasattr(pool, "reset") and callable(pool.reset):
                    pool.reset()
                # 或者尝试释放资源
                if hasattr(pool, "release") and callable(pool.release):
                    pool.release()
            except Exception as e:
                print(f"Failed to reset resource pool {name}: {e}")


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    add_process_reward=False,
):
    # prepare response group
    # TODO: add other ways to estimate advantages
    token_level_rewards = data.batch["token_level_rewards"]
    responses = data.batch["responses"]
    response_length = responses.size(-1)
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    if adv_estimator == "gae":
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == "grpo":
        index = data.non_tensor_batch["uid"]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        if add_process_reward:
            process_advantages, process_returns = core_algos.compute_grpo_process_advantage(
                data.batch["rm_scores"],
                data.batch["step_indicator"],
                data.batch["avg_pr_per_step"],
                response_mask,
                index,
            )
            advantages += process_advantages
            returns += process_returns
    elif adv_estimator.startswith("rloo"):
        index = data.non_tensor_batch["uid"]
        process_reward = data.batch["rm_scores"] if add_process_reward and "critic" not in adv_estimator else None
        advantages, returns = core_algos.compute_rloo_advantage(
            token_level_rewards,
            response_mask,
            index,
            adv_estimator,
            process_reward,
            data.batch["step_indicator"] if "step_indicator" in data.batch else None,
            data.batch["avg_pr_per_step"] if "avg_pr_per_step" in data.batch else None,
        )
        if adv_estimator == "rloo-critic" and "rm_scores" in data.batch:
            pr_advantages, pr_returns = core_algos.compute_step_pg_advantage_return(
                token_level_rewards=data.batch["rm_scores"],
                values=data.batch["values"],
                eos_mask=response_mask,
                gamma=gamma,
                lam=lam,
                step_indicator=data.batch["step_indicator"],
            )
            if add_process_reward:
                # advantages combine both outcome and process rewards
                advantages += pr_advantages
            # returns only use process rewards, because critic only fits on process rewards
            returns = pr_returns
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # score & response length for each level
    level_score = {}
    level_length = {}
    levels = batch.non_tensor_batch["level"]  # numpy array
    for level in np.unique(levels):
        index = torch.from_numpy(levels == level)
        level_score[level] = sequence_score[index]
        level_length[level] = response_length[index]

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # level score & length
        **{f"level_score/{level}": torch.mean(level_score[level]).detach().item() for level in level_score.keys()},
        **{f"level_length/{level}": torch.mean(level_length[level]).detach().item() for level in level_length.keys()},
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


async def request_analyzerhost(endpoint, json_data=None, timeout=10, max_retries=3, hard_timeout=30):
    """发送请求到analyzerhost，支持重试和硬超时"""
    url = f"http://localhost:5000/{endpoint}"

    # 创建总体超时的计时器
    start_time = time.time()
    retries = 0
    last_error = None

    while retries < max_retries and (time.time() - start_time) < hard_timeout:
        try:
            async with managed_http_session() as session:
                # 设置请求超时
                request_timeout = min(timeout, hard_timeout - (time.time() - start_time))
                if request_timeout <= 0:
                    break

                print(f"Sending request to {endpoint}, attempt {retries+1}/{max_retries}...")
                # 发送请求
                if json_data:
                    async with session.post(url, json=json_data, timeout=request_timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            last_error = f"Request failed: HTTP {response.status}, {await response.text()}"
                else:
                    async with session.post(url, timeout=request_timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            last_error = f"Request failed: HTTP {response.status}, {await response.text()}"

        except asyncio.TimeoutError:
            last_error = (
                f"Request timeout (attempt {retries+1}/{max_retries}, time elapsed {time.time() - start_time:.1f}s)"
            )
        except aiohttp.ClientConnectorError as e:
            last_error = f"Connection error: {str(e)} (attempt {retries+1}/{max_retries})"
        except Exception as e:
            last_error = f"Request error: {str(e)} (attempt {retries+1}/{max_retries})"

        retries += 1
        # 指数退避重试
        if retries < max_retries:
            await asyncio.sleep(min(1 * (2 ** (retries - 1)), 5))

    print(f"Failed to request analyzerhost: {last_error}")
    return {"status": "error", "message": last_error}


async def unload_analyzer_model(skip_lock=False):
    """请求卸载analyzerhost的模型，带锁保护"""
    if skip_lock:
        print("Requesting to unload analyzerhost model...")
        result = await request_analyzerhost("unload_model", timeout=20, hard_timeout=60)
        if result and result.get("status") == "success":
            print("Analyzerhost model unloaded")
            return True
        print(f"Failed to unload analyzerhost model: {result}")
        return False
    else:
        async with _analyzer_lock:
            return await unload_analyzer_model(skip_lock=True)


async def load_analyzer_model(skip_lock=False):
    """请求重新加载analyzerhost的模型，带锁保护"""
    if skip_lock:
        print("Requesting to reload analyzerhost model...")
        result = await request_analyzerhost("load_model", timeout=20, hard_timeout=60)
        if result and result.get("status") == "success":
            print("Analyzerhost model reloaded")
            return True
        print(f"Failed to reload analyzerhost model: {result}")
        return False
    else:
        async with _analyzer_lock:
            return await load_analyzer_model(skip_lock=True)


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'
        # 添加Replay Buffer
        self.use_replay_buffer = config.algorithm.get("use_replay_buffer", False)
        if self.use_replay_buffer:
            from verl.trainer.ppo.replay_buffer import ReplayBuffer

            self.replay_buffer = ReplayBuffer(
                capacity=config.algorithm.replay_buffer.capacity,
                top_percent=config.algorithm.replay_buffer.get("top_percent", 0.05),
                min_reward=config.algorithm.replay_buffer.get("min_reward", 0.85),
                min_distill_interval=config.algorithm.replay_buffer.get("min_distill_interval", 2),
                max_distill_interval=config.algorithm.replay_buffer.get("max_distill_interval", 20),
                distill_threshold_ratio=config.algorithm.replay_buffer.get("distill_threshold_ratio", 0.3),
                microbatch_size=config.algorithm.replay_buffer.get("microbatch_size", 8),  # 添加microbatch_size参数
            )
            self.off_policy_weight = config.algorithm.replay_buffer.off_policy_weight
            self.off_policy_batch_size = config.algorithm.replay_buffer.batch_size
            self.max_is_weight = config.algorithm.replay_buffer.max_is_weight
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert (
                    config.algorithm.kl_ctrl.horizon > 0
                ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )

        self.train_sampler = DistributedSampler(
            dataset=self.train_dataset,
            seed=self.config.trainer.seed,
            shuffle=True,
            drop_last=True,
            consumed_samples=0,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=self.train_sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation="error",
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    async def async_validate(self):
        all_results = []
        all_val_results = []
        uid_start = 0
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            test_gen_batch = test_batch.pop(["input_ids", "attention_mask", "position_ids"])
            do_sample = self.config.trainer.multisample_val
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": do_sample,
                "validate": True,
            }
            if self.config.trainer.multisample_val:
                test_gen_batch.meta_info["temperature"] = 0.6
                # test_gen_batch.meta_info["max_tokens"] = 8000
                # test_gen_batch.meta_info["max_tokens"] = 32000
                test_gen_batch.meta_info["top_p"] = 0.95

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            if self.config.trainer.multisample_val:
                pad_size *= self.config.actor_rollout_ref.rollout.n
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            test_batch.non_tensor_batch["uid"] = np.array(
                list(range(uid_start, uid_start + len(test_gen_batch))), dtype=object
            )
            uid_start += len(test_gen_batch)
            if self.config.trainer.multisample_val:
                # repeat to align with repeated responses in rollout
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            val_result = await self.val_reward_fn(test_batch)
            if isinstance(val_result, DataProto):
                reward_tensor = val_result.batch["token_level_scores"]
            else:
                reward_tensor = val_result

            val_result.non_tensor_batch["data_source"] = test_batch.non_tensor_batch.get(
                "data_source", ["unknown"] * reward_tensor.shape[0]
            )
            all_val_results.append(val_result)

            # Cache results
            for i in range(len(test_batch)):
                data_item = test_batch[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                all_results.append(
                    {
                        "problem": self.tokenizer.decode(valid_prompt_ids),
                        "response": self.tokenizer.decode(valid_response_ids),
                        "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                    }
                )

        # Cache reults to local
        eval_log_path = (
            Path(
                self.config.trainer.default_local_dir,
            )
            / "evallog"
        )
        eval_log_path.mkdir(parents=True, exist_ok=True)
        eval_log_path = str(
            eval_log_path / f"global_step_{self.global_steps}.jsonl",
        )
        with open(eval_log_path, "w") as f:
            for line in all_results:
                f.write(json.dumps(line) + "\n")

        # Merge all val_result
        all_val_result = reduce(lambda x, y: x.concat(y), all_val_results)

        metric_dict = {}
        per_sample_results = all_val_result.groupby("uid")
        for todo_key in ["data_source", "level"]:
            if todo_key in all_val_result.non_tensor_batch:
                score_lst = defaultdict(list)
                for sample in per_sample_results:
                    sample: DataProto
                    scores = sample.batch["token_level_scores"]
                    scores = scores.sum(-1).cpu().tolist()
                    group_key = sample.non_tensor_batch[todo_key][0]

                    for passk in [1, 2, 4, 8]:
                        if passk <= len(sample):
                            tmpscore = [max(scores[i : i + passk]) for i in range(0, len(scores), passk)]
                            passkscore = np.mean(tmpscore)
                            score_lst[f"{group_key}-pass@{passk}"].append(passkscore)

                for key, val in score_lst.items():
                    metric_dict[f"val/test_score/{key}"] = np.mean(val)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            if self.config.trainer.resume_checkpoint:
                self.config.actor_rollout_ref.model.checkpoint_path = os.path.join(
                    self.config.trainer.default_local_dir, "actor"
                )

            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator in ["gae", "rloo-critic"]:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            if self.config.trainer.resume_checkpoint:
                self.config.critic.model.checkpoint_path = os.path.join(self.config.trainer.default_local_dir, "critic")

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ["grpo", "rloo", "rloo-prime", "rloo-correct"]:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    # TODO jiabao: by 02/07, right now we only consider saving actor sice most experiments done on grpo, but may need to consider critic for PPO training
    def _save_checkpoint(self):
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir,
            "actor",
            f"global_step_{self.global_steps}",
        )
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "actor")
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(
                self.config.trainer.default_local_dir,
                "critic",
                f"global_step_{self.global_steps}",
            )
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, "critic")
            )
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

        # Save global information
        global_states = {
            "global_steps": self.global_steps,
            "epochs": self.epochs,
            "consumed_samples": self.consumed_samples,
            "rng_states": get_rng_states(),
        }
        torch.save(
            global_states,
            open(
                os.path.join(
                    actor_local_path,
                    "global_states.pth",
                ),
                "wb",
            ),
        )

        rotate_clean_checkpoint(
            ckpt_dir=os.path.join(
                self.config.trainer.default_local_dir,
                "actor",
            ),
            keep_num=self.config.trainer.get("num_keep_checkpoint", 1),
            clean_optimizer=self.config.trainer.get("clean_optimizer", False),
        )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    async def fit_async(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # 初始化 Replay Buffer (如果配置中启用)
        use_replay_buffer = self.config.algorithm.get("use_replay_buffer", False)
        if use_replay_buffer and not hasattr(self, "replay_buffer"):
            from verl.trainer.ppo.replay_buffer import ReplayBuffer

            self.replay_buffer = ReplayBuffer(
                capacity=self.config.algorithm.replay_buffer.capacity,
                top_percent=self.config.algorithm.replay_buffer.get("top_percent", 0.05),
                min_reward=self.config.algorithm.replay_buffer.get("min_reward", 0.85),
                min_distill_interval=self.config.algorithm.replay_buffer.get("min_distill_interval", 2),
                max_distill_interval=self.config.algorithm.replay_buffer.get("max_distill_interval", 10),
                distill_threshold_ratio=self.config.algorithm.replay_buffer.get("distill_threshold_ratio", 0.3),
                microbatch_size=self.config.algorithm.replay_buffer.get("microbatch_size", 8),  # 添加microbatch参数
            )
            self.off_policy_weight = self.config.algorithm.replay_buffer.get("off_policy_weight", 0.1)
            self.off_policy_batch_size = self.config.algorithm.replay_buffer.get("batch_size", 64)
            self.max_is_weight = self.config.algorithm.replay_buffer.get("max_is_weight", 10.0)

        try:
            if self.config.trainer.resume_checkpoint:
                actor_local_path = os.path.join(
                    self.config.trainer.default_local_dir,
                    "actor",
                )
                if os.path.exists(actor_local_path):
                    latest_checkpint = find_latest_checkpoint(actor_local_path)
                    if latest_checkpint is not None:
                        global_states = torch.load(os.path.join(latest_checkpint, "global_states.pth"))
                        restore_rng_states(global_states["rng_states"])
                        print("Resuming to {}".format(latest_checkpint))
                    else:
                        global_states = {}
                else:
                    global_states = {}
            else:
                global_states = {}

            self.global_steps = global_states.get("global_steps", 0)

            # perform validation before training
            if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = await self.async_validate()
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                if self.config.trainer.get("val_only", False):
                    await close_global_session()
                    return

            self.global_steps += 1
            for epoch in range(self.config.trainer.total_epochs):
                if epoch < global_states.get("epochs", -1):
                    continue

                self.epochs = epoch
                if "consumed_samples" in global_states:
                    self.consumed_samples = global_states.get("consumed_samples")
                    global_states.pop("consumed_samples")
                else:
                    self.consumed_samples = 0

                self.train_dataloader.sampler.set_epoch(
                    self.epochs,
                    consumed_samples=self.consumed_samples,
                )
                for batch_dict in self.train_dataloader:
                    metrics = {}
                    timing_raw = {}

                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    self.consumed_samples += len(batch)

                    gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                    with _timer("step", timing_raw):
                        with _timer("gen", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        batch = batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                            interleave=True,
                        )
                        batch = batch.union(gen_batch_output)

                        self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with _timer("async_computes", timing_raw):
                            tasks = [
                                asyncio.create_task(asyncio.to_thread(self.actor_rollout_wg.compute_log_prob, batch)),
                                asyncio.create_task(self.reward_fn(batch)),
                            ]
                            if self.use_reference_policy:
                                tasks.append(
                                    asyncio.create_task(
                                        asyncio.to_thread(self.ref_policy_wg.compute_ref_log_prob, batch)
                                    )
                                )
                            if self.use_critic:
                                tasks.append(
                                    asyncio.create_task(asyncio.to_thread(self.critic_wg.compute_values, batch))
                                )

                            results = await asyncio.gather(*tasks)

                            old_log_prob = results[0]
                            batch = batch.union(old_log_prob)
                            reward_tensor = results[1]
                            index = 2
                            if self.use_reference_policy:
                                ref_log_prob = results[index]
                                batch = batch.union(ref_log_prob)
                                index += 1
                            if self.use_critic:
                                critic_values = results[index]
                                batch = batch.union(critic_values)

                        with _timer("rm_score", timing_raw):
                            batch = batch.union(reward_tensor)
                            if self.use_rm and (
                                self.global_steps >= self.config.reward_model.policy_warmup_step
                                or (
                                    "critic" in self.config.algorithm.adv_estimator
                                    and self.global_steps >= self.config.reward_model.critic_warmup_step
                                )
                            ):
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                        with _timer("adv", timing_raw):
                            if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch,
                                    kl_ctrl=self.kl_ctrl,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            add_process_reward = (
                                self.config.reward_model.use_potential_reward
                                and self.global_steps >= self.config.reward_model.policy_warmup_step
                            )
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                add_process_reward=add_process_reward,
                            )

                        if self.use_critic and self.global_steps >= self.config.reward_model.critic_warmup_step:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        if use_replay_buffer:
                            with _timer("replay_buffer", timing_raw):
                                # 1) 加样本到 buffer
                                rewards = batch.batch.get("token_level_scores", None)
                                if rewards is not None:
                                    for i in range(len(batch)):
                                        sample = batch.select_by_indices([i])
                                        self.replay_buffer.add(sample, sample.batch["old_log_probs"][0])

                                # 2) 判断是否触发蒸馏
                                current_reward_mean = metrics.get("critic/score/mean", 0)
                                if self.replay_buffer.should_distill(self.global_steps, current_reward_mean):
                                    print(
                                        f"Distillation triggered at step {self.global_steps}, current reward mean: {current_reward_mean:.4f}"
                                    )

                                    # 使用锁保护整个蒸馏流程
                                    async with _analyzer_lock:
                                        try:
                                            print("Starting distillation process...")
                                            # 一次性拉取 N 个样本作为多个microbatch
                                            replay_batches = self.replay_buffer.sample(
                                                min(self.off_policy_batch_size, len(self.replay_buffer))
                                            )

                                            print(
                                                f"Sampling completed: got {len(replay_batches)} microbatches, total {sum(len(batch) for batch in replay_batches)} samples"
                                            )

                                            if replay_batches:
                                                all_metrics = {}
                                                total_samples = 0

                                                # 添加任务列表以便于等待所有任务完成
                                                pending_tasks = []

                                                # 对每个microbatch分别进行处理
                                                for batch_idx, micro_batch in enumerate(replay_batches):
                                                    if not micro_batch:
                                                        continue

                                                    micro_size = len(micro_batch)
                                                    total_samples += micro_size
                                                    print(
                                                        f"Processing microbatch {batch_idx+1}/{len(replay_batches)}, containing {micro_size} samples"
                                                    )

                                                    # 处理边缘情况
                                                    if micro_size < 2:
                                                        micro_batch = micro_batch * 2
                                                        print(
                                                            f"Warning: microbatch has only 1 sample, duplicating to 2"
                                                        )

                                                    # 确保样本数为偶数
                                                    if len(micro_batch) % 2 == 1:
                                                        micro_batch.append(micro_batch[0])
                                                        print(
                                                            f"Warning: microbatch has odd number of samples, adding duplicate to make it even ({len(micro_batch)})"
                                                        )

                                                    # 创建当前microbatch的DataProto
                                                    replay_batch = DataProto.from_list(micro_batch)

                                                    # 添加5秒超时控制，防止卡死
                                                    print(f"Computing log_prob for microbatch {batch_idx+1}...")
                                                    try:
                                                        # 使用带超时的异步调用
                                                        log_probs_task = asyncio.create_task(
                                                            asyncio.wait_for(
                                                                asyncio.to_thread(
                                                                    self.actor_rollout_wg.compute_log_prob, replay_batch
                                                                ),
                                                                timeout=60.0,  # 设置更长的超时时间
                                                            )
                                                        )
                                                        # 等待当前任务完成后再处理下一个批次
                                                        log_probs_data = await log_probs_task

                                                        # 计算IS权重并处理数据
                                                        from verl.trainer.ppo.core_algos import (
                                                            compute_importance_sampling_weights,
                                                        )

                                                        log_prob = log_probs_data.batch["old_log_probs"]
                                                        log_prob_old = replay_batch.batch["log_prob_old"]
                                                        eos_mask = replay_batch.batch["attention_mask"][
                                                            :, -log_prob.shape[1] :
                                                        ]
                                                        advantages = replay_batch.batch["advantages"]

                                                        # 计算IS权重
                                                        is_weights = compute_importance_sampling_weights(
                                                            log_prob=log_prob,
                                                            log_prob_old=log_prob_old,
                                                            eos_mask=eos_mask,
                                                            max_weight=self.max_is_weight,
                                                        )

                                                        is_weights_cpu = is_weights.detach().cpu()

                                                        # 设置元数据
                                                        replay_batch.meta_info["is_weights"] = is_weights_cpu
                                                        replay_batch.meta_info["off_policy_weight"] = (
                                                            self.off_policy_weight
                                                        )
                                                        replay_batch.meta_info["global_token_num"] = torch.sum(
                                                            replay_batch.batch["attention_mask"], dim=-1
                                                        ).tolist()

                                                        # 执行单个microbatch的离线更新
                                                        print(
                                                            f"Executing off-policy update for microbatch {batch_idx+1}..."
                                                        )
                                                        # 等待当前更新完成
                                                        off_policy_output = await asyncio.wait_for(
                                                            asyncio.to_thread(
                                                                self.actor_rollout_wg.update_actor_off_policy,
                                                                replay_batch,
                                                            ),
                                                            timeout=60.0,
                                                        )

                                                        # 处理指标
                                                        if isinstance(off_policy_output, dict):
                                                            batch_metrics = off_policy_output
                                                        elif (
                                                            hasattr(off_policy_output, "meta_info")
                                                            and "metrics" in off_policy_output.meta_info
                                                        ):
                                                            batch_metrics = off_policy_output.meta_info["metrics"]
                                                        else:
                                                            batch_metrics = {}

                                                        if "off_policy_empty" not in batch_metrics:
                                                            for k, v in batch_metrics.items():
                                                                if k not in all_metrics:
                                                                    all_metrics[k] = []
                                                                all_metrics[k].extend(v if isinstance(v, list) else [v])
                                                        else:
                                                            print(f"Warning: microbatch {batch_idx+1} update is empty")

                                                        # 强制执行一次垃圾回收
                                                        import gc

                                                        gc.collect()
                                                        torch.cuda.empty_cache()

                                                    except asyncio.TimeoutError:
                                                        print(
                                                            f"Warning: microbatch {batch_idx+1} processing timeout, skipping"
                                                        )
                                                        # 尝试恢复Ray workers状态
                                                        try:
                                                            ray.experimental.force_reset()
                                                            print("Attempted to reset Ray resources")
                                                        except Exception as reset_err:
                                                            print(f"Failed to reset Ray resources: {reset_err}")
                                                        continue
                                                    except Exception as e:
                                                        print(f"Error processing microbatch {batch_idx+1}: {str(e)}")
                                                        import traceback

                                                        print(traceback.format_exc())
                                                        continue

                                                    # 每个microbatch处理后暂停一小段时间，给Ray资源释放的机会
                                                    await asyncio.sleep(0.5)

                                                # 汇总所有microbatch的指标
                                                if all_metrics:
                                                    print(
                                                        f"Off-policy update completed: processed {total_samples} samples, aggregating metrics..."
                                                    )
                                                    off_policy_metrics = reduce_metrics(all_metrics)
                                                    for k, v in off_policy_metrics.items():
                                                        metrics[f"off_policy_{k}"] = v

                                                    # 更新 buffer 蒸馏指标
                                                    reward_improvement = (
                                                        metrics.get("critic/rewards/mean", 0) - current_reward_mean
                                                    )
                                                    self.replay_buffer.update_distill_metrics(
                                                        self.global_steps, reward_improvement
                                                    )
                                                    distill_stats = self.replay_buffer.get_distill_stats()
                                                    metrics.update(
                                                        {f"distill_{k}": v for k, v in distill_stats.items()}
                                                    )
                                                    metrics.update(
                                                        {
                                                            "replay_buffer/samples_used": total_samples,
                                                            "replay_buffer/buffer_size": len(self.replay_buffer),
                                                            "replay_buffer/reward_improvement": reward_improvement,
                                                        }
                                                    )
                                                else:
                                                    print("Warning: all microbatch updates are empty, no valid metrics")
                                        except Exception as e:
                                            print(f"Exception during distillation process: {str(e)}")
                                            import traceback

                                            print(traceback.format_exc())
                                        finally:
                                            # 蒸馏结束，确保资源释放
                                            print("Distillation process ended, ensuring resource cleanup")
                                            torch.cuda.empty_cache()
                                            await asyncio.sleep(1.0)  # 给系统一点时间来释放资源

                                    # 蒸馏结束后等待一段时间再继续训练，确保资源释放
                                    print("Distillation flow ended, brief wait before continuing training")
                                    await asyncio.sleep(2.0)

                        # 无论是否执行蒸馏，都记录当前的缓冲区状态
                        if use_replay_buffer:
                            metrics["replay_buffer/current_size"] = len(self.replay_buffer)
                            metrics["replay_buffer/capacity"] = self.replay_buffer.capacity
                            metrics["replay_buffer/fullness"] = len(self.replay_buffer) / self.replay_buffer.capacity

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and self.global_steps % self.config.trainer.test_freq == 0
                        ):
                            with _timer("testing", timing_raw):
                                val_metrics: dict = await self.async_validate()
                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                    logger.log(data=metrics, step=self.global_steps)

                    self.global_steps += 1

                    if self.global_steps >= self.total_training_steps:
                        if self.val_reward_fn is not None:
                            val_metrics = await self.async_validate()
                            pprint(f"Final validation metrics: {val_metrics}")
                            logger.log(data=val_metrics, step=self.global_steps)

                        # 训练结束时关闭会话
                        await close_global_session()
                        return
        except Exception as e:
            # 确保异常退出时也能关闭会话
            print(f"Exception during training: {str(e)}")
            await close_global_session()
            raise

    def fit(self):
        """
        For debug only
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        if self.config.trainer.resume_checkpoint:
            actor_local_path = os.path.join(
                self.config.trainer.default_local_dir,
                "actor",
            )
            if os.path.exists(actor_local_path):
                latest_checkpint = find_latest_checkpoint(actor_local_path)
                if latest_checkpint is not None:
                    global_states = torch.load(os.path.join(latest_checkpint, "global_states.pth"))
                    restore_rng_states(global_states["rng_states"])
                    print("Resuming to {}".format(latest_checkpint))
                else:
                    global_states = {}
        else:
            global_states = {}

        self.global_steps = global_states.get("global_steps", 0)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1
        for epoch in range(self.config.trainer.total_epochs):
            if epoch < global_states.get("epochs", -1):
                continue

            self.epochs = epoch
            if "consumed_samples" in global_states:
                self.consumed_samples = global_states.get("consumed_samples") // self.config.actor_rollout_ref.rollout.n
                global_states.pop("consumed_samples")
            else:
                self.consumed_samples = 0

            self.train_dataloader.sampler.set_epoch(
                self.epochs,
                consumed_samples=self.consumed_samples,
            )
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.

                        # we first compute rule-based scores
                        reward_tensor = self.reward_fn(batch)
                        batch = batch.union(reward_tensor)

                        # We then compute the scores using reward model.
                        if self.use_rm and (
                            self.global_steps >= self.config.reward_model.policy_warmup_step
                            or "critic" in self.config.algorithm.adv_estimator
                        ):
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        add_process_reward = (
                            self.config.reward_model.use_potential_reward
                            and self.global_steps >= self.config.reward_model.policy_warmup_step
                        )
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            add_process_reward=add_process_reward,
                        )

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # collect metrics
                        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

    # TODO (jiabao) 02/05: currently this pipeline does not improve speed, need to investigate
    async def pipeline_async_validate(self):
        reward_tensor_lst = asyncio.Queue()
        data_source_lst = asyncio.Queue()
        all_results = asyncio.Queue()

        comm_queue = asyncio.Queue()

        async def gen_producer():
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)
                if (
                    self.config.reward_model.enable
                    and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
                ):
                    await comm_queue.put({})

                test_gen_batch = test_batch.pop(["input_ids", "attention_mask", "position_ids"])
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": False,
                    "validate": True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print("validation generation end")

                test_batch = test_batch.union(test_output_gen_batch)

                # return generated result
                await comm_queue.put(test_batch)

            await comm_queue.put(None)  # Stop signal

        async def reward_consumer():
            while True:
                test_batch = await comm_queue.get()
                if test_batch is None:
                    await comm_queue.put(None)  # Stop signal
                    break

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                val_result = await self.val_reward_fn(test_batch)
                if isinstance(val_result, DataProto):
                    reward_tensor = val_result.batch["token_level_scores"]
                else:
                    reward_tensor = val_result

                    # Cache results
                for i in range(len(test_batch)):
                    data_item = test_batch[i]
                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    response_ids = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]
                    await all_results.put(
                        {
                            "problem": self.tokenizer.decode(valid_prompt_ids),
                            "response": self.tokenizer.decode(valid_response_ids),
                            "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                        }
                    )

                await reward_tensor_lst.put(reward_tensor)
                await data_source_lst.put(
                    test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
                )

        # We do a 2-stage async pipeline here
        await asyncio.gather(gen_producer(), reward_consumer())

        # Collect result list
        final_results = []
        while True:
            batch = await reward_tensor_lst.get()
            if batch is None:
                break
            final_results.extend(batch)
