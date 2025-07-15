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
FSDP PPO Trainer with Trail replay buffer integration.
This trainer extends the Ray PPO trainer with off-policy learning capabilities.
"""
import uuid
import gc
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import list_of_dict_to_dict_of_list
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    compute_reward,
    compute_reward_async,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .replay_buffer import ReplayBuffer
from .data_proto import TrailDataProto


class TrailPPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer with Trail replay buffer integration.

    This trainer extends the Ray PPO trainer with off-policy learning capabilities,
    managing actor rollouts, critic training, and reward computation with replay buffer support.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize distributed PPO trainer with Trail replay buffer integration.
        """
        super().__init__(*args, **kwargs)

        # Initialize replay buffer if enabled
        self.use_replay_buffer = self.config.algorithm.get("use_replay_buffer", False)
        if self.use_replay_buffer:
            replay_config = self.config.algorithm.replay_buffer
            self.replay_buffer = ReplayBuffer(
                capacity=replay_config.get("capacity", 10000),
                top_percent=replay_config.get("top_percent", 0.05),
                min_reward=replay_config.get("min_reward", 0.85),
                microbatch_size=replay_config.get("microbatch_size", 8),
                fixed_distill_interval=replay_config.get("fixed_distill_interval", 10),
            )
            print(f"Initialized replay buffer with capacity {self.replay_buffer.capacity}")
        else:
            self.replay_buffer = None

    def _perform_distillation(self, timing_raw: dict):
        """Perform distillation using replay buffer samples."""
        if not self.use_replay_buffer or not self.replay_buffer.should_distill(self.global_steps):
            return {}

        with marked_timer("distillation", timing_raw, color="purple"):
            print(f"--- Starting distillation at step {self.global_steps} ---")

            replay_config = self.config.algorithm.replay_buffer
            batch_size = replay_config.get("batch_size", 64)
            replay_samples_list = self.replay_buffer.sample(batch_size)

            if not replay_samples_list:
                print("No samples available in replay buffer for distillation.")
                return {}

            distill_metrics = {}
            initial_actor_loss = 0
            final_actor_loss = 0
            num_microbatches = len(replay_samples_list)

            for i, microbatch_samples in enumerate(replay_samples_list):
                if not microbatch_samples:
                    continue

                microbatch = TrailDataProto.from_list(microbatch_samples)
                microbatch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(microbatch.batch))], dtype=object
                )

                if "response_mask" not in microbatch.batch:
                    microbatch.batch["response_mask"] = compute_response_mask(microbatch)

                # Recompute log probabilities for importance sampling
                current_log_prob_proto = self.actor_rollout_wg.compute_log_prob(microbatch)
                # Rename to avoid key collision in union. This is the log_prob from the current policy.
                current_log_prob_proto.batch["new_log_probs"] = current_log_prob_proto.batch.pop("old_log_probs")
                microbatch = microbatch.union(current_log_prob_proto)

                # Compute importance sampling weights
                # log_prob_old is from the behavior policy (when data was generated)
                behavior_log_probs = microbatch.batch["log_prob_old"]
                # new_log_probs is from the current policy
                current_log_probs = microbatch.batch["new_log_probs"]
                max_is_weight = replay_config.get("max_is_weight", 10.0)
                is_weights = torch.exp(current_log_probs.sum(-1) - behavior_log_probs.sum(-1))
                is_weights = torch.clamp(is_weights, max=max_is_weight)
                microbatch.batch["is_weights"] = is_weights

                if self.use_reference_policy:
                    # ref_log_prob should already be in the replay buffer samples.
                    # Only recompute if it's missing for some reason.
                    if "ref_log_prob" not in microbatch.batch:
                        ref_log_prob = (
                            self.actor_rollout_wg.compute_ref_log_prob(microbatch)
                            if self.ref_in_actor
                            else self.ref_policy_wg.compute_ref_log_prob(microbatch)
                        )
                        microbatch = microbatch.union(ref_log_prob)

                if self.config.algorithm.use_kl_in_reward:
                    microbatch, kl_metrics = apply_kl_penalty(
                        microbatch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    distill_metrics.update({f"distill/kl_{k}": v for k, v in kl_metrics.items()})
                else:
                    microbatch.batch["token_level_rewards"] = microbatch.batch["token_level_scores"]

                if self.use_critic:
                    values = self.critic_wg.compute_values(microbatch)
                    microbatch = microbatch.union(values)

                microbatch = compute_advantage(
                    microbatch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    config=self.config.algorithm,
                )

                if self.use_critic:
                    critic_output = self.critic_wg.update_critic(microbatch)
                    critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    distill_metrics.update({f"distill/critic_{k}": v for k, v in critic_metrics.items()})

                microbatch.meta_info["off_policy_weight"] = replay_config.get("off_policy_weight", 0.5)
                actor_output = self.actor_rollout_wg.update_actor(microbatch)
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                distill_metrics.update({f"distill/actor_{k}": v for k, v in actor_metrics.items()})

                if i == 0:
                    initial_actor_loss = actor_metrics.get("actor/loss", 0)
                if i == num_microbatches - 1:
                    final_actor_loss = actor_metrics.get("actor/loss", 0)

            reward_improvement = initial_actor_loss - final_actor_loss
            self.replay_buffer.update_distill_metrics(self.global_steps, reward_improvement)
            distill_metrics.update(self.replay_buffer.get_distill_stats())
            print(f"--- Finished distillation at step {self.global_steps} ---")
            return distill_metrics

    def fit(self):
        """
        The training loop of PPO with Trail replay buffer integration.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n=== Starting Epoch {epoch} ===")
            
            # Perform garbage collection at the start of each epoch
            gc.collect()
            
            for batch_dict in self.train_dataloader:
                try:
                    metrics = {}
                    timing_raw = {}

                    do_profile = (
                        self.global_steps in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    with marked_timer("start_profile", timing_raw):
                        if do_profile:
                            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                            if self.use_reference_policy:
                                self.ref_policy_wg.start_profile()
                            if self.use_critic:
                                self.critic_wg.start_profile()
                            if self.use_rm:
                                self.rm_wg.start_profile()

                    batch_raw: DataProto = DataProto.from_single_dict(batch_dict)
                    batch = TrailDataProto(
                        batch=batch_raw.batch,
                        non_tensor_batch=batch_raw.non_tensor_batch,
                        meta_info=batch_raw.meta_info,
                    )
                    
                    # Clean up temporary variables
                    del batch_raw
                    
                    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                    non_tensor_batch_keys_to_pop = [
                        "raw_prompt_ids",
                        "raw_prompt",
                        "tools_kwargs",
                        "interaction_kwargs",
                        "index",
                        "agent_name",
                    ]
                    non_tensor_batch_keys_to_pop = [k for k in non_tensor_batch_keys_to_pop if k in batch.non_tensor_batch]
                    if "multi_modal_data" in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append("multi_modal_data")

                    gen_batch = batch.pop(
                        batch_keys=batch_keys_to_pop,
                        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    )
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    is_last_step = self.global_steps >= self.total_training_steps

                    with marked_timer("step", timing_raw):
                        with marked_timer("gen", timing_raw, color="red"):
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        
                        # Clean up temporary variables
                        del gen_batch, gen_batch_output

                        if "response_mask" not in batch.batch:
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)
                                del reward_tensor
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            del old_log_prob, entropys, response_masks

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw, color="olive"):
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                    if self.ref_in_actor
                                    else self.ref_policy_wg.compute_ref_log_prob(batch)
                                )
                                batch = batch.union(ref_log_prob)
                                del ref_log_prob

                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)
                                del values

                        with marked_timer("adv", timing_raw, color="brown"):
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            del reward_tensor

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                                config=self.config.algorithm,
                            )

                        # Add high-quality samples to replay buffer
                        if self.use_replay_buffer:
                            with marked_timer("add_to_buffer", timing_raw, color="orange"):
                                for i in range(len(batch.batch)):
                                    sample_data = batch[i]
                                    # Create a deep copy and immediately delete the reference
                                    sample_copy = deepcopy(sample_data)
                                    self.replay_buffer.add(sample_copy, sample_data.batch["old_log_probs"])
                                    del sample_data, sample_copy

                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                            del critic_output

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                            del actor_output

                        # Perform distillation from replay buffer
                        distill_metrics = self._perform_distillation(timing_raw)
                        metrics.update(distill_metrics)
                        del distill_metrics

                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        ):
                            with marked_timer("testing", timing_raw, color="green"):
                                val_metrics = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                    with marked_timer("stop_profile", timing_raw):
                        if do_profile:
                            self.actor_rollout_wg.stop_profile()
                            if self.use_reference_policy:
                                self.ref_policy_wg.stop_profile()
                            if self.use_critic:
                                self.critic_wg.stop_profile()
                            if self.use_rm:
                                self.rm_wg.stop_profile()

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1

                    # Clean up large objects related to batch
                    del batch, metrics, timing_raw
                    
                    # Force garbage collection every 10 steps
                    if self.global_steps % 10 == 0:
                        gc.collect()
                        if self.use_replay_buffer and hasattr(self.replay_buffer, 'clear_buffer'):
                            # If memory usage is high, clean up part of the buffer
                            import psutil
                            memory_percent = psutil.virtual_memory().percent
                            if memory_percent > 85:  # Clean up if memory usage exceeds 85%
                                print(f"High memory usage detected ({memory_percent:.1f}%), performing cleanup")
                                # Optionally clean up part of the buffer instead of all
                                if len(self.replay_buffer.buffer) > self.replay_buffer.capacity // 2:
                                    # Retain the latest half of the data
                                    new_buffer = deque(list(self.replay_buffer.buffer)[-self.replay_buffer.capacity//2:], 
                                                     maxlen=self.replay_buffer.capacity)
                                    self.replay_buffer.buffer = new_buffer
                                gc.collect()

                    if is_last_step:
                        if last_val_metrics:
                            print(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        
                        # Final cleanup
                        if self.use_replay_buffer:
                            self.replay_buffer.clear_buffer()
                        gc.collect()
                        return

                    if hasattr(self.train_dataset, "on_batch_end"):
                        self.train_dataset.on_batch_end(batch=batch)

                except Exception as e:
                    print(f"Error in training step {self.global_steps}: {e}")
                    # Clean up memory on error
                    gc.collect()
                    raise e
                    
            # Perform memory cleanup at the end of each epoch
            print(f"=== Completed Epoch {epoch}, performing memory cleanup ===")
            gc.collect()
            
            # Clean up old data in replay buffer
            if self.use_replay_buffer and epoch > 0:
                # Retain the latest 80% of data after each epoch
                if len(self.replay_buffer.buffer) > 100:
                    keep_size = int(len(self.replay_buffer.buffer) * 0.8)
                    new_buffer = deque(list(self.replay_buffer.buffer)[-keep_size:], 
                                     maxlen=self.replay_buffer.capacity)
                    self.replay_buffer.buffer = new_buffer
                    print(f"Cleaned replay buffer, kept {keep_size} samples")
                    gc.collect()
