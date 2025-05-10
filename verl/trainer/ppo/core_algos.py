# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from typing import Optional

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def calculate_return(rewards: torch.Tensor):
    cum_sum = torch.cumsum(rewards, dim=-1)
    return rewards + cum_sum[:, -1:] - cum_sum


def compute_grpo_process_advantage(token_level_rewards: torch.Tensor,
                                   step_indicator: torch.Tensor,
                                   avg_pr_per_step: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    returns = calculate_return(token_level_rewards)
    step_indicator = torch.cat([torch.zeros_like(step_indicator[:, :1]), step_indicator], dim=-1)
    step_indicator = step_indicator[:, :-1]
    num_steps = step_indicator.sum(dim=-1, keepdim=True)

    id2mean = defaultdict(list)

    bsz = returns.shape[0]
    for i in range(bsz):
        id2mean[index[i]].append(avg_pr_per_step[i])
    id2mean = {k: torch.stack(v).mean() for k, v in id2mean.items()}
    mean = torch.stack([id2mean[index[i]] for i in range(bsz)])
    num_remain_steps = num_steps - torch.cumsum(step_indicator, dim=-1)
    advantages = returns - mean.view(-1, 1) * num_remain_steps
    advantages *= eos_mask

    return advantages, advantages


def compute_rloo_advantage(outcome_rewards: torch.Tensor,
                           eos_mask: torch.Tensor,
                           index: torch.Tensor,
                           adv_estimator: str,
                           process_rewards: Optional[torch.Tensor] = None,
                           step_indicator: Optional[torch.Tensor] = None,
                           avg_pr_per_step: Optional[torch.Tensor] = None):
    """
    Compute advantage for RLOO, operating on outcome reward and process reward
    Args:
        outcome_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    def calculate_loo_baseline(scores, adaptive=False):
        id2score = defaultdict(list)
        n_rollouts = {}
        id2sum = {}

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            assert len(id2score[idx]) > 1, "Number of samples has to be greater than 1 for RLOO"
            scores_i = torch.stack(id2score[idx])
            id2sum[idx] = scores_i.sum(dim=0)
            if adaptive:
                n_rollouts[idx] = (scores_i != 0).sum(dim=0)
            else:
                n_rollouts = len(id2score[idx])
        r_sum = torch.stack([id2sum[index[i]] for i in range(bsz)])
        if adaptive:
            n_rollouts = torch.stack([n_rollouts[index[i]] for i in range(bsz)])
        return (r_sum - scores) / (n_rollouts - 1 + 1e-8)
    
    # Outcome advantages
    response_length = outcome_rewards.shape[-1]
    returns = outcome_rewards.sum(dim=-1)
    baseline = calculate_loo_baseline(returns)
    advantages = returns - baseline
    advantages = advantages.unsqueeze(-1).tile([1, response_length]) * eos_mask
    returns = returns.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    # Process advantages
    if process_rewards is not None:
        pr_returns = calculate_return(process_rewards)
        if "prime" in adv_estimator:
            num_steps = step_indicator.sum(dim=-1, keepdim=True)
            step_indicator = torch.cat([torch.zeros_like(step_indicator[:, :1]), step_indicator], dim=-1)
            step_indicator = step_indicator[:, :-1]
            baseline = calculate_loo_baseline(avg_pr_per_step)
            num_remain_steps = num_steps - torch.cumsum(step_indicator, dim=-1)
            pr_advantages = pr_returns - baseline.view(-1, 1) * num_remain_steps
        else:
            baseline = calculate_loo_baseline(pr_returns, adaptive=True)
            pr_advantages = pr_returns - baseline
        
        pr_advantages *= eos_mask
        advantages += pr_advantages
        returns += pr_returns

    return advantages, returns


def compute_step_pg_advantage_return(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor,
        eos_mask: torch.Tensor,
        gamma: torch.Tensor,
        lam: torch.Tensor,
        step_indicator: torch.Tensor):
    """
    REINFORCE with critic as baseline
    """
    advantages, returns = compute_gae_advantage_return(
        token_level_rewards=token_level_rewards,
        values=values,
        eos_mask=eos_mask,
        gamma=gamma,
        lam=lam,
    )
    # set advantages within each step the same
    for i in range(advantages.shape[0]): # batch size
        boundaries = torch.cat([torch.tensor([0]), torch.nonzero(step_indicator[i]).squeeze(-1) + 1])
        for j in range(len(boundaries) - 1):
            advantages[i, boundaries[j]:boundaries[j+1]] = advantages[i, boundaries[j]]
            returns[i, boundaries[j]:boundaries[j+1]] = returns[i, boundaries[j]]
    # TODO: (yujian) check if we need whiten here
    advantages *= eos_mask
    returns *= eos_mask
    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def compute_step_value_loss(vpreds, returns, values, step_indicator, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    # TODO: (yujian) check if this is correct when step size is 1
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    loss_sum, clipfrac_sum, count = 0, 0, 0
    for i in range(vpreds.shape[0]): # batch size
        boundaries = torch.cat(
            [torch.tensor([0], device=step_indicator.device),
             torch.nonzero(step_indicator[i]).squeeze(-1) + 1])
        vpred_i = vpreds[i, boundaries[:-1]]
        vpredclipped_i = vpredclipped[i, boundaries[:-1]]
        returns_i = returns[i, boundaries[:-1]]
        vf_losses1 = (vpred_i - returns_i)**2
        vf_losses2 = (vpredclipped_i - returns_i)**2
        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2)
        vf_clipfrac = torch.gt(vf_losses2, vf_losses1).float()
        loss_sum += vf_loss.sum()
        clipfrac_sum += vf_clipfrac.sum()
        count += vf_loss.numel()
    vf_loss = loss_sum / count
    vf_clipfrac = clipfrac_sum / count
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_importance_sampling_weights(
    log_prob: torch.Tensor, 
    log_prob_old: torch.Tensor, 
    eos_mask: torch.Tensor,
    max_weight: float = 10.0
) -> torch.Tensor:
    """Compute importance sampling weights with clipping.
    
    Args:
        log_prob: Log probability under current policy
        log_prob_old: Log probability under behavior policy
        eos_mask: Mask for valid tokens
        max_weight: Maximum IS weight to prevent instability
        
    Returns:
        is_weights: Importance sampling weights
    """
    # Compute ratio of probabilities
    ratio = torch.exp(log_prob - log_prob_old)
    
    # Clip weights to prevent instability
    clipped_ratio = torch.clamp(ratio, max=max_weight)
    
    # Apply mask
    is_weights = clipped_ratio * eos_mask
    
    return is_weights

def compute_off_policy_loss(
    log_prob: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    max_is_weight: float = 10.0
) -> torch.Tensor:
    """Compute off-policy loss with importance sampling correction.
    
    Args:
        log_prob: Log probability under current policy
        log_prob_old: Log probability under behavior policy
        advantages: Advantage estimates
        eos_mask: Mask for valid tokens
        max_is_weight: Maximum IS weight to prevent instability
        
    Returns:
        off_policy_loss: Off-policy loss with IS correction
    """
    # Compute IS weights
    is_weights = compute_importance_sampling_weights(
        log_prob, log_prob_old, eos_mask, max_is_weight
    )
    
    # Compute off-policy loss
    off_policy_losses = -is_weights * advantages * log_prob
    
    # Average over valid tokens
    off_policy_loss = verl_F.masked_mean(off_policy_losses, eos_mask)
    
    return off_policy_loss
