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

import torch
import numpy as np
import random
import copy
import heapq
import time
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from verl import DataProto


class ReplayBuffer:
    """Replay buffer for off-policy learning with FIFO replacement and random sampling."""

    def __init__(
        self,
        capacity: int = 10000,
        top_percent: float = 0.05,
        min_reward: float = 0.85,
        min_distill_interval: int = 2,
        max_distill_interval: int = 10,
        distill_threshold_ratio: float = 0.3,
        lambda_decay: float = 5.0,  # 控制衰减速度的参数
        microbatch_size: int = 8,  # 添加microbatch_size参数
        fixed_distill_interval: int = 10,  # 添加fixed_distill_interval参数
    ):
        """Initialize a replay buffer with FIFO replacement and random sampling.

        Args:
            capacity: Maximum size of the buffer
            top_percent: Only samples in the top X% of recent rewards will be added to buffer
            min_reward: Minimum reward threshold for samples to be added to buffer
            min_distill_interval: Minimum steps between distillations
            max_distill_interval: Maximum steps between distillations
            distill_threshold_ratio: Buffer fullness ratio to trigger distillation (0.3 = 30% full)
            lambda_decay: Control parameter for exponential decay based on reward convergence
            microbatch_size: Batch size to use during distillation to avoid OOM
            fixed_distill_interval: Fixed number of steps between distillations
        """
        self.capacity = capacity
        self.top_percent = top_percent  # 重新启用: 只有奖励值在前X%的样本才会被添加
        self.min_reward = min_reward
        self.min_distill_interval = min_distill_interval
        self.max_distill_interval = max_distill_interval
        self.distill_threshold_ratio = distill_threshold_ratio
        self.microbatch_size = microbatch_size
        self.lambda_decay = lambda_decay

        # 使用队列存储数据，实现FIFO
        self.buffer = deque(maxlen=capacity)

        # 保存所有样本的奖励记录(无论是否添加到buffer)
        self.all_rewards = deque(maxlen=capacity)  # 记录最近capacity条样本的奖励
        self.reward_window_size = capacity

        # 自动蒸馏的追踪变量
        self.last_distill_step = 0
        self.steps_since_last_distill = 0
        self.samples_added_since_last_distill = 0
        self.reward_improvement = []  # 存储每次蒸馏后的奖励改进
        self.recent_distill_history = deque(maxlen=5)  # 最近5次蒸馏的结果

        # 使用传入的参数设置固定蒸馏间隔
        self.fixed_distill_interval = fixed_distill_interval

        # 保留这些变量供指标记录使用，但不再用于动态计算间隔
        self.avg_rewards = []
        self.reward_convergence_factor = 0.0
        self.target_distill_interval = self.fixed_distill_interval

    def add(self, data_item: DataProto, log_prob_old: torch.Tensor):
        """添加样本到缓冲区（FIFO策略）
        样本需要同时满足两个条件才会被添加:
        1. 奖励高于min_reward阈值
        2. 奖励在最近的capacity条样本的前top_percent百分比内

        Args:
            data_item: DataProto containing prompt, response, and reward
            log_prob_old: Log probability of the response under the behavior policy
        """
        # 提取奖励值
        reward = data_item.batch.get("token_level_scores", None)
        if reward is None:
            reward = data_item.batch.get("answer_correctness", None)

        added_to_buffer = False
        message = ""

        if reward is not None:
            # 计算样本总奖励
            sample_reward = torch.sum(reward).item()

            # 将当前奖励添加到历史记录中(无论是否添加到buffer)
            self.all_rewards.append(sample_reward)

            # 检查最低奖励阈值
            if sample_reward < self.min_reward:
                message = f"Sample rejected: reward {sample_reward:.4f} below minimum threshold {self.min_reward}"
                print(message)
                return False

            # 检查是否在前top_percent百分比内
            if len(self.all_rewards) >= 10:  # 至少有10个样本才开始比较
                # 计算奖励阈值(前top_percent百分比的最低奖励)
                sorted_rewards = sorted(self.all_rewards, reverse=True)  # 按奖励降序排序
                threshold_idx = max(0, int(len(sorted_rewards) * self.top_percent) - 1)
                reward_threshold = sorted_rewards[threshold_idx]

                # 如果当前样本奖励不在前top_percent内，拒绝添加
                if sample_reward < reward_threshold:
                    message = (
                        f"Sample rejected: reward {sample_reward:.4f} not in top {self.top_percent*100}% "
                        f"(threshold: {reward_threshold:.4f})"
                    )
                    print(message)
                    return False

            # 创建数据的深拷贝
            data_copy = data_item.select(deepcopy=True)

            # 确保log_prob_old具有正确的批次维度
            if (
                isinstance(log_prob_old, torch.Tensor)
                and log_prob_old.dim() > 0
                and log_prob_old.shape[0] != data_copy.batch.batch_size[0]
            ):
                log_prob_old = log_prob_old.unsqueeze(0) if log_prob_old.dim() == 1 else log_prob_old[:1]

            # 将log_prob_old添加到数据中
            data_copy.batch["log_prob_old"] = log_prob_old

            # FIFO策略: 将样本添加到队列末尾（如果容量已满，自动移除最旧的样本）
            self.buffer.append((sample_reward, data_copy))
            added_to_buffer = True
            message = f"Sample added: reward {sample_reward:.4f}, buffer size {len(self.buffer)}/{self.capacity}"

            # 如果缓冲区超出容量（通常不会发生，因为deque有maxlen），移除最早添加的样本
            if len(self.buffer) > self.capacity:
                removed = self.buffer.popleft()
                message += f" | Removed oldest sample with reward: {removed[0]:.4f}"

        # 只在成功添加到缓冲区时打印完整状态
        if added_to_buffer:
            print("==================================")
            print(message)
            print(f"current buffer len: {len(self.buffer)}")
            print(f"min reward threshold: {self.min_reward}")
            if len(self.all_rewards) >= 10:
                top_reward = sorted(self.all_rewards, reverse=True)[0]
                print(f"current top reward: {top_reward:.4f}")
                print(f"top {self.top_percent*100}% threshold: {reward_threshold:.4f}")
            print("==================================")

        return added_to_buffer

    def sample(self, batch_size: int) -> List[DataProto]:
        """从缓冲区中随机采样（不基于奖励权重）

        Args:
            batch_size: Number of samples to return

        Returns:
            List of DataProto objects, with total length up to batch_size, but each
            sub-list has at most microbatch_size elements
        """
        # 处理边缘情况：如果buffer为空，直接返回空列表
        if len(self.buffer) == 0:
            print("Warning: Buffer is empty, cannot sample")
            return []

        # 处理边缘情况：确保至少采样1个样本
        batch_size = max(1, batch_size)

        if len(self.buffer) <= batch_size:
            # 如果样本不足，返回所有可用样本并清空buffer
            samples = [item[1] for item in self.buffer]
            self.buffer.clear()

            # 处理只有一个样本的特殊情况
            if len(samples) == 1:
                # 对单个样本复制一次以确保至少有两个样本，避免批处理问题
                samples = samples * 2
                print(f"Warning: Only 1 sample available, duplicated to {len(samples)} samples")

            # 如果是奇数条样本，补充一个副本确保为偶数，有利于均匀分布到多个GPU
            if len(samples) % 2 == 1:
                samples.append(samples[0])
                print(f"Warning: Odd number of samples, added 1 duplicate to make even: {len(samples)}")

            # 按microbatch_size分组
            return [samples[i : i + self.microbatch_size] for i in range(0, len(samples), self.microbatch_size)]
        else:
            # 完全随机采样（不考虑奖励）
            indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

            # 如果采样结果数量为奇数，多采样一个样本确保为偶数
            if len(indices) % 2 == 1:
                remaining_indices = [i for i in range(len(self.buffer)) if i not in indices]
                if remaining_indices:  # 如果还有可用样本
                    extra_index = np.random.choice(remaining_indices)
                    indices = np.append(indices, extra_index)
                    print(f"Sampled odd {len(indices)-1} samples, added 1 extra to make even: {len(indices)}")
                else:  # 如果没有更多样本可用，复制一个已有的样本
                    indices = np.append(indices, indices[0])
                    print(f"Sampled odd {len(indices)-1} samples, duplicated 1 to make even: {len(indices)}")

            # 构建采样结果
            samples = []
            # 为了安全删除，我们需要按降序处理索引
            sorted_indices = sorted(indices, reverse=True)

            # 从buffer中提取并删除选中的样本
            buffer_list = list(self.buffer)  # 将deque转为列表以便按索引访问
            for idx in sorted_indices:
                samples.append(buffer_list[int(idx)][1])

            # 从buffer中删除已采样的样本
            # 创建新deque并过滤掉被采样的样本
            remaining = [item for i, item in enumerate(buffer_list) if i not in indices]
            self.buffer = deque(remaining, maxlen=self.capacity)

            print(f"Sampling completed: randomly sampled {len(samples)} samples")
            # 按microbatch_size分组返回
            return [samples[i : i + self.microbatch_size] for i in range(0, len(samples), self.microbatch_size)]

    def __len__(self) -> int:
        return len(self.buffer)

    def should_distill(self, current_step: int, reward_mean: float = None) -> bool:
        """Determine if distillation should be performed based on buffer state and training progress.

        Args:
            current_step: Current training step
            reward_mean: Current mean reward (optional, used for tracking improvement)

        Returns:
            bool: True if distillation should be performed
        """
        # 如果buffer为空，不进行蒸馏
        if len(self.buffer) == 0:
            return False

        # 计算自上次蒸馏以来的步数
        steps_since_last_distill = current_step - self.last_distill_step

        # 确保至少经过最小蒸馏间隔
        if steps_since_last_distill < self.min_distill_interval:
            return False

        # 固定间隔蒸馏：每固定步数执行一次蒸馏
        if steps_since_last_distill >= self.fixed_distill_interval:
            print(f"Reached fixed distillation interval {self.fixed_distill_interval} steps, triggering distillation")
            return True

        # 记录当前状态供调试
        if steps_since_last_distill % 5 == 0:  # 每5步记录一次，避免日志过多
            print(
                f"Distillation status: steps={steps_since_last_distill}, fixed_interval={self.fixed_distill_interval}, "
                f"fullness={len(self.buffer) / self.capacity:.2f}"
            )

        return False

    def update_distill_metrics(self, current_step: int, reward_improvement: float = 0.0):
        """Update distillation-related metrics after a distillation operation.

        Args:
            current_step: Current training step
            reward_improvement: Improvement in reward after distillation
        """
        # 更新蒸馏步骤追踪
        self.steps_since_last_distill = 0
        self.last_distill_step = current_step
        self.samples_added_since_last_distill = 0

        # 记录本次蒸馏的效果
        self.reward_improvement.append(reward_improvement)

        # 计算当前buffer中样本的平均奖励
        current_avg_reward = np.mean([r for r, _ in self.buffer]) if self.buffer else 0.0
        self.avg_rewards.append(current_avg_reward)

        # 目标蒸馏间隔始终为固定值
        self.target_distill_interval = self.fixed_distill_interval

        # 计算一些指标用于监控（但不再用于调整间隔）
        if len(self.avg_rewards) >= 2:
            reward_diff = abs(self.avg_rewards[-1] - self.avg_rewards[-2])
            # 仍然计算收敛因子，但只用于监控
            self.reward_convergence_factor = np.exp(-self.lambda_decay * reward_diff)
        else:
            reward_diff = 0
            self.reward_convergence_factor = 0.0

        # 记录本次蒸馏的信息
        self.recent_distill_history.append(
            {
                "step": current_step,
                "buffer_size": len(self.buffer),
                "reward_improvement": reward_improvement,
                "buffer_fullness": len(self.buffer) / self.capacity if self.capacity > 0 else 0,
                "avg_reward": current_avg_reward,
                "convergence_factor": self.reward_convergence_factor,
                "target_interval": self.fixed_distill_interval,
            }
        )

        # 打印蒸馏结果
        print(f"\n===== Distillation Execution Results =====")
        print(f"Step: {current_step}, Reward improvement: {reward_improvement:.4f}")
        if len(self.avg_rewards) >= 2:
            print(f"Reward change: {reward_diff:.4f}")
        print(f"Next distillation interval: {self.fixed_distill_interval} (fixed)")
        print(f"Buffer size: {len(self.buffer)}/{self.capacity}")
        print(f"Current threshold ratio: {self.distill_threshold_ratio:.2f}")

        # 动态调整蒸馏阈值 - 更智能的调整逻辑
        if len(self.reward_improvement) > 3:
            recent_improvements = self.reward_improvement[-3:]
            avg_improvement = np.mean(recent_improvements)

            # 根据平均改进幅度调整阈值
            if avg_improvement < 0.03:  # 效果很差
                # 大幅增加阈值，显著减少蒸馏频率
                self.distill_threshold_ratio = min(0.7, self.distill_threshold_ratio * 1.3)
                print(f"Poor distillation effect, increasing threshold to {self.distill_threshold_ratio:.2f}")
            elif avg_improvement < 0.05:  # 效果一般
                # 小幅增加阈值，适度减少蒸馏频率
                self.distill_threshold_ratio = min(0.6, self.distill_threshold_ratio * 1.15)
                print(f"Average distillation effect, increasing threshold to {self.distill_threshold_ratio:.2f}")
            elif avg_improvement > 0.15:  # 效果极好
                # 大幅降低阈值，显著增加蒸馏频率
                self.distill_threshold_ratio = max(0.1, self.distill_threshold_ratio * 0.75)
                print(f"Excellent distillation effect, decreasing threshold to {self.distill_threshold_ratio:.2f}")
            elif avg_improvement > 0.08:  # 效果良好
                # 小幅降低阈值，适度增加蒸馏频率
                self.distill_threshold_ratio = max(0.15, self.distill_threshold_ratio * 0.85)
                print(f"Good distillation effect, decreasing threshold to {self.distill_threshold_ratio:.2f}")
            else:  # 效果中等，保持不变
                print(f"Medium distillation effect, keeping threshold unchanged: {self.distill_threshold_ratio:.2f}")

        print("========================\n")

    def get_distill_stats(self) -> Dict[str, Any]:
        """获取蒸馏相关的统计信息，用于日志记录和监控

        Returns:
            Dict: 包含蒸馏相关统计信息的字典
        """
        # 计算基本统计信息
        avg_buffer_reward = np.mean([r for r, _ in self.buffer]) if self.buffer else 0.0
        buffer_fullness_ratio = len(self.buffer) / self.capacity if self.capacity > 0 else 0
        steps_since_last_distill = 0 if not hasattr(self, "last_distill_step") else self.steps_since_last_distill

        # 计算动态阈值相关因子（与should_distill方法中的计算保持一致）
        base_interval_factor = (
            max(0.5, 1.0 - (steps_since_last_distill / self.max_distill_interval))
            if self.max_distill_interval > 0
            else 1.0
        )
        fullness_factor = (
            min(1.0, buffer_fullness_ratio / self.distill_threshold_ratio) if self.distill_threshold_ratio > 0 else 1.0
        )

        quality_factor = 0.5
        if avg_buffer_reward > 0:
            quality_factor = min(1.5, max(0.5, avg_buffer_reward / self.min_reward))

        history_factor = 1.0
        if len(self.reward_improvement) > 2:
            recent_improvements = self.reward_improvement[-3:]
            avg_improvement = np.mean(recent_improvements)
            if avg_improvement > 0.1:  # 历史改进明显
                history_factor = 0.8
            elif avg_improvement < 0.02:  # 历史改进不明显
                history_factor = 1.2

        # 计算当前的动态阈值，添加 epsilon 防止除零错误
        epsilon = 1e-8
        dynamic_threshold = (
            self.distill_threshold_ratio
            * base_interval_factor
            * history_factor
            / (fullness_factor * quality_factor + epsilon)
        )

        # 返回详细的统计信息
        stats = {
            # 基本信息
            "last_distill_step": self.last_distill_step,
            "steps_since_last_distill": steps_since_last_distill,
            "buffer_size": len(self.buffer),
            "buffer_fullness_ratio": buffer_fullness_ratio,
            "distill_threshold_ratio": self.distill_threshold_ratio,
            "avg_buffer_reward": avg_buffer_reward,
            # 动态阈值相关因子
            "dynamic_threshold": dynamic_threshold,
            "base_interval_factor": base_interval_factor,
            "fullness_factor": fullness_factor,
            "quality_factor": quality_factor,
            "history_factor": history_factor,
            # 历史改进信息
            "avg_recent_improvement": (
                np.mean(self.reward_improvement[-3:]) if len(self.reward_improvement) >= 3 else 0.0
            ),
            "total_distillations": len(self.reward_improvement),
            # 蒸馏状态 - 更新同样的阈值
            "should_distill_soon": buffer_fullness_ratio >= dynamic_threshold
            or (avg_buffer_reward > 0.95 and len(self.buffer) >= 10),
            # 指数衰减相关信息
            "reward_convergence_factor": self.reward_convergence_factor,
            "target_distill_interval": self.target_distill_interval,
            "reward_diff": abs(self.avg_rewards[-1] - self.avg_rewards[-2]) if len(self.avg_rewards) >= 2 else 0,
            "avg_reward_current": self.avg_rewards[-1] if self.avg_rewards else 0,
            "avg_reward_last": self.avg_rewards[-2] if len(self.avg_rewards) >= 2 else 0,
            # 固定间隔相关信息
            "fixed_distill_interval": self.fixed_distill_interval,
            "steps_until_next_distill": self.fixed_distill_interval
            - (steps_since_last_distill if hasattr(self, "steps_since_last_distill") else 0),
        }

        return stats
