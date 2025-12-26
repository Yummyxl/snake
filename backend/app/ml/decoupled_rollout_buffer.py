from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer


def _compute_gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_starts: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    last_values: np.ndarray,
    dones: np.ndarray,
) -> np.ndarray:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(int(rewards.shape[0]))):
        if step == int(rewards.shape[0]) - 1:
            next_non_terminal = 1.0 - dones.astype(np.float32)
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + float(gamma) * next_values * next_non_terminal - values[step]
        last_gae_lam = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae_lam
        adv[step] = last_gae_lam
    return adv


class DecoupledGAERolloutBuffer(RolloutBuffer):
    """
    VC-PPO 风格：advantages 用 actor λ，returns 用 critic λ（默认 1.0）。

    - self.advantages: 给 policy loss 用（保持方差可控）
    - self.returns: 给 value loss 用（critic 用 λ=1.0 近似 MC，避免长时序信号衰减）
    """

    def __init__(self, *args: Any, critic_gae_lambda: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.critic_gae_lambda = float(critic_gae_lambda)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_v = last_values.clone().cpu().numpy().flatten()
        adv_actor = _compute_gae_advantages(
            self.rewards,
            self.values,
            self.episode_starts,
            gamma=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            last_values=last_v,
            dones=dones,
        )
        adv_critic = _compute_gae_advantages(
            self.rewards,
            self.values,
            self.episode_starts,
            gamma=float(self.gamma),
            gae_lambda=float(self.critic_gae_lambda),
            last_values=last_v,
            dones=dones,
        )
        self.advantages[:] = adv_actor
        self.returns[:] = adv_critic + self.values
