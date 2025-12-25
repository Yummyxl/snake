from __future__ import annotations

import numpy as np
import torch
from gymnasium import Env, spaces

from app.config import SnakeEnvCfg
from app.ml.encoding import encode_grid
from app.sim.snake_env import SnakeEnv


class SnakeGymEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, *, size: int, seed: int, max_steps: int) -> None:
        self.size = int(size)
        self.base_seed = int(seed)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(11, self.size, self.size), dtype=np.float32)
        self._device = torch.device("cpu")
        self._actions = ["L", "S", "R"]
        self._episode = 0
        self._env: SnakeEnv | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: ANN001
        _ = options
        if seed is not None:
            self.base_seed = int(seed)
        self._episode += 1
        self._env = SnakeEnv(SnakeEnvCfg(size=self.size, seed=self.base_seed + self._episode, max_steps=self.max_steps))
        return self._obs(), {}

    def step(self, action):  # noqa: ANN001
        if self._env is None:
            raise RuntimeError("env not reset")
        a = self._actions[int(action)] if 0 <= int(action) <= 2 else "S"
        res = self._env.step(a)
        obs = self._obs()
        terminated = bool(res.done and not res.truncated)
        info = {"ate": bool(res.ate), "collision": res.collision}
        return obs, float(res.reward), terminated, bool(res.truncated), info

    def _obs(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("env not reset")
        s = self._env.snapshot()
        denom = float(max(1, int(self._env.max_steps)))
        time_left = float(max(0.0, min(1.0, (float(self._env.max_steps) - float(self._env.steps)) / denom)))
        hunger_steps = max(0, int(self._env.steps_since_last_eat) - int(self._env.hunger_grace_steps))
        hunger = float(max(0.0, min(1.0, float(hunger_steps) / denom)))
        coverage_norm = float(len(self._env.snake)) / float(max(1, int(self.size) * int(self.size)))
        x = encode_grid(
            size=self.size,
            snake=list(s["snake"]),
            food=list(s["food"]),
            dir_name=str(s["dir"]),
            device=self._device,
            time_left=time_left,
            hunger=hunger,
            coverage_norm=coverage_norm,
        )
        return x.detach().cpu().numpy()
