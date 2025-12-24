from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from app.config import SnakeEnvCfg
from app.sim.snake_rollout_gen import DIRS, DIR_NAMES, now_ms, to_xy

Coord = tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    ate: bool
    collision: str | None
    truncated: bool


class SnakeEnv:
    def __init__(self, cfg: SnakeEnvCfg) -> None:
        self.cfg = cfg
        self.size = int(cfg.size)
        self.max_steps = int(cfg.max_steps)
        self.rng = random.Random(int(cfg.seed))
        self.steps = 0
        self.snake, self.cur_dir = self._init_snake_and_dir()
        self.food = self._spawn_food()

    def snapshot(self) -> dict[str, Any]:
        return {"snake": [to_xy(p) for p in self.snake], "food": to_xy(self.food), "dir": DIR_NAMES[self.cur_dir]}

    def step(self, action: str) -> StepResult:
        next_dir = _apply_relative(self.cur_dir, action)
        head = self.snake[0]
        dr, dc = DIRS[next_dir]
        nxt = (head[0] + dr, head[1] + dc)
        self.steps += 1
        truncated = self.steps >= self.max_steps
        if not (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size):
            return StepResult(reward=_reward(self.size, ate=False, death=True), done=True, ate=False, collision="wall", truncated=truncated)
        grow = nxt == self.food
        body = self.snake if grow else self.snake[:-1]
        if nxt in body:
            return StepResult(reward=_reward(self.size, ate=False, death=True), done=True, ate=False, collision="self", truncated=truncated)
        if not grow:
            self.snake.pop()
        self.snake.insert(0, nxt)
        self.cur_dir = next_dir
        if grow:
            self.food = self._spawn_food()
        done = truncated or len(self.snake) >= self.size * self.size
        return StepResult(reward=_reward(self.size, ate=grow, death=False), done=done, ate=grow, collision=None, truncated=truncated)

    def _init_snake_and_dir(self) -> tuple[list[Coord], int]:
        size = self.size
        length = 2
        cur_dir = self.rng.randrange(4)
        head = self._pick_valid_head(cur_dir, length)
        snake = _build_straight_snake(head, cur_dir, length)
        return snake, cur_dir

    def _pick_valid_head(self, cur_dir: int, length: int) -> Coord:
        size = self.size
        dr_f, dc_f = DIRS[cur_dir]
        dr_b, dc_b = DIRS[(cur_dir + 2) % 4]
        r = _rand_axis(self.rng, size, dr_f, dr_b, length)
        c = _rand_axis(self.rng, size, dc_f, dc_b, length)
        return (r, c)

    def _spawn_food(self) -> Coord:
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in set(self.snake)]
        return self.rng.choice(empties) if empties else (-1, -1)


def eval_seed(stage_id: int, phase: str, eval_id: str, k: int) -> int:
    base = int(eval_id) * 1000 + int(stage_id) * 10 + (0 if phase == "bc" else 5)
    return base + int(k)


def eval_meta(stage_id: int, size: int, phase: str, eval_id: str, rollout_id: str, seed: int, max_steps: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage_id": int(stage_id),
        "size": int(size),
        "phase": str(phase),
        "kind": "eval",
        "coord": "xy",
        "action_space": "relative_lsr",
        "action_map": ["L", "S", "R"],
        "step_state": "pre_action",
        "eval_id": str(eval_id),
        "rollout_id": str(rollout_id),
        "seed": int(seed),
        "created_at_ms": now_ms(),
        "max_steps": int(max_steps),
    }


def _apply_relative(cur_dir: int, action: str) -> int:
    a = str(action).strip().upper()
    if a == "L":
        return (cur_dir + 3) % 4
    if a == "R":
        return (cur_dir + 1) % 4
    return cur_dir


def _reward(size: int, *, ate: bool, death: bool) -> float:
    w_len = 1.5
    w_death = 3.0
    w_step = 0.01 * (10.0 / float(max(1, int(size)))) ** 2
    return (w_len if ate else 0.0) - w_step - (w_death if death else 0.0)


def _build_straight_snake(head: Coord, cur_dir: int, length: int) -> list[Coord]:
    dr_b, dc_b = DIRS[(cur_dir + 2) % 4]
    snake: list[Coord] = [head]
    for i in range(1, int(length)):
        snake.append((head[0] + dr_b * i, head[1] + dc_b * i))
    return snake


def _rand_axis(rng: random.Random, size: int, fwd_d: int, back_d: int, length: int) -> int:
    lo, hi = 0, size - 1
    if fwd_d == 1:
        hi = min(hi, size - 2)
    elif fwd_d == -1:
        lo = max(lo, 1)
    span = int(length) - 1
    if back_d == 1:
        hi = min(hi, size - 1 - span)
    elif back_d == -1:
        lo = max(lo, span)
    if lo > hi:
        return rng.randrange(size)
    return rng.randrange(lo, hi + 1)
