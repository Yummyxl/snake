from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from app.config import SnakeEnvCfg, snake_reward_cfg
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
        reward_cfg = snake_reward_cfg()
        self.dist_shaping_weight = float(reward_cfg.get("dist_shaping_weight") or 0.0)
        self.dist_shaping_clip = float(reward_cfg.get("dist_shaping_clip") or 0.0)
        self.hunger_budget = float(reward_cfg.get("hunger_budget") or 0.0)
        self.hunger_grace_steps = int(reward_cfg.get("hunger_grace_steps") or 0)
        self.terminal_incomplete_beta = float(reward_cfg.get("terminal_incomplete_beta") or 0.0)
        self.completion_bonus = float(reward_cfg.get("completion_bonus") or 0.0)
        self.rng = random.Random(int(cfg.seed))
        self.steps = 0
        self.steps_since_last_eat = 0
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
        self.steps_since_last_eat += 1
        if not _in_bounds(nxt, self.size):
            return self._terminal_step(truncated=truncated, collision="wall")
        grow = nxt == self.food
        if grow:
            self.steps_since_last_eat = 0
        if _hits_body(nxt, self.snake, grow=grow):
            return self._terminal_step(truncated=truncated, collision="self")
        dist_delta = _dist_delta(head, nxt, self.food)
        self._advance(nxt, next_dir, grow)
        done = bool(truncated or self._is_complete())
        reward = self._step_reward(ate=grow, death=False, dist_delta=dist_delta, done=done)
        return StepResult(
            reward=reward,
            done=done,
            ate=grow,
            collision=None,
            truncated=truncated,
        )

    def _terminal_step(self, *, truncated: bool, collision: str) -> StepResult:
        reward = self._step_reward(ate=False, death=True, dist_delta=0.0, done=True)
        return StepResult(reward=reward, done=True, ate=False, collision=str(collision), truncated=bool(truncated))

    def _step_reward(self, *, ate: bool, death: bool, dist_delta: float, done: bool) -> float:
        r = _reward(ate=ate, death=death, dist_delta=dist_delta, dist_weight=self.dist_shaping_weight, dist_clip=self.dist_shaping_clip)
        r += self._hunger_penalty()
        if done:
            r += self._terminal_adjust()
        return float(r)

    def _hunger_penalty(self) -> float:
        if self.hunger_budget <= 0 or self.max_steps <= 0:
            return 0.0
        if int(self.steps_since_last_eat) <= int(self.hunger_grace_steps):
            return 0.0
        return -float(self.hunger_budget) / float(max(1, int(self.max_steps)))

    def _terminal_adjust(self) -> float:
        if self._is_complete():
            return float(self.completion_bonus)
        if self.terminal_incomplete_beta <= 0:
            return 0.0
        total = int(self.size) * int(self.size)
        cov = float(len(self.snake)) / float(max(1, total))
        return -float(self.terminal_incomplete_beta) * float(max(0.0, 1.0 - cov))

    def _is_complete(self) -> bool:
        return len(self.snake) >= self.size * self.size

    def _advance(self, nxt: Coord, next_dir: int, grow: bool) -> None:
        if not grow:
            self.snake.pop()
        self.snake.insert(0, nxt)
        self.cur_dir = next_dir
        if grow:
            self.food = self._spawn_food()

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
    _ = eval_id
    phase_off = 0 if str(phase) == "bc" else 500_000
    base = int(stage_id) * 1_000_000 + int(phase_off)
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


def _reward(*, ate: bool, death: bool, dist_delta: float, dist_weight: float, dist_clip: float) -> float:
    w_eat = 1.0
    w_death = 10.0
    w_step = 0.0
    dd = 0.0
    if float(dist_weight) != 0.0 and float(dist_clip) > 0.0:
        dd = float(max(-float(dist_clip), min(float(dist_clip), float(dist_delta or 0.0))))
    return (w_eat if ate else 0.0) - (w_death if death else 0.0) + (float(dist_weight) * dd) - w_step


def _in_bounds(p: Coord, size: int) -> bool:
    r, c = p
    return 0 <= int(r) < int(size) and 0 <= int(c) < int(size)


def _hits_body(nxt: Coord, snake: list[Coord], *, grow: bool) -> bool:
    body = snake if grow else snake[:-1]
    return nxt in body


def _dist_delta(head: Coord, nxt: Coord, food: Coord) -> float:
    if int(food[0]) < 0 or int(food[1]) < 0:
        return 0.0
    return float(_manhattan(head, food) - _manhattan(nxt, food))


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


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
