from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from app.workers.worker_stop import StopRequested

Coord = tuple[int, int]  # (row, col)

DIRS: list[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
DIR_NAMES = ["U", "R", "D", "L"]


@dataclass(frozen=True)
class SnakeRollout:
    steps: list[dict[str, Any]]
    summary: dict[str, Any]
    sim: dict[str, Any]


def now_ms() -> int:
    return int(time.time() * 1000)


def target_length_from_coverage(size: int, coverage: float) -> int:
    return max(1, min(size * size, int(math.ceil(float(coverage) * (size * size)))))


def generate_rollout(
    size: int,
    seed: int,
    coverage_target: float,
    max_steps: int,
    steps_saved: int,
    reject_below: bool = False,
    max_attempts: int = 1,
    *,
    stop_requested: Any | None = None,
) -> SnakeRollout:
    for attempt in range(int(max_attempts)):
        if stop_requested and stop_requested():
            raise StopRequested("stop requested during rollout generation")
        attempt_seed = int(seed) + attempt
        rollout = _rollout_once(size, attempt_seed, float(coverage_target), int(max_steps), int(steps_saved), stop_requested)
        if reject_below and float(rollout.summary.get("coverage_max") or 0.0) < float(coverage_target):
            continue
        return rollout
    raise RuntimeError(f"reject sampling failed: size={size} target={coverage_target} attempts={max_attempts}")


def _rollout_once(size: int, seed: int, coverage_target: float, max_steps: int, steps_saved: int, stop_requested: Any | None) -> SnakeRollout:
    rng = random.Random(seed)
    path, is_cycle = build_traversal(size)
    init_len = 2
    dir_sign = rng.choice([1, -1])
    head_idx = pick_start_idx(len(path), dir_sign, size, coverage_target, rng, is_cycle, init_len)
    env = SnakeEnv(size=size, seed=seed, init_len=init_len, path=path, is_cycle=is_cycle, head_idx=head_idx, dir_sign=dir_sign)
    steps, summary = collect_steps(env, max_steps=max_steps, steps_saved=steps_saved, coverage_target=coverage_target, stop_requested=stop_requested)
    sim = {"seed": seed, "is_cycle": is_cycle, "init_len": init_len, "dir_sign": dir_sign, "head_idx": head_idx}
    return SnakeRollout(steps=steps, summary=summary, sim=sim)


def build_traversal(size: int) -> tuple[list[Coord], bool]:
    if size % 2 == 0:
        return build_hamiltonian_cycle(size), True
    return build_snake_path(size), False


def build_hamiltonian_cycle(size: int) -> list[Coord]:
    if size % 2 != 0:
        raise ValueError("hamiltonian cycle requires even size")
    cycle: list[Coord] = [(0, 0)]
    for r in range(size):
        if r % 2 == 0:
            cycle.extend([(r, c) for c in range(1, size)])
        else:
            cycle.extend([(r, c) for c in range(size - 1, 0, -1)])
    cycle.append((size - 1, 0))
    cycle.extend([(r, 0) for r in range(size - 2, 0, -1)])
    if len(cycle) != size * size:
        raise RuntimeError("cycle length mismatch")
    return cycle


def build_snake_path(size: int) -> list[Coord]:
    path: list[Coord] = []
    for r in range(size):
        cols = range(size) if r % 2 == 0 else range(size - 1, -1, -1)
        path.extend([(r, c) for c in cols])
    if len(path) != size * size:
        raise RuntimeError("path length mismatch")
    return path


def dir_from_to(a: Coord, b: Coord) -> int:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0:
        return 0
    if dr == 0 and dc == 1:
        return 1
    if dr == 1 and dc == 0:
        return 2
    if dr == 0 and dc == -1:
        return 3
    raise ValueError(f"non-adjacent move from {a} to {b}")


def relative_action(cur_dir: int, next_dir: int) -> str:
    delta = (next_dir - cur_dir) % 4
    if delta == 0:
        return "S"
    if delta == 3:
        return "L"
    if delta == 1:
        return "R"
    return "S"


def pick_start_idx(
    path_len: int, dir_sign: int, size: int, coverage_target: float, rng: random.Random, is_cycle: bool, init_len: int
) -> int:
    if is_cycle:
        return rng.randrange(path_len)
    if int(init_len) < 2:
        return rng.randrange(path_len)
    back_span = int(init_len) - 1
    lo = back_span if int(dir_sign) == 1 else 0
    hi = (path_len - 1) if int(dir_sign) == 1 else (path_len - 1 - back_span)
    if lo > hi:
        return rng.randrange(path_len)
    target_len = target_length_from_coverage(size, coverage_target)
    min_remaining = min(target_len, path_len)
    if dir_sign == 1:
        hi2 = min(hi, max(lo, path_len - min_remaining))
        return rng.randrange(lo, hi2 + 1)
    lo2 = max(lo, min(path_len - 1, int(min_remaining) - 1))
    return rng.randrange(lo2, hi + 1) if lo2 <= hi else hi


class SnakeEnv:
    def __init__(
        self, *, size: int, seed: int, init_len: int, path: list[Coord], is_cycle: bool, head_idx: int, dir_sign: int
    ) -> None:
        self.size = size
        self.rng = random.Random(seed)
        self.path = path
        self.path_len = len(path)
        self.is_cycle = is_cycle
        self.index_by_coord = {coord: i for i, coord in enumerate(path)}
        self.dir_sign = dir_sign
        self.head_idx = head_idx
        self.snake = init_snake(path, head_idx, dir_sign, init_len)
        if len(self.snake) < 2:
            raise RuntimeError("snake too short to define direction")
        self.cur_dir = dir_from_to(self.snake[1], self.snake[0])
        self.food = self.spawn_food()

    def can_step(self) -> bool:
        if self.is_cycle:
            return True
        nxt = self.head_idx + self.dir_sign
        return 0 <= nxt < self.path_len

    def tail_idx(self) -> int:
        return self.index_by_coord[self.snake[-1]]

    def default_next_cell(self) -> Coord:
        next_idx = (self.head_idx + self.dir_sign) % self.path_len if self.is_cycle else self.head_idx + self.dir_sign
        return self.path[next_idx]

    def step_to(self, next_cell: Coord) -> tuple[str, bool, bool, str | None]:
        next_dir = dir_from_to(self.snake[0], next_cell)
        action = relative_action(self.cur_dir, next_dir)
        grow = next_cell == self.food
        body = self.snake if grow else self.snake[:-1]
        if next_cell in body:
            return action, True, False, "self"
        if not grow:
            self.snake.pop()
        self.snake.insert(0, next_cell)
        self.head_idx = self.index_by_coord[next_cell]
        self.cur_dir = next_dir
        if grow:
            self.food = self.spawn_food()
        return action, False, grow, None

    def spawn_food(self) -> Coord:
        empties = [(r, c) for r in range(self.size) for c in range(self.size)]
        occupied = set(self.snake)
        empties = [p for p in empties if p not in occupied]
        if not empties:
            return (-1, -1)
        if self.is_cycle:
            return self.rng.choice(empties)
        forward = self.forward_empty(empties)
        return self.rng.choice(forward) if forward else (-1, -1)

    def forward_empty(self, empties: list[Coord]) -> list[Coord]:
        if self.dir_sign == 1:
            return [p for p in empties if self.index_by_coord[p] >= self.head_idx]
        return [p for p in empties if self.index_by_coord[p] <= self.head_idx]


def init_snake(path: list[Coord], head_idx: int, dir_sign: int, init_len: int) -> list[Coord]:
    snake: list[Coord] = []
    for i in range(init_len):
        idx = head_idx - i * dir_sign
        snake.append(path[idx % len(path)])
    return list(dict.fromkeys(snake))


def collect_steps(
    env: SnakeEnv, max_steps: int, steps_saved: int, coverage_target: float, stop_requested: Any | None = None
) -> tuple[list[dict], dict]:
    out: list[dict] = []
    best_len = len(env.snake)
    step_total = 0
    terminated = False
    for t in range(int(max_steps)):
        if stop_requested and stop_requested():
            raise StopRequested("stop requested during rollout steps")
        if not env.can_step():
            terminated = True
            break
        pre = _snapshot(env)
        nxt = pick_next_cell(env, enable_shortcut=env.is_cycle)
        if nxt is None:
            terminated = True
            break
        action, done, ate, collision = env.step_to(nxt)
        post = _snapshot(env)
        best_len = max(best_len, len(env.snake))
        step_total += 1
        if t < int(steps_saved):
            out.append(step_obj(t, pre, action, done, ate, collision, post))
        if done:
            terminated = True
            break
        if (float(best_len) / float(env.size * env.size)) >= float(coverage_target):
            break
    return out, summary_obj(env.size, best_len, step_total, terminated=terminated)


def pick_next_cell(env: SnakeEnv, enable_shortcut: bool) -> Coord | None:
    head = env.snake[0]
    default = env.default_next_cell()
    if not enable_shortcut:
        return default if _is_legal_move(env, default) else None
    best = _pick_best_shortcut(env, head, default)
    return best if best is not None else (default if _is_legal_move(env, default) else _any_legal(env, head))


def _pick_best_shortcut(env: SnakeEnv, head: Coord, default: Coord) -> Coord | None:
    default_idx = env.index_by_coord[default]
    best: tuple[int, Coord] | None = None
    for c in _candidate_cells(env, head):
        if not _is_shortcut_safe(env, c):
            continue
        score = _cycle_dist(env, env.index_by_coord[c], env.index_by_coord[env.food])
        if score >= _cycle_dist(env, default_idx, env.index_by_coord[env.food]):
            continue
        if best is None or score < best[0]:
            best = (score, c)
    return best[1] if best else None


def _candidate_cells(env: SnakeEnv, head: Coord) -> list[Coord]:
    cur = env.cur_dir
    dirs = [(cur + 3) % 4, cur, (cur + 1) % 4]  # left/straight/right (absolute dirs)
    out: list[Coord] = []
    for d in dirs:
        dr, dc = DIRS[d]
        cell = (head[0] + dr, head[1] + dc)
        if 0 <= cell[0] < env.size and 0 <= cell[1] < env.size and _is_legal_move(env, cell):
            out.append(cell)
    return out


def _any_legal(env: SnakeEnv, head: Coord) -> Coord | None:
    for c in _candidate_cells(env, head):
        return c
    return None


def _is_legal_move(env: SnakeEnv, next_cell: Coord) -> bool:
    grow = next_cell == env.food
    body = env.snake if grow else env.snake[:-1]
    return next_cell not in body


def _is_shortcut_safe(env: SnakeEnv, next_cell: Coord) -> bool:
    head_idx = env.head_idx
    tail_idx = env.tail_idx()
    nxt_idx = env.index_by_coord[next_cell]
    return _cycle_dist(env, head_idx, nxt_idx) < _cycle_dist(env, head_idx, tail_idx)


def _cycle_dist(env: SnakeEnv, a: int, b: int) -> int:
    n = env.path_len
    if env.dir_sign == 1:
        return (b - a) % n
    return (a - b) % n


def _snapshot(env: SnakeEnv) -> dict[str, Any]:
    return {
        "snake": [to_xy(p) for p in env.snake],
        "food": to_xy(env.food),
        "dir": DIR_NAMES[env.cur_dir],
    }


def step_obj(t: int, pre: dict[str, Any], action: str, done: bool, ate: bool, collision: str | None, post: dict[str, Any]) -> dict:
    return {
        "t": t,
        "snake": pre["snake"],
        "food": pre["food"],
        "dir": pre["dir"],
        "action": action,
        "done": bool(done),
        "info": {"ate": bool(ate), "collision": collision},
        "snake_next": post["snake"],
        "food_next": post["food"],
        "dir_next": post["dir"],
    }


def to_xy(p: Coord) -> list[int]:
    return [int(p[1]), int(p[0])]


def summary_obj(size: int, best_len: int, steps: int, terminated: bool) -> dict[str, Any]:
    cov = float(best_len) / float(size * size) if size > 0 else 0.0
    return {
        "coverage": cov,
        "coverage_max": cov,
        "snake_length_max": best_len,
        "length_max": best_len,
        "steps": int(steps),
        "terminated": bool(terminated),
        "truncated": False,
    }
