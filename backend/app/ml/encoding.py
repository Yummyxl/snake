from __future__ import annotations

from typing import Any

import torch


def encode_grid(
    *,
    size: int,
    snake: list[list[int]],
    food: list[int],
    dir_name: str,
    device: torch.device,
    time_left: float | None = None,
    hunger: float | None = None,
    coverage_norm: float | None = None,
) -> torch.Tensor:
    x = torch.zeros((11, size, size), device=device, dtype=torch.float32)
    _paint_head(x, snake)
    _paint_food(x, food)
    _paint_occupied(x, snake)
    _paint_body_order(x, snake)
    _paint_dir(x, dir_name)
    _paint_scalar(x, 8, time_left)
    _paint_scalar(x, 9, hunger)
    _paint_scalar(x, 10, coverage_norm)
    return x


def build_2d_sincos_pe(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    _assert_dim(dim)
    row = _axis_pe(h, dim // 2, device)
    col = _axis_pe(w, dim // 2, device)
    pe = torch.empty((h, w, dim), device=device, dtype=torch.float32)
    pe[:, :, : dim // 2] = row[:, None, :]
    pe[:, :, dim // 2 :] = col[None, :, :]
    return pe.reshape(h * w, dim)


def _assert_dim(dim: int) -> None:
    if dim % 4 != 0:
        raise ValueError("positional encoding dim must be divisible by 4")


def _axis_pe(n: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("axis pe dim must be even")
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    omega = _omega(dim // 2, device)[None, :]
    angles = pos * omega
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def _omega(k: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(k, device=device, dtype=torch.float32)
    return 1.0 / (10000.0 ** (idx / float(max(1, k))))


def _paint_head(x: torch.Tensor, snake: list[list[int]]) -> None:
    if not snake:
        return
    hx, hy = _xy(snake[0])
    x[0, hy, hx] = 1.0


def _paint_food(x: torch.Tensor, food: list[int]) -> None:
    fx, fy = _xy(food)
    if fx < 0 or fy < 0:
        return
    x[1, fy, fx] = 1.0


def _paint_occupied(x: torch.Tensor, snake: list[list[int]]) -> None:
    for p in snake:
        px, py = _xy(p)
        x[2, py, px] = 1.0


def _paint_body_order(x: torch.Tensor, snake: list[list[int]]) -> None:
    if len(snake) <= 1:
        return
    denom = float(max(1, len(snake) - 1))
    for idx, p in enumerate(snake):
        px, py = _xy(p)
        x[3, py, px] = 1.0 - float(idx) / denom


def _paint_dir(x: torch.Tensor, dir_name: str) -> None:
    idx = {"U": 0, "R": 1, "D": 2, "L": 3}.get(str(dir_name).strip().upper())
    if idx is None:
        return
    x[4 + idx, :, :] = 1.0


def _paint_scalar(x: torch.Tensor, ch: int, v: float | None) -> None:
    if v is None:
        return
    vv = float(v)
    if vv != vv:
        return
    vv = max(0.0, min(1.0, vv))
    x[int(ch), :, :] = vv


def _xy(p: Any) -> tuple[int, int]:
    if not isinstance(p, list) or len(p) != 2:
        return -1, -1
    return int(p[0]), int(p[1])
