from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from app.ml.encoding import build_2d_sincos_pe


@dataclass(frozen=True)
class ModelCfg:
    in_channels: int = 11
    stride: int = 2
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4


class CnnVitActorCritic(nn.Module):
    def __init__(self, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _Backbone(cfg.in_channels, cfg.d_model)
        self.vit = _Vit(cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio)
        self.heads = _Heads(cfg.d_model)
        self._pe_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        tokens = _to_tokens(f)
        pe = self._pe(f.shape[-2], f.shape[-1], tokens.shape[-1], tokens.device)
        tokens = tokens + pe[None, :, :]
        z = self.vit(tokens)
        pooled = z.mean(dim=1)
        return self.heads(pooled)

    def _pe(self, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        key = (int(h), int(w), str(device))
        out = self._pe_cache.get(key)
        if out is None or out.device != device:
            out = build_2d_sincos_pe(h, w, dim, device)
            self._pe_cache[key] = out
        return out


class _Backbone(nn.Module):
    def __init__(self, in_ch: int, d_model: int) -> None:
        super().__init__()
        self.stem = _conv(in_ch, 64, stride=1)
        self.res64 = nn.Sequential(_Res(64), _Res(64))
        self.down = _conv(64, 128, stride=2)
        self.res128 = nn.Sequential(_Res(128), _Res(128))
        self.proj = nn.Conv2d(128, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res64(self.stem(x))
        x = self.res128(self.down(x))
        return self.proj(x)


class _Vit(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        ff = int(d_model) * int(mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff, dropout=0.0, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class _Heads(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.policy = nn.Linear(d_model, 3)
        self.value = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(x), self.value(x).squeeze(-1)


class _Res(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = _conv(ch, ch, stride=1)
        self.c2 = _conv(ch, ch, stride=1, act=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.c2(self.c1(x)))


def _conv(in_ch: int, out_ch: int, *, stride: int, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def _to_tokens(f: torch.Tensor) -> torch.Tensor:
    b, d, h, w = f.shape
    return f.permute(0, 2, 3, 1).reshape(b, h * w, d)
