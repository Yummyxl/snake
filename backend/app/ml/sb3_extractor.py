from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from app.ml.encoding import build_2d_sincos_pe
from app.ml.model import ModelCfg, _Backbone, _Vit, _to_tokens


class CnnVitFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Any, *, cfg: ModelCfg | None = None, freeze_bn: bool = True) -> None:
        c = cfg or ModelCfg()
        super().__init__(observation_space, features_dim=int(c.d_model))
        self.cfg = c
        self.freeze_bn = bool(freeze_bn)
        self.backbone = _Backbone(int(c.in_channels), int(c.d_model))
        self.vit = _Vit(int(c.d_model), int(c.n_layers), int(c.n_heads), int(c.mlp_ratio))
        self._pe_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        f = self.backbone(obs)
        tokens = _to_tokens(f)
        pe = self._pe(f.shape[-2], f.shape[-1], tokens.shape[-1], tokens.device)
        z = self.vit(tokens + pe[None, :, :])
        return z.mean(dim=1)

    def train(self, mode: bool = True):  # noqa: ANN001
        super().train(mode)
        if self.freeze_bn:
            self._set_bn_eval()
        return self

    def _set_bn_eval(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _pe(self, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        key = (int(h), int(w), str(device))
        out = self._pe_cache.get(key)
        if out is None or out.device != device:
            out = build_2d_sincos_pe(int(h), int(w), int(dim), device)
            self._pe_cache[key] = out
        return out
