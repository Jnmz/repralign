"""Small model factories used by examples and CLI smoke tests."""

from __future__ import annotations

import torch
from torch import nn


class ToyVisionModel(nn.Module):
    def __init__(self, embed_dim: int, depth: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.patch_embed = nn.Linear(16, embed_dim)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU()) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_embed(x)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return self.head(hidden)


def build_toy_candidate() -> nn.Module:
    return ToyVisionModel(embed_dim=32, depth=6, seed=7)


def build_toy_semantic_reference() -> nn.Module:
    return ToyVisionModel(embed_dim=32, depth=4, seed=11)


def build_toy_generation_reference() -> nn.Module:
    return ToyVisionModel(embed_dim=32, depth=4, seed=19)
