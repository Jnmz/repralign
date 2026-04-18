"""Base adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModelAdapter(ABC):
    """Abstract interface for model-specific feature extraction adapters."""

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model
        self.layer_names = layer_names

    @abstractmethod
    def forward(self, batch: object) -> object:
        """Run a forward pass using the adapter's expected batch format."""

    @abstractmethod
    def transform_hook_output(self, output: object) -> torch.Tensor:
        """Convert a hooked module output into a tensor."""

    def discover_layers(self) -> list[str]:
        """Return all named modules on the wrapped model."""

        return [name for name, _ in self.model.named_modules()]
