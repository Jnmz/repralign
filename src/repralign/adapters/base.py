"""Base adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
from torch import nn


class BaseModelAdapter(ABC):
    """Abstract interface for model-specific feature extraction adapters."""

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
        processor: Optional[Any] = None,
        adapter_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.layer_names = layer_names
        self.processor = processor
        self.adapter_kwargs = adapter_kwargs or {}

    @abstractmethod
    def forward(self, batch: object) -> object:
        """Run a forward pass using the adapter's expected batch format."""

    @abstractmethod
    def transform_hook_output(self, output: object) -> torch.Tensor:
        """Convert a hooked module output into a tensor."""

    def prepare_batch(self, batch: object) -> object:
        """Convert raw dataset examples into model inputs when needed."""

        return batch

    def get_hook_model(self) -> nn.Module:
        """Return the module whose layers should be hooked."""

        return self.model

    def move_to_device(self, device: Union[str, torch.device]) -> None:
        """Move the relevant model components to the target device."""

        self.get_hook_model().to(device)

    def discover_layers(self) -> list[str]:
        """Return all named modules on the wrapped model."""

        return [name for name, _ in self.get_hook_model().named_modules()]
