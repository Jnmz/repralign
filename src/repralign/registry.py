"""Registries for adapters and metrics."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from repralign.adapters import GenericTorchAdapter, HuggingFaceVisionAdapter
from repralign.metrics import cknna, linear_cka


MetricFn = Callable[..., float]


ADAPTERS: dict[str, type] = {
    "generic_torch": GenericTorchAdapter,
    "hf_vision": HuggingFaceVisionAdapter,
}

METRICS: dict[str, MetricFn] = {
    "cka": linear_cka,
    "cknna": cknna,
}


def create_adapter(name: str, model: nn.Module, layer_names: list[str]) -> object:
    """Instantiate a registered adapter."""

    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}")
    return ADAPTERS[name](model=model, layer_names=layer_names)


def compute_metric(name: str, features_a: torch.Tensor, features_b: torch.Tensor, **kwargs: object) -> float:
    """Compute a registered similarity metric."""

    if name not in METRICS:
        raise KeyError(f"Unknown metric: {name}")
    return METRICS[name](features_a, features_b, **kwargs)
