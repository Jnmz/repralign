"""Registries for adapters and metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import torch
from torch import nn

from repralign.adapters import GenericTorchAdapter, HuggingFaceVisionAdapter, StableDiffusion3ReferenceAdapter
from repralign.metrics import cknna, linear_cka


MetricFn = Callable[..., float]


ADAPTERS: dict[str, type] = {
    "generic_torch": GenericTorchAdapter,
    "hf_vision": HuggingFaceVisionAdapter,
    "sd3_reference": StableDiffusion3ReferenceAdapter,
}

METRICS: dict[str, MetricFn] = {
    "cka": linear_cka,
    "cknna": cknna,
}


def create_adapter(
    name: str,
    model: nn.Module,
    layer_names: list[str],
    processor: Optional[Any] = None,
    adapter_kwargs: Optional[dict[str, Any]] = None,
) -> object:
    """Instantiate a registered adapter."""

    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}")
    return ADAPTERS[name](model=model, layer_names=layer_names, processor=processor, adapter_kwargs=adapter_kwargs)


def compute_metric(name: str, features_a: torch.Tensor, features_b: torch.Tensor, **kwargs: object) -> float:
    """Compute a registered similarity metric."""

    if name not in METRICS:
        raise KeyError(f"Unknown metric: {name}")
    return METRICS[name](features_a, features_b, **kwargs)
