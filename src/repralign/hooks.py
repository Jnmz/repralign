"""Utilities for registering and managing forward hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


ActivationTransform = Callable[[torch.Tensor], torch.Tensor]


def _default_transform(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.detach()
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0].detach()
    raise TypeError(f"Unsupported hooked output type: {type(output)!r}")


@dataclass
class HookHandleCollection:
    """Track registered forward hooks and their captured activations."""

    activations: dict[str, torch.Tensor]
    handles: list[torch.utils.hooks.RemovableHandle]

    def clear(self) -> None:
        self.activations.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Return a submodule by dotted name."""

    modules = dict(model.named_modules())
    if module_name not in modules:
        available = ", ".join(sorted(modules.keys())[:20])
        raise KeyError(f"Module {module_name!r} was not found. Example available names: {available}")
    return modules[module_name]


def register_activation_hooks(
    model: nn.Module,
    layer_names: list[str],
    transform: Optional[ActivationTransform] = None,
) -> HookHandleCollection:
    """Register forward hooks on the requested layers.

    Parameters
    ----------
    model:
        PyTorch module to inspect.
    layer_names:
        Dotted module names from ``model.named_modules()``.
    transform:
        Optional callable to convert raw hook outputs into tensors.

    Returns
    -------
    HookHandleCollection
        Container with captured activations and hook handles.
    """

    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    output_transform = transform or _default_transform

    for layer_name in layer_names:
        module = get_module_by_name(model, layer_name)

        def _hook(_: nn.Module, __: tuple[object, ...], output: object, key: str = layer_name) -> None:
            activations[key] = output_transform(output)

        handles.append(module.register_forward_hook(_hook))

    return HookHandleCollection(activations=activations, handles=handles)
