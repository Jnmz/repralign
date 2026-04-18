from __future__ import annotations

import torch
from torch import nn

from repralign.hooks import get_module_by_name, register_activation_hooks


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        return self.linear2(x)


def test_get_module_by_name_finds_submodule() -> None:
    model = TinyModel()
    module = get_module_by_name(model, "linear1")
    assert isinstance(module, nn.Linear)


def test_register_activation_hooks_captures_output() -> None:
    model = TinyModel()
    hooks = register_activation_hooks(model, ["linear1"])
    with torch.no_grad():
        model(torch.randn(2, 4))
    assert "linear1" in hooks.activations
    assert hooks.activations["linear1"].shape == (2, 4)
    hooks.remove()
