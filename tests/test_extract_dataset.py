from __future__ import annotations

import torch
from torch import nn

from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dataset


class TinyTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def test_extract_feature_dataset_concatenates_batches() -> None:
    model = TinyTokenModel()
    adapter = GenericTorchAdapter(model=model, layer_names=["layers.0", "layers.1"])
    batches = [torch.randn(2, 3, 4), torch.randn(1, 3, 4)]

    features = extract_feature_dataset(adapter=adapter, batches=batches, pooling="mean_tokens", normalize=False)

    assert set(features.keys()) == {"layers.0", "layers.1"}
    assert features["layers.0"].shape == (3, 4)
