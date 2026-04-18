from __future__ import annotations

import torch

from repralign.pooling import apply_pooling


def test_cls_pooling_returns_first_token() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pooled = apply_pooling(features, mode="cls")
    assert pooled.shape == (2, 4)
    assert torch.equal(pooled[0], features[0, 0])


def test_mean_tokens_pooling_averages_token_dimension() -> None:
    features = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    pooled = apply_pooling(features, mode="mean_tokens")
    assert torch.allclose(pooled, torch.tensor([[3.0, 5.0]]))


def test_flatten_mean_returns_single_feature_column() -> None:
    features = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    pooled = apply_pooling(features, mode="flatten_mean")
    assert pooled.shape == (1, 1)
    assert torch.allclose(pooled, torch.tensor([[4.0]]))
