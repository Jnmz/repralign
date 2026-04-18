from __future__ import annotations

import torch

from repralign.metrics.cka import linear_cka


def test_linear_cka_is_high_for_identical_features() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    score = linear_cka(features, features)
    assert score > 0.99


def test_linear_cka_accepts_different_feature_dims() -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
    b = torch.tensor([[1.0], [2.0], [0.0]])
    score = linear_cka(a, b)
    assert 0.0 <= score <= 1.0
