from __future__ import annotations

import torch

from repralign.metrics.cknna import cknna


def test_cknna_is_one_for_identical_neighbor_structure() -> None:
    features = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    score = cknna(features, features, k=2)
    assert score == 1.0


def test_cknna_handles_large_k_by_clipping() -> None:
    features = torch.randn(5, 3)
    score = cknna(features, features, k=100)
    assert 0.0 <= score <= 1.0
