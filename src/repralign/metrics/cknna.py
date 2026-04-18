"""Nearest-neighbor overlap similarity."""

from __future__ import annotations

import torch


def cknna(features_a: torch.Tensor, features_b: torch.Tensor, k: int = 5) -> float:
    """Compute a simple neighborhood-overlap similarity between two spaces.

    Parameters
    ----------
    features_a:
        Tensor of shape ``(n_samples, n_features_a)``.
    features_b:
        Tensor of shape ``(n_samples, n_features_b)``.
    k:
        Number of nearest neighbors per sample to compare. The effective value
        is clipped to ``n_samples - 1``.

    Returns
    -------
    float
        Mean overlap ratio between the top-k neighborhoods induced by the two
        feature spaces.

    Notes
    -----
    This implementation is intentionally simple and reusable for v0.1. It is
    best interpreted as a local-structure agreement score. It assumes rows are
    sample-aligned and that neighborhood comparison is meaningful for the batch.
    """

    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("cknna expects 2D feature matrices")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("cknna requires the same number of samples")
    if features_a.shape[0] < 2:
        raise ValueError("cknna requires at least two samples")

    n_samples = features_a.shape[0]
    effective_k = min(max(k, 1), n_samples - 1)

    distances_a = torch.cdist(features_a, features_a)
    distances_b = torch.cdist(features_b, features_b)
    distances_a.fill_diagonal_(float("inf"))
    distances_b.fill_diagonal_(float("inf"))

    neighbors_a = torch.topk(distances_a, k=effective_k, largest=False).indices
    neighbors_b = torch.topk(distances_b, k=effective_k, largest=False).indices

    overlaps = []
    for idx in range(n_samples):
        set_a = set(neighbors_a[idx].tolist())
        set_b = set(neighbors_b[idx].tolist())
        overlaps.append(len(set_a & set_b) / effective_k)

    return float(sum(overlaps) / len(overlaps))
