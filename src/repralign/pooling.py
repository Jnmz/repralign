"""Feature pooling helpers."""

from __future__ import annotations

from typing import Literal

import torch


PoolingMode = Literal["cls", "mean_tokens", "flatten_mean"]


def apply_pooling(features: torch.Tensor, mode: PoolingMode) -> torch.Tensor:
    """Pool a tensor into a sample-by-feature matrix.

    Parameters
    ----------
    features:
        Tensor with batch in the first dimension. Common shapes are
        ``(batch, tokens, dim)`` or ``(batch, dim)``.
    mode:
        Pooling strategy.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(batch, feature_dim)``.
    """

    if features.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {tuple(features.shape)}")

    if mode == "cls":
        if features.ndim < 3:
            raise ValueError("CLS pooling expects a token dimension")
        return features[:, 0, :]
    if mode == "mean_tokens":
        if features.ndim == 2:
            return features
        return features.mean(dim=1)
    if mode == "flatten_mean":
        return features.reshape(features.shape[0], -1).mean(dim=1, keepdim=True)

    raise ValueError(f"Unsupported pooling mode: {mode}")


def l2_normalize(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Apply row-wise L2 normalization."""

    denom = torch.linalg.norm(features, dim=1, keepdim=True).clamp_min(eps)
    return features / denom
