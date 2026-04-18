"""Linear CKA implementation."""

from __future__ import annotations

import torch


def linear_cka(features_a: torch.Tensor, features_b: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute linear CKA between two sample-aligned feature matrices.

    Parameters
    ----------
    features_a:
        Tensor of shape ``(n_samples, n_features_a)``.
    features_b:
        Tensor of shape ``(n_samples, n_features_b)``.
    eps:
        Numerical stabilizer for degenerate inputs.

    Returns
    -------
    float
        Scalar similarity score in approximately ``[0, 1]`` for typical inputs.

    Notes
    -----
    This function centers features row-wise across samples before computing the
    Hilbert-Schmidt independence criterion in the primal form. It expects that
    rows correspond to the same ordered samples in both inputs.
    """

    if features_a.ndim != 2 or features_b.ndim != 2:
        raise ValueError("linear_cka expects 2D feature matrices")
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError("linear_cka requires the same number of samples")

    a = features_a - features_a.mean(dim=0, keepdim=True)
    b = features_b - features_b.mean(dim=0, keepdim=True)

    cross = torch.linalg.norm(a.T @ b, ord="fro") ** 2
    self_a = torch.linalg.norm(a.T @ a, ord="fro")
    self_b = torch.linalg.norm(b.T @ b, ord="fro")
    denom = (self_a * self_b).clamp_min(eps)
    return float((cross / denom).item())
