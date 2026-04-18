"""Feature cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def save_feature_cache(
    path: Union[str, Path],
    features: dict[str, torch.Tensor],
    metadata: Optional[dict[str, object]] = None,
) -> Path:
    """Save extracted features into an ``.npz`` cache file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in features.items()}
    np.savez(path, **arrays)

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_feature_cache(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    """Load an ``.npz`` feature cache."""

    with np.load(Path(path), allow_pickle=False) as data:
        return {key: torch.from_numpy(data[key]).float() for key in data.files}
