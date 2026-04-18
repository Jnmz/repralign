"""Minimal runnable example comparing two toy vision models."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dict
from repralign.metrics import linear_cka
from repralign.plotting import plot_layer_alignment, write_similarity_csv

from examples.model_factories import build_toy_candidate, build_toy_semantic_reference


def main() -> None:
    torch.manual_seed(0)
    batch = torch.randn(24, 8, 16)

    candidate = build_toy_candidate()
    reference = build_toy_semantic_reference()

    candidate_layers = [f"layers.{idx}" for idx in range(len(candidate.layers))]
    reference_layers = [f"layers.{idx}" for idx in range(len(reference.layers))]

    candidate_adapter = GenericTorchAdapter(candidate, candidate_layers)
    reference_adapter = GenericTorchAdapter(reference, reference_layers)

    candidate_features = extract_feature_dict(candidate_adapter, batch=batch, pooling="mean_tokens")
    reference_features = extract_feature_dict(reference_adapter, batch=batch, pooling="mean_tokens")
    reference_layer_name = reference_layers[-1]
    reference_tensor = reference_features[reference_layer_name]

    rows = []
    scores = []
    for idx, layer_name in enumerate(candidate_layers):
        score = linear_cka(candidate_features[layer_name], reference_tensor)
        scores.append(score)
        rows.append(
            {
                "metric": "cka",
                "candidate_layer_index": idx,
                "candidate_layer_name": layer_name,
                "reference_layer_name": reference_layer_name,
                "score": score,
            }
        )

    output_dir = Path("outputs/minimal_compare")
    write_similarity_csv(output_dir / "similarity.csv", rows)
    plot_layer_alignment(
        [
            {
                "label": "candidate vs semantic reference",
                "x": list(range(len(scores))),
                "y": scores,
            }
        ],
        output_path=output_dir / "similarity.png",
        title="Minimal Layer-wise CKA",
        ylabel="CKA",
    )
    print(f"Saved example outputs to {output_dir}")


if __name__ == "__main__":
    main()
