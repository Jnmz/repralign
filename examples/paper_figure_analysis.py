"""Paper-inspired alignment workflow using toy models.

This example follows the general organization suggested by the attached
screenshot: compare one candidate model against two different references and
plot layer-wise similarity curves. It is intentionally generic and is not an
official reproduction of any specific paper implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.cache import save_feature_cache
from repralign.extract import extract_feature_dict
from repralign.plotting import plot_layer_alignment, write_similarity_csv
from repralign.registry import compute_metric

from examples.model_factories import (
    build_toy_candidate,
    build_toy_generation_reference,
    build_toy_semantic_reference,
)


def _compare(
    candidate_features: dict[str, torch.Tensor],
    reference_features: dict[str, torch.Tensor],
    metric: str,
    reference_label: str,
) -> tuple[list[dict[str, object]], list[float]]:
    candidate_layer_names = sorted(candidate_features.keys())
    reference_layer_name = sorted(reference_features.keys())[-1]
    reference_tensor = reference_features[reference_layer_name]

    rows = []
    scores = []
    for index, layer_name in enumerate(candidate_layer_names):
        kwargs = {"k": 5} if metric == "cknna" else {}
        score = compute_metric(metric, candidate_features[layer_name], reference_tensor, **kwargs)
        scores.append(score)
        rows.append(
            {
                "metric": metric,
                "candidate_layer_index": index,
                "candidate_layer_name": layer_name,
                "reference_layer_name": reference_layer_name,
                "reference_label": reference_label,
                "score": score,
            }
        )
    return rows, scores


def main() -> None:
    torch.manual_seed(42)
    batch = torch.randn(32, 8, 16)

    candidate = build_toy_candidate()
    semantic_reference = build_toy_semantic_reference()
    generation_reference = build_toy_generation_reference()

    candidate_layers = [f"layers.{idx}" for idx in range(len(candidate.layers))]
    semantic_layers = [f"layers.{idx}" for idx in range(len(semantic_reference.layers))]
    generation_layers = [f"layers.{idx}" for idx in range(len(generation_reference.layers))]

    candidate_features = extract_feature_dict(
        GenericTorchAdapter(candidate, candidate_layers),
        batch=batch,
        pooling="mean_tokens",
    )
    semantic_features = extract_feature_dict(
        GenericTorchAdapter(semantic_reference, semantic_layers),
        batch=batch,
        pooling="mean_tokens",
    )
    generation_features = extract_feature_dict(
        GenericTorchAdapter(generation_reference, generation_layers),
        batch=batch,
        pooling="mean_tokens",
    )

    output_dir = Path("outputs/paper_like")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_feature_cache(output_dir / "candidate_features.npz", candidate_features, metadata={"name": "candidate"})
    save_feature_cache(output_dir / "semantic_reference_features.npz", semantic_features, metadata={"name": "semantic_reference"})
    save_feature_cache(output_dir / "generation_reference_features.npz", generation_features, metadata={"name": "generation_reference"})

    cka_semantic_rows, cka_semantic_scores = _compare(candidate_features, semantic_features, metric="cka", reference_label="semantic")
    cka_generation_rows, cka_generation_scores = _compare(candidate_features, generation_features, metric="cka", reference_label="generation")
    cknna_semantic_rows, cknna_semantic_scores = _compare(candidate_features, semantic_features, metric="cknna", reference_label="semantic")
    cknna_generation_rows, cknna_generation_scores = _compare(candidate_features, generation_features, metric="cknna", reference_label="generation")

    write_similarity_csv(output_dir / "cka_semantic.csv", cka_semantic_rows)
    write_similarity_csv(output_dir / "cka_generation.csv", cka_generation_rows)
    write_similarity_csv(output_dir / "cknna_semantic.csv", cknna_semantic_rows)
    write_similarity_csv(output_dir / "cknna_generation.csv", cknna_generation_rows)

    plot_layer_alignment(
        [
            {"label": "semantic reference", "x": list(range(len(cka_semantic_scores))), "y": cka_semantic_scores},
            {"label": "generation reference", "x": list(range(len(cka_generation_scores))), "y": cka_generation_scores},
        ],
        output_path=output_dir / "cka_comparison.png",
        title="Paper-inspired Layer-wise CKA",
        ylabel="CKA",
    )
    plot_layer_alignment(
        [
            {"label": "semantic reference", "x": list(range(len(cknna_semantic_scores))), "y": cknna_semantic_scores},
            {"label": "generation reference", "x": list(range(len(cknna_generation_scores))), "y": cknna_generation_scores},
        ],
        output_path=output_dir / "cknna_comparison.png",
        title="Paper-inspired Layer-wise CKNNA",
        ylabel="CKNNA",
    )

    metadata = {
        "description": "Paper-inspired workflow using generic toy models. Not an official reproduction.",
        "candidate_layers": candidate_layers,
        "semantic_reference_layers": semantic_layers,
        "generation_reference_layers": generation_layers,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved paper-inspired outputs to {output_dir}")


if __name__ == "__main__":
    main()
