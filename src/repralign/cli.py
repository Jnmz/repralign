"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from repralign.cache import load_feature_cache, save_feature_cache
from repralign.config import load_batch_tensor, load_yaml_config, resolve_factory
from repralign.datasets import iter_batch_records, load_dataset_records
from repralign.extract import extract_feature_dataset, extract_feature_dict
from repralign.plotting import plot_csv_curves, write_similarity_csv
from repralign.registry import compute_metric, create_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repralign")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_layers = subparsers.add_parser("list-layers", help="List named modules from a model")
    list_layers.add_argument("--factory", required=True, help="Model factory in module:function form")
    list_layers.add_argument("--adapter", default="generic_torch")

    extract_cmd = subparsers.add_parser("extract-features", help="Extract and cache features from a YAML config")
    extract_cmd.add_argument("--config", required=True)

    analyze = subparsers.add_parser("analyze", help="Compute similarity from cached features")
    analyze.add_argument("--candidate-cache", required=True)
    analyze.add_argument("--reference-cache", required=True)
    analyze.add_argument("--metric", required=True, choices=["cka", "cknna"])
    analyze.add_argument("--output-csv", required=True)
    analyze.add_argument("--output-json", required=True)
    analyze.add_argument("--reference-layer", action="append", default=None)
    analyze.add_argument("--k", type=int, default=5, help="Neighborhood size for CKNNA")

    plot_cmd = subparsers.add_parser("plot", help="Plot one or more similarity CSVs")
    plot_cmd.add_argument("--input-csv", action="append", required=True)
    plot_cmd.add_argument("--output-png", required=True)
    plot_cmd.add_argument("--title", required=True)

    return parser


def _run_extract(config_path: str) -> None:
    config = load_yaml_config(config_path)
    output_dir = Path(config.outputs.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_specs = [config.candidate, *config.references]
    for spec in all_specs:
        factory = resolve_factory(spec.factory)
        model = factory(**spec.factory_kwargs)

        processor = None
        if spec.processor_factory is not None:
            processor_factory = resolve_factory(spec.processor_factory)
            processor = processor_factory(**spec.processor_kwargs)

        adapter = create_adapter(
            spec.adapter,
            model=model,
            layer_names=spec.layers,
            processor=processor,
            adapter_kwargs=spec.adapter_kwargs,
        )

        if config.dataset is not None:
            records = load_dataset_records(config.dataset.__dict__)
            batches = iter_batch_records(records, batch_size=config.dataset.batch_size)
            features = extract_feature_dataset(
                adapter=adapter,
                batches=batches,
                pooling=config.extract.pooling,
                normalize=config.extract.normalize,
                device=spec.device,
            )
        else:
            batch = load_batch_tensor(config.extract.batch_path, batch_key=config.extract.batch_key)
            features = extract_feature_dict(
                adapter=adapter,
                batch=batch,
                pooling=config.extract.pooling,
                normalize=config.extract.normalize,
                device=spec.device,
            )
        save_feature_cache(
            output_dir / f"{spec.name}_features.npz",
            features=features,
            metadata={
                "name": spec.name,
                "adapter": spec.adapter,
                "layers": spec.layers,
                "pooling": config.extract.pooling,
                "normalize": config.extract.normalize,
                "dataset": config.dataset.__dict__ if config.dataset is not None else None,
            },
        )


def _run_analysis(
    candidate_cache: str,
    reference_cache: str,
    metric: str,
    output_csv: str,
    output_json: str,
    reference_layers: Optional[list[str]],
    k: int,
) -> None:
    candidate_features = load_feature_cache(candidate_cache)
    reference_features = load_feature_cache(reference_cache)

    selected_reference_layers = reference_layers or [sorted(reference_features.keys())[-1]]
    for reference_layer in selected_reference_layers:
        if reference_layer not in reference_features:
            raise KeyError(f"Reference layer {reference_layer!r} not found in cache")

    rows: list[dict[str, object]] = []
    for reference_layer in selected_reference_layers:
        reference_tensor = reference_features[reference_layer]
        for index, layer_name in enumerate(sorted(candidate_features.keys())):
            kwargs = {"k": k} if metric == "cknna" else {}
            score = compute_metric(metric, candidate_features[layer_name], reference_tensor, **kwargs)
            rows.append(
                {
                    "metric": metric,
                    "candidate_layer_index": index,
                    "candidate_layer_name": layer_name,
                    "reference_layer_name": reference_layer,
                    "score": score,
                }
            )

    write_similarity_csv(output_csv, rows)
    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(
            {
                "metric": metric,
                "candidate_cache": candidate_cache,
                "reference_cache": reference_cache,
                "reference_layers": selected_reference_layers,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-layers":
        factory = resolve_factory(args.factory)
        model = factory()
        adapter = create_adapter(args.adapter, model=model, layer_names=[])
        for name in adapter.discover_layers():
            print(name)
        return

    if args.command == "extract-features":
        _run_extract(args.config)
        return

    if args.command == "analyze":
        _run_analysis(
            candidate_cache=args.candidate_cache,
            reference_cache=args.reference_cache,
            metric=args.metric,
            output_csv=args.output_csv,
            output_json=args.output_json,
            reference_layers=args.reference_layer,
            k=args.k,
        )
        return

    if args.command == "plot":
        plot_csv_curves(args.input_csv, output_path=args.output_png, title=args.title)
        return


if __name__ == "__main__":
    main()
