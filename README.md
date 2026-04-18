# repralign

`repralign` is a small, reusable Python toolkit for layer-wise representation alignment analysis in PyTorch and multimodal research workflows. It is designed for experiments where you want to compare intermediate features from one model against one or more reference models, compute similarity curves across layers, and save clean artifacts for later inspection.

The project is inspired by the general workflow shown in the attached paper screenshot: compare a candidate model against different reference models, inspect how alignment evolves across layers, and summarize the result with publication-style figures. It does **not** claim to reproduce the paper authors' private implementation, hidden internals, or unpublished code.

## Motivation

Many multimodal papers reason about whether a learned representation behaves more like a semantic encoder, a generation-oriented encoder, or something in between. `repralign` provides a generic foundation for that style of analysis:

- extract intermediate features from configurable layers
- pool or normalize features into comparable representations
- compute layer-wise similarity with reusable metrics
- cache extracted features to avoid repeated forward passes
- save CSV, PNG, and JSON outputs for review and paper drafting

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional Hugging Face support:

```bash
pip install -e .[dev,huggingface]
```

## Package Structure

```text
repralign/
  src/repralign/
    adapters/
    metrics/
    cache.py
    cli.py
    config.py
    extract.py
    hooks.py
    plotting.py
    pooling.py
    registry.py
  tests/
  examples/
  configs/
```

## Practical View Of The Metrics

### CKA

Linear CKA measures whether two representation spaces preserve similar pairwise structure across examples. In practice, it is useful when comparing layers with different feature dimensionalities because it focuses on relative geometry rather than exact coordinates.

Expected input shape:

- `features_a`: `(n_samples, n_features_a)`
- `features_b`: `(n_samples, n_features_b)`

### CKNNA

`repralign` implements CKNNA as a neighborhood-overlap style metric: for each sample, compare its top-`k` nearest neighbors in one representation space with its top-`k` nearest neighbors in the other space, then average the overlap ratio. This makes it easy to ask whether two layers induce similar local neighborhoods even if their global geometry differs.

Expected input shape:

- `features_a`: `(n_samples, n_features_a)`
- `features_b`: `(n_samples, n_features_b)`

Limitations:

- both metrics expect aligned samples in the same row order
- CKNNA becomes unstable for very small batch sizes
- pooled features discard token-level structure in this v0.1

## Quickstart

Run the toy comparison example:

```bash
python examples/minimal_compare.py
```

Run the paper-inspired workflow example:

```bash
python examples/paper_figure_analysis.py
```

Both examples save outputs under `outputs/`.

## CLI Usage

List layers from a model factory:

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

Extract features with a YAML config:

```bash
repralign extract-features --config configs/example_analysis.yaml
```

Run similarity analysis from cached feature files:

```bash
repralign analyze \
  --candidate-cache outputs/paper_like/candidate_features.npz \
  --reference-cache outputs/paper_like/semantic_reference_features.npz \
  --metric cka \
  --output-csv outputs/paper_like/cka_semantic.csv \
  --output-json outputs/paper_like/cka_semantic.json
```

Plot one or more similarity curves:

```bash
repralign plot \
  --input-csv outputs/paper_like/cka_semantic.csv \
  --input-csv outputs/paper_like/cka_generation.csv \
  --output-png outputs/paper_like/cka_comparison.png \
  --title "Layer-wise Alignment"
```

## Python API

```python
from repralign.adapters.generic_torch import GenericTorchAdapter
from repralign.extract import extract_feature_dict
from repralign.metrics import compute_metric
from repralign.pooling import apply_pooling

adapter = GenericTorchAdapter(model, layer_names=["encoder.layers.0", "encoder.layers.1"])
raw = extract_feature_dict(adapter=adapter, batch=batch, pooling="mean_tokens")
score = compute_metric("cka", raw["encoder.layers.0"], raw["encoder.layers.1"])
```

## YAML Configs

The v0.1 CLI uses explicit YAML configuration so model factories, layer names, input tensors, pooling, and output directories are visible in one place. See [configs/example_analysis.yaml](/Users/jnmz/Desktop/code/repralign/configs/example_analysis.yaml) for a runnable example.

## Screenshot-Inspired Workflow

The attached screenshot should be treated as inspiration for:

- comparing a candidate model against more than one reference model
- organizing figures as layer-wise similarity curves
- structuring experiments around semantic-style and generation-style references

It should **not** be treated as a source of hidden implementation detail. This repository intentionally keeps model internals configurable and only relies on public, generic analysis primitives.

## TODO

- add token-level similarity analysis without pooling
- add dataset loaders beyond tensor files
- add richer multimodal adapters for custom architectures such as Show-o2
- add diffusion and DiT-style reference adapters
- add batched large-scale cache writing for very large datasets
- add confidence intervals and multi-run aggregation utilities

## License

MIT
