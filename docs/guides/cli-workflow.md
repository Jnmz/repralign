# CLI Workflow

The CLI entrypoint is `repralign`.

## List Layers

Use this to inspect named modules before deciding which layers to hook.

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

## Extract Features

```bash
repralign extract-features --config configs/example_analysis.yaml
```

This command:

- loads the candidate and reference model configs
- extracts pooled features from the requested layers
- saves `.npz` feature caches
- writes JSON metadata next to those caches

## Analyze Similarity

```bash
repralign analyze \
  --candidate-cache outputs/paper_like/candidate_features.npz \
  --reference-cache outputs/paper_like/semantic_reference_features.npz \
  --metric cka \
  --output-csv outputs/paper_like/cka_semantic.csv \
  --output-json outputs/paper_like/cka_semantic.json
```

For CKNNA:

```bash
repralign analyze \
  --candidate-cache outputs/paper_like/candidate_features.npz \
  --reference-cache outputs/paper_like/semantic_reference_features.npz \
  --metric cknna \
  --k 5 \
  --output-csv outputs/paper_like/cknna_semantic.csv \
  --output-json outputs/paper_like/cknna_semantic.json
```

## Plot Curves

```bash
repralign plot \
  --input-csv outputs/paper_like/cka_semantic.csv \
  --input-csv outputs/paper_like/cka_generation.csv \
  --output-png outputs/paper_like/cka_comparison.png \
  --title "Layer-wise Alignment"
```
