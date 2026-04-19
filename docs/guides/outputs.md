# Outputs

`repralign` writes several output types during an experiment.

## Feature Caches

Extension:

- `.npz`

These contain pooled feature matrices keyed by layer name. They are intended to avoid repeated forward passes when you want to recompute metrics or regenerate plots.

## Metadata

Extension:

- `.json`

Metadata files store information such as:

- model name
- adapter name
- selected layers
- pooling mode
- normalization choice
- dataset settings when present

## Similarity Tables

Extension:

- `.csv`

These tables are the main intermediate artifact for layer-wise analysis. Each row typically contains:

- metric name
- candidate layer index
- candidate layer name
- reference layer name
- score

## Plots

Extension:

- `.png`

The plotting command renders one or more similarity curves into a publication-style figure.
