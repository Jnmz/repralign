# repralign

`repralign` is a reusable Python toolkit for layer-wise representation alignment analysis in multimodal models.

It is designed for workflows where you want to:

- extract intermediate features from candidate and reference models
- normalize or pool those features into comparable representations
- compute layer-wise similarity with metrics such as CKA and CKNNA
- save CSV, JSON, and PNG outputs for later inspection

## What It Supports Today

- forward-hook based feature extraction from configurable layers
- pooling modes: `cls`, `mean_tokens`, `flatten_mean`
- linear CKA and neighbor-overlap CKNNA
- batched extraction across tensor batches or image datasets
- semantic reference workflows through Hugging Face vision encoders
- generation-reference workflows through Diffusers pipelines such as SD3
- CLI commands for listing layers, extracting features, running analysis, and plotting

## What It Is Good For

`repralign` is a good fit when you want to run paper-style alignment studies such as:

- candidate model vs semantic reference model
- candidate model vs generation-oriented reference model
- layer-wise similarity curves across a representation encoder

## TUNA-Style Analysis Status

`repralign` can now support the **same class of analysis workflow** as the TUNA representation-alignment section:

- compare one candidate model against multiple reference models
- compute layer-wise similarity curves
- use semantic and generation-oriented references in the same experiment

What it does **not** provide automatically is a turnkey, official TUNA experiment package. To run a TUNA-like study in practice, you still need to supply:

- the actual candidate model adapter
- the exact layer selections you want to analyze
- the chosen semantic reference model
- the chosen generation reference model
- the dataset and preprocessing settings for your experiment

So the current status is:

- the analysis framework is ready
- a TUNA-style experiment is feasible
- exact paper-matching setup still depends on your model-specific configuration

## Start Here

- [Installation](getting-started/installation.md)
- [CLI Workflow](guides/cli-workflow.md)
- [YAML Configs](guides/yaml-configs.md)
- [Reference Models](guides/reference-models.md)
- [Real Image Folder Workflow](tutorials/real-image-workflow.md)
