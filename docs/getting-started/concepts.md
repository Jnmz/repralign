# Concepts

## Candidate Model

The model whose internal representations you want to study.

## Reference Model

A model used as an external comparison target. In `repralign`, a reference model is typically used in one of two roles:

- semantic reference
- generation-oriented reference

## Layer-Wise Alignment

For each chosen candidate layer, `repralign` extracts features, compares them against one or more reference features, and writes the resulting scores as a curve across depth.

## Pooling

Many hooked layers produce token-level or spatial features. `repralign` reduces these to sample-by-feature matrices using one of:

- `cls`
- `mean_tokens`
- `flatten_mean`

## Metrics

### CKA

Linear CKA measures whether two feature spaces preserve similar pairwise structure across samples.

### CKNNA

CKNNA in `repralign` measures agreement between top-`k` nearest-neighbor structure induced by two feature spaces.
