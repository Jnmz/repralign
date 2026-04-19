# Real Image Folder Workflow

This is the path to use when moving beyond toy tensors.

## 1. Start From The Template

Use:

- `configs/reference_alignment_template.yaml`

## 2. Fill In The Candidate Model

Replace the toy candidate factory with your real candidate model factory and layer names.

## 3. Choose Reference Models

Typical pattern:

- semantic reference: SigLIP or another HF vision encoder
- generation reference: SD3-Medium through the `sd3_reference` adapter

## 4. Point To Your Images

```yaml
dataset:
  type: image_folder
  image_root: /path/to/images
  recursive: true
  batch_size: 2
  prompt: ""
```

If your generation reference needs meaningful per-image text, use a CSV or JSONL manifest instead.

## 5. Extract Features

```bash
repralign extract-features --config configs/reference_alignment_template.yaml
```

## 6. Analyze Against Each Reference

```bash
repralign analyze \
  --candidate-cache outputs/reference_alignment/candidate_features.npz \
  --reference-cache outputs/reference_alignment/semantic_reference_features.npz \
  --metric cknna \
  --output-csv outputs/reference_alignment/semantic_cknna.csv \
  --output-json outputs/reference_alignment/semantic_cknna.json
```

Repeat for the generation reference cache.

## 7. Plot Curves

```bash
repralign plot \
  --input-csv outputs/reference_alignment/semantic_cknna.csv \
  --input-csv outputs/reference_alignment/generation_cknna.csv \
  --output-png outputs/reference_alignment/cknna_comparison.png \
  --title "Representation Alignment"
```

## Practical Readiness For TUNA-Style Analysis

With a real candidate-model adapter in place, this workflow is enough to run the same category of analysis used in the TUNA representation-alignment section:

- compare candidate layers against semantic references
- compare the same candidate layers against generation references
- visualize the resulting layer-wise curves

What still remains experiment-specific is the exact choice of:

- candidate model hooks
- reference model checkpoints
- prompts
- dataset
- layer subsets
- metric settings
