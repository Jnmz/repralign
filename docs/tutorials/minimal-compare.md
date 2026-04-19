# Minimal Comparison

Run the bundled toy example:

```bash
python examples/minimal_compare.py
```

This tutorial demonstrates the smallest end-to-end workflow:

- build two toy vision models
- hook candidate and reference layers
- extract pooled features
- compute layer-wise CKA
- save CSV and PNG outputs

Outputs are written to:

- `outputs/minimal_compare/similarity.csv`
- `outputs/minimal_compare/similarity.png`
