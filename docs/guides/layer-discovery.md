# Layer Discovery

Choosing the right layers is one of the most important parts of a useful alignment experiment.

## List Named Modules

Use the CLI to inspect all named modules exposed by a model factory:

```bash
repralign list-layers \
  --factory examples.model_factories:build_toy_candidate \
  --adapter generic_torch
```

For real models, replace the toy factory with your actual model factory.

## What To Hook

In practice, the best hook points depend on the model family:

- vision backbones: encoder blocks or transformer layers
- multimodal encoders: representation-encoder blocks or fusion blocks
- diffusion references: transformer blocks in the denoising stack

## Good Default Strategy

If you are not sure where to start:

1. choose a contiguous run of internal layers rather than isolated layers
2. avoid very shallow preprocessing modules unless they are meaningful for your question
3. keep the candidate and reference analysis focused on representation-bearing layers

## Common Mistakes

- hooking modules that return tuples you did not inspect
- mixing token-level and pooled representations without realizing it
- comparing layers from mismatched preprocessing pipelines
