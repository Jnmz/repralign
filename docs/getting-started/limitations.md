# Limitations

`repralign` is usable for real experiments, but it is still an early research toolkit.

## Current Practical Limits

- token-level alignment analysis is not yet first-class
- very large datasets may need more efficient on-disk feature accumulation
- generation-reference workflows depend on the target pipeline exposing stable internal modules
- exact paper-matching experiments still require model-specific configuration and validation

## Important Interpretation Note

`repralign` helps you run the same **kind** of alignment study used in multimodal papers. It does not imply that a result matches any published paper unless:

- the same candidate model is used
- the same reference models are used
- the same dataset and preprocessing are used
- the same layer selections and metric settings are used

## SD3 Reference Caveat

The SD3 adapter is designed to make generation-oriented reference analysis practical, but real SD3 experiments still depend on:

- available hardware
- Diffusers version compatibility
- prompt-handling choices
- selected timestep and image encoding settings
