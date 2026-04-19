# YAML Configs

`repralign` is intentionally config-driven. The YAML file controls:

- model factories
- adapter choice
- processor factories
- adapter-specific arguments
- layer names
- dataset source
- pooling and normalization
- output location

## Tensor-Batch Style Config

See:

- `configs/example_analysis.yaml`

This format is useful for:

- toy models
- synthetic tests
- preconstructed tensor batches

## Dataset-Driven Config

See:

- `configs/reference_alignment_template.yaml`

This format is useful for:

- image folders
- CSV/JSONL manifests
- semantic and generation references in the same experiment

## Model Spec Fields

Example:

```yaml
name: semantic_reference
adapter: hf_vision
factory: repralign.factories:load_siglip_vision_model
processor_factory: repralign.factories:load_siglip_image_processor
factory_kwargs:
  model_name: google/siglip-so400m-patch14-384
processor_kwargs:
  model_name: google/siglip-so400m-patch14-384
layers:
  - vision_model.encoder.layers.0
device: cpu
```

Important fields:

- `adapter`: adapter registry key
- `factory`: model factory in `module:function` form
- `processor_factory`: optional preprocessor factory
- `factory_kwargs`: passed into the model factory
- `processor_kwargs`: passed into the processor factory
- `adapter_kwargs`: adapter-specific behavior
- `layers`: named modules to hook
- `device`: extraction device for that model
