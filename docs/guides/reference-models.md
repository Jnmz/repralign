# Reference Models

## Semantic References

The built-in semantic path is based on Hugging Face vision encoders.

Current built-in factory helpers:

- `repralign.factories:load_siglip_vision_model`
- `repralign.factories:load_siglip_image_processor`
- `repralign.factories:load_hf_image_processor`

This path is a good fit for models such as:

- SigLIP
- ViT-style image encoders
- other HF vision backbones with stable named modules

## Generation References

The built-in generation path is based on Diffusers pipelines.

Current built-in factory helper:

- `repralign.factories:load_sd3_pipeline`

Current adapter:

- `sd3_reference`

This path is designed to:

- encode images through the VAE
- encode prompts through the pipeline text stack
- hook the SD3 transformer blocks
- treat the denoising transformer as the generation-oriented representation source

## Important Note About SD3-Medium

The repository now includes a practical SD3 reference adapter, but the exact SD3-Medium setup used in any paper still depends on experiment choices such as:

- prompt text
- timestep
- image resolution
- selected hooked transformer blocks
- Diffusers version and model weights

So the codebase now supports SD3-oriented reference analysis, but you still need to decide the exact experiment protocol.
