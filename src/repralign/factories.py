"""Predefined model and processor factories for common reference models."""

from __future__ import annotations

from typing import Any, Optional

import torch


def _resolve_torch_dtype(dtype: Optional[str]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype}")
    resolved = getattr(torch, dtype)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported torch dtype string: {dtype}")
    return resolved


def load_siglip_vision_model(model_name: str, torch_dtype: Optional[str] = None, **kwargs: Any) -> Any:
    """Load a SigLIP vision model from Transformers."""

    from transformers import SiglipVisionModel

    dtype = _resolve_torch_dtype(torch_dtype)
    return SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def load_hf_image_processor(model_name: str, **kwargs: Any) -> Any:
    """Load a Hugging Face image processor."""

    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(model_name, **kwargs)


def load_siglip_image_processor(model_name: str, **kwargs: Any) -> Any:
    """Load a SigLIP image processor."""

    from transformers import SiglipImageProcessor

    return SiglipImageProcessor.from_pretrained(model_name, **kwargs)


def load_sd3_pipeline(
    model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype: Optional[str] = None,
    drop_t5: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a Stable Diffusion 3 pipeline from Diffusers."""

    from diffusers import StableDiffusion3Pipeline

    dtype = _resolve_torch_dtype(torch_dtype)
    if drop_t5:
        kwargs.setdefault("text_encoder_3", None)
        kwargs.setdefault("tokenizer_3", None)
    return StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
