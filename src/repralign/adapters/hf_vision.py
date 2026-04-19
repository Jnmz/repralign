"""Hugging Face vision encoder adapter."""

from __future__ import annotations

import torch

from repralign.adapters.base import BaseModelAdapter
from repralign.datasets import batch_to_images


class HuggingFaceVisionAdapter(BaseModelAdapter):
    """Adapter for ViT- and SigLIP-like vision backbones from Hugging Face.

    This adapter intentionally stays generic: callers must still specify the
    layer names to inspect. The only model-specific behavior here is that hook
    outputs and model forward inputs follow common Hugging Face conventions.
    """

    def prepare_batch(self, batch: object) -> object:
        if self.processor is None:
            if isinstance(batch, dict):
                return batch
            raise TypeError("Hugging Face vision extraction requires a processor when using raw image batches")

        if isinstance(batch, dict):
            return batch
        images = batch_to_images(batch)
        if images is None:
            raise TypeError("Hugging Face vision adapter expects a batch of image records or a tensor dict")
        return self.processor(images=images, return_tensors="pt")

    def forward(self, batch: object) -> object:
        if not isinstance(batch, dict):
            raise TypeError("Hugging Face vision models expect a dict batch, usually with pixel_values")
        return self.model(**batch)

    def transform_hook_output(self, output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0].detach()
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
            return output.last_hidden_state.detach()
        raise TypeError(f"Unsupported Hugging Face hook output type: {type(output)!r}")
