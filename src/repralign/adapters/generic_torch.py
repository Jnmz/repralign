"""Generic PyTorch adapter."""

from __future__ import annotations

from typing import Any

import torch

from repralign.adapters.base import BaseModelAdapter
from repralign.datasets import batch_to_images


class GenericTorchAdapter(BaseModelAdapter):
    """Generic adapter where the caller controls model inputs and layer names."""

    def prepare_batch(self, batch: object) -> object:
        if self.processor is None:
            return batch

        if isinstance(batch, dict):
            return self.processor(**batch)
        if isinstance(batch, (list, tuple)):
            images = batch_to_images(batch)
            if images is not None:
                return self.processor(images=images, return_tensors="pt")
        return self.processor(batch)

    def forward(self, batch: object) -> object:
        if isinstance(batch, dict):
            return self.model(**batch)
        if isinstance(batch, (tuple, list)):
            return self.model(*batch)
        return self.model(batch)

    def transform_hook_output(self, output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0].detach()
        raise TypeError(f"Unsupported generic hook output type: {type(output)!r}")
