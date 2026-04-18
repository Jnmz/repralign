"""Generic PyTorch adapter."""

from __future__ import annotations

import torch
from torch import nn

from repralign.adapters.base import BaseModelAdapter


class GenericTorchAdapter(BaseModelAdapter):
    """Generic adapter where the caller controls model inputs and layer names."""

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
