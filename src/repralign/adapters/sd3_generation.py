"""Stable Diffusion 3 generation reference adapter."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn

from repralign.adapters.base import BaseModelAdapter
from repralign.datasets import batch_to_image_records


class StableDiffusion3ReferenceAdapter(BaseModelAdapter):
    """Adapter for using the SD3 transformer as a generation-oriented reference.

    The wrapped ``model`` is expected to be a Diffusers ``StableDiffusion3Pipeline``.
    Hooks are registered on ``pipeline.transformer`` while prompt encoding and VAE
    image encoding are coordinated through the full pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: list[str],
        processor: Optional[Any] = None,
        adapter_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model, layer_names=layer_names, processor=processor, adapter_kwargs=adapter_kwargs)
        self.pipeline = model

    def get_hook_model(self) -> nn.Module:
        return self.pipeline.transformer

    def move_to_device(self, device: Union[str, torch.device]) -> None:
        self.pipeline.to(device)

    def prepare_batch(self, batch: object) -> object:
        records = batch_to_image_records(batch)
        if records is None:
            raise TypeError("SD3 generation adapter expects batches of image records")
        return records

    def forward(self, batch: object) -> object:
        records = self.prepare_batch(batch)
        device = getattr(self.pipeline, "_execution_device", None) or next(self.pipeline.transformer.parameters()).device
        image_tensor = self._prepare_images(records, device=device)
        latents = self._encode_images(image_tensor)
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(records, device=device)
        timestep = self._build_timestep(latents.shape[0], device=device)
        return self.pipeline.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timestep,
            return_dict=True,
        )

    def transform_hook_output(self, output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output.detach()
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0].detach()
        if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
            return output.sample.detach()
        raise TypeError(f"Unsupported SD3 hook output type: {type(output)!r}")

    def _prepare_images(self, records: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
        height = int(self.adapter_kwargs.get("height", 512))
        width = int(self.adapter_kwargs.get("width", 512))
        tensors = []
        for record in records:
            image = record["image"]
            if not isinstance(image, Image.Image):
                raise TypeError("SD3 generation adapter expects PIL images inside batch records")
            resized = image.convert("RGB").resize((width, height))
            array = np.asarray(resized).astype("float32") / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)

        image_tensor = torch.stack(tensors, dim=0).to(device=device)
        image_tensor = image_tensor * 2.0 - 1.0
        dtype = next(self.pipeline.vae.parameters()).dtype
        return image_tensor.to(dtype=dtype)

    def _encode_images(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.pipeline.vae.encode(image_tensor)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is None:
            raise AttributeError("VAE encode result does not expose latent_dist")
        if hasattr(latent_dist, "mean"):
            latents = latent_dist.mean
        else:
            latents = latent_dist.sample()
        scaling_factor = float(getattr(self.pipeline.vae.config, "scaling_factor", 1.0))
        shift_factor = float(getattr(self.pipeline.vae.config, "shift_factor", 0.0))
        return (latents - shift_factor) * scaling_factor

    def _encode_prompts(self, records: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        default_prompt = str(self.adapter_kwargs.get("default_prompt", ""))
        prompt_key = str(self.adapter_kwargs.get("prompt_key", "prompt"))
        prompts = [str(record.get(prompt_key, default_prompt)) for record in records]
        prompt_2 = [str(record.get("prompt_2", prompt)) for record, prompt in zip(records, prompts)]
        prompt_3 = [str(record.get("prompt_3", prompt)) for record, prompt in zip(records, prompts)]

        encoded = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=int(self.adapter_kwargs.get("max_sequence_length", 256)),
        )
        if not isinstance(encoded, tuple) or len(encoded) < 3:
            raise TypeError("Unexpected return structure from StableDiffusion3Pipeline.encode_prompt")
        prompt_embeds = encoded[0]
        pooled_prompt_embeds = encoded[2]
        return prompt_embeds, pooled_prompt_embeds

    def _build_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        timestep = int(self.adapter_kwargs.get("timestep", 0))
        return torch.full((batch_size,), timestep, device=device, dtype=torch.long)
