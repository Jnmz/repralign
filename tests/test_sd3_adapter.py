from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch import nn

from repralign.adapters.sd3_generation import StableDiffusion3ReferenceAdapter
from repralign.extract import extract_feature_dataset


class FakeLatentDist:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean


class FakeEncodeResult:
    def __init__(self, mean: torch.Tensor) -> None:
        self.latent_dist = FakeLatentDist(mean=mean)


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"scaling_factor": 1.0, "shift_factor": 0.0})()

    def encode(self, image_tensor: torch.Tensor) -> FakeEncodeResult:
        latents = image_tensor.mean(dim=1, keepdim=True)
        return FakeEncodeResult(mean=latents)


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FakeTransformerOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool = True,
    ) -> FakeTransformerOutput:
        x = hidden_states
        for block in self.blocks:
            x = block(x)
        bias = encoder_hidden_states.mean().view(1, 1, 1, 1) + pooled_projections.mean().view(1, 1, 1, 1)
        sample = x + bias + timestep.float().view(-1, 1, 1, 1)
        return FakeTransformerOutput(sample=sample)


class FakeSD3Pipeline:
    def __init__(self) -> None:
        self.transformer = FakeTransformer()
        self.vae = FakeVAE()
        self._execution_device = torch.device("cpu")

    def to(self, device: Union[str, torch.device]) -> "FakeSD3Pipeline":
        self.transformer.to(device)
        self.vae.to(device)
        self._execution_device = torch.device(device)
        return self

    def encode_prompt(
        self,
        prompt: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, None, torch.Tensor, None]:
        batch = len(prompt)
        prompt_embeds = torch.ones(batch, 4, 3, device=device)
        pooled = torch.ones(batch, 4, device=device)
        return prompt_embeds, None, pooled, None


def test_sd3_reference_adapter_extracts_features(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(image_path)

    pipeline = FakeSD3Pipeline()
    adapter = StableDiffusion3ReferenceAdapter(
        model=pipeline,
        layer_names=["blocks.0", "blocks.1"],
        adapter_kwargs={"height": 16, "width": 16, "timestep": 1, "default_prompt": "test"},
    )

    records = [{"id": "sample", "image_path": str(image_path), "prompt": "a test prompt"}]
    features = extract_feature_dataset(adapter=adapter, batches=[records], pooling="flatten_mean", normalize=False)

    assert features["blocks.0"].shape == (1, 1)
    assert features["blocks.1"].shape == (1, 1)
