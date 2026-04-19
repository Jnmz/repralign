"""Configuration loading for YAML-driven workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml


@dataclass
class ModelSpec:
    name: str
    adapter: str
    factory: str
    layers: list[str]
    factory_kwargs: dict[str, Any] = field(default_factory=dict)
    processor_factory: Optional[str] = None
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None


@dataclass
class ExtractSpec:
    pooling: str
    normalize: bool
    batch_path: Optional[str] = None
    batch_key: Optional[str] = None


@dataclass
class OutputSpec:
    directory: str


@dataclass
class DatasetSpec:
    type: str = "image_folder"
    image_root: Optional[str] = None
    manifest_path: Optional[str] = None
    image_key: str = "image"
    prompt_key: str = "prompt"
    prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    recursive: bool = False
    patterns: list[str] = field(default_factory=lambda: ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"])
    batch_size: int = 8
    limit: Optional[int] = None


@dataclass
class AnalysisConfig:
    candidate: ModelSpec
    references: list[ModelSpec]
    extract: ExtractSpec
    outputs: OutputSpec
    dataset: Optional[DatasetSpec] = None


def load_yaml_config(path: Union[str, Path]) -> AnalysisConfig:
    """Load the top-level experiment config."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    candidate = ModelSpec(**raw["candidate"])
    references = [ModelSpec(**item) for item in raw["references"]]
    extract = ExtractSpec(**raw["extract"])
    outputs = OutputSpec(**raw["outputs"])
    dataset = DatasetSpec(**raw["dataset"]) if "dataset" in raw else None
    return AnalysisConfig(candidate=candidate, references=references, extract=extract, outputs=outputs, dataset=dataset)


def resolve_factory(factory_path: str) -> Any:
    """Resolve a ``module:function`` factory path."""

    module_name, function_name = factory_path.split(":", maxsplit=1)
    module = import_module(module_name)
    return getattr(module, function_name)


def load_batch_tensor(path: Union[str, Path], batch_key: Optional[str] = None) -> object:
    """Load a tensor batch from a ``.pt`` file."""

    data = torch.load(Path(path), map_location="cpu")
    if batch_key is None:
        return data
    if not isinstance(data, dict):
        raise TypeError("batch_key was provided, but the loaded object is not a dict")
    return data[batch_key]
