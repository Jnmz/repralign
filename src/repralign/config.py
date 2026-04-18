"""Configuration loading for YAML-driven workflows."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ExtractSpec:
    pooling: str
    normalize: bool
    batch_path: str
    batch_key: Optional[str] = None


@dataclass
class OutputSpec:
    directory: str


@dataclass
class AnalysisConfig:
    candidate: ModelSpec
    references: list[ModelSpec]
    extract: ExtractSpec
    outputs: OutputSpec


def load_yaml_config(path: Union[str, Path]) -> AnalysisConfig:
    """Load the top-level experiment config."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    candidate = ModelSpec(**raw["candidate"])
    references = [ModelSpec(**item) for item in raw["references"]]
    extract = ExtractSpec(**raw["extract"])
    outputs = OutputSpec(**raw["outputs"])
    return AnalysisConfig(candidate=candidate, references=references, extract=extract, outputs=outputs)


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
