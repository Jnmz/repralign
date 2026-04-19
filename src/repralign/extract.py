"""Feature extraction pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch

from repralign.adapters.base import BaseModelAdapter
from repralign.hooks import register_activation_hooks
from repralign.pooling import PoolingMode, apply_pooling, l2_normalize


def extract_feature_dict(
    adapter: BaseModelAdapter,
    batch: object,
    pooling: PoolingMode,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, torch.Tensor]:
    """Extract pooled features from the configured layers of a model.

    Parameters
    ----------
    adapter:
        Model adapter describing how to run the forward pass.
    batch:
        Input batch accepted by ``adapter.forward``.
    pooling:
        Pooling mode applied to each captured layer output.
    normalize:
        If ``True``, apply row-wise L2 normalization after pooling.
    device:
        Optional device to move the model and batch onto before extraction.
    """

    if device is not None:
        adapter.move_to_device(device)

    prepared_batch = adapter.prepare_batch(batch)
    if device is not None:
        prepared_batch = move_batch_to_device(prepared_batch, device)

    hooks = register_activation_hooks(
        adapter.get_hook_model(),
        adapter.layer_names,
        transform=adapter.transform_hook_output,
    )
    hook_model = adapter.get_hook_model()
    was_training = hook_model.training
    hook_model.eval()

    try:
        with torch.no_grad():
            adapter.forward(prepared_batch)
        pooled: dict[str, torch.Tensor] = {}
        for layer_name in adapter.layer_names:
            layer_features = apply_pooling(hooks.activations[layer_name], pooling)
            pooled[layer_name] = l2_normalize(layer_features) if normalize else layer_features
        return pooled
    finally:
        hooks.remove()
        hook_model.train(was_training)


def extract_feature_dataset(
    adapter: BaseModelAdapter,
    batches: Iterable[object],
    pooling: PoolingMode,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, torch.Tensor]:
    """Extract pooled features across multiple batches and concatenate them."""

    collected: dict[str, list[torch.Tensor]] = {layer_name: [] for layer_name in adapter.layer_names}
    for batch in batches:
        batch_features = extract_feature_dict(
            adapter=adapter,
            batch=batch,
            pooling=pooling,
            normalize=normalize,
            device=device,
        )
        for layer_name, tensor in batch_features.items():
            collected[layer_name].append(tensor.detach().cpu())

    return {
        layer_name: torch.cat(tensors, dim=0) if tensors else torch.empty((0, 0), dtype=torch.float32)
        for layer_name, tensors in collected.items()
    }


def move_batch_to_device(batch: Any, device: Union[str, torch.device]) -> Any:
    """Recursively move a nested batch structure onto a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch
