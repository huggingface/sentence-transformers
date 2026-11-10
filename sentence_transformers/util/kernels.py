"""Helpers for configuring optional kernels."""

from __future__ import annotations

import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers.utils import is_kernels_available

if is_kernels_available():
    from transformers.integrations.hub_kernels import use_kernel_forward_from_hub
else:

    def use_kernel_forward_from_hub(name):
        return lambda module: module


if TYPE_CHECKING:
    from transformers import KernelConfig


def _split_kernel_config(config: KernelConfig, targets: set[str]) -> tuple[KernelConfig, KernelConfig]:
    from transformers import KernelConfig

    if not isinstance(config, KernelConfig):
        raise TypeError("kernel_config must be a transformers.KernelConfig")

    selected, remaining = deepcopy(config), deepcopy(config)
    selected.kernel_mapping = {name: entry for name, entry in selected.kernel_mapping.items() if name in targets}
    remaining.kernel_mapping = {name: entry for name, entry in remaining.kernel_mapping.items() if name not in targets}
    return selected, remaining


def _create_kernel_mapping(config: KernelConfig, device: torch.device) -> dict:
    from kernels import LayerRepository, LocalLayerRepository

    device_type = "rocm" if device.type == "cuda" and torch.version.hip else device.type
    mapping = {}

    for target, value in config.kernel_mapping.items():
        devices = {device_type: value} if isinstance(value, (str, tuple)) else value
        if not isinstance(devices, dict):
            raise TypeError(f"Kernel mapping for {target!r} must be a repository or a device mapping")

        mapping[target] = {}
        for backend, entry in devices.items():
            if isinstance(entry, (str, tuple)):
                repository, metadata = (entry, {}) if isinstance(entry, str) else entry
                repo_id, separator, layer_name = repository.rpartition(":")
                if not separator or not repo_id or not layer_name:
                    raise ValueError(f"Expected repository:layer_name for {target!r}, got {repository!r}")

                if config.use_local_kernel:
                    entry = LocalLayerRepository(repo_path=Path(repo_id), layer_name=layer_name)
                else:
                    metadata = dict(metadata)
                    if metadata.get("version") is None and metadata.get("revision") is None:
                        metadata["version"] = 1

                    entry = LayerRepository(repo_id=repo_id, layer_name=layer_name, **metadata)

            mapping[target][backend] = entry

    return mapping


def _configure_kernel(
    kernel_layers: list[nn.Module],
    config: KernelConfig | None,
    device: torch.device,
    *,
    training: bool = False,
    defaults: dict[str, tuple[str, str, int]] | None = None,
    trust_remote_code: bool | list[str] = False,
) -> None:
    """Kernelize the supplied kernel-aware layers in place."""
    from transformers import KernelConfig

    if config is None:
        config = KernelConfig()
    if not isinstance(config, KernelConfig):
        raise TypeError("kernel_config must be a transformers.KernelConfig")
    if os.getenv("USE_HUB_KERNELS", "YES").upper() in {"0", "OFF", "NO", "FALSE"}:
        return
    if not is_kernels_available():
        raise ImportError('Install compatible kernel support with pip install "sentence-transformers[kernels]"')

    from kernels import LayerRepository, Mode
    from transformers.integrations.hub_kernels import kernelize

    target_names = {layer.kernel_layer_name for layer in kernel_layers}
    unknown_targets = config.kernel_mapping.keys() - target_names
    if unknown_targets:
        raise ValueError(f"Unknown kernel targets: {sorted(unknown_targets)}. Available: {sorted(target_names)}")

    mapping = {}
    if config.inherit_mapping and not config.use_local_kernel:
        mapping = {
            target: {
                "cuda": LayerRepository(repo, layer_name=layer, version=version, trust_remote_code=trust_remote_code),
            }
            for target, (repo, layer, version) in (defaults or {}).items()
        }

    mapping.update(_create_kernel_mapping(config, device))

    # Parameterless layers need an explicit device for Transformers' kernel loader.
    view = nn.ModuleList(kernel_layers)
    view.device = device
    view.train(training)
    view.kernel_config = KernelConfig(mapping, inherit_mapping=False)

    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        kernelize(view, mode=Mode.TRAINING if training else Mode.INFERENCE)
