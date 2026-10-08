"""Native model merging, with no external dependencies.

Every input model is treated as a stack of weight directories: the transformer
body at the root plus the weight-bearing Sentence Transformers modules (``Dense``,
``LayerNorm``, ...) in their subfolders. Each directory's ``state_dict`` is merged
with the chosen method. Stateless modules (``Pooling``, ``Normalize``, ...) and
all sidecar configs are copied from the first model after an equality check.

The merge algorithms operate purely on ``state_dict`` arithmetic, so nothing here
depends on the underlying architecture.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file as save_safetensors_file

from sentence_transformers.util import load_dir_path, load_file_path

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)


SUPPORTED_METHODS: tuple[str, ...] = (
    "linear",
    "slerp",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_linear",
)
"""Natively implemented merge methods."""

REQUIRES_BASE_MODEL: frozenset[str] = frozenset(
    {"task_arithmetic", "ties", "dare_ties", "dare_linear"}
)
"""Methods that merge deltas relative to an explicit ``base_model``."""

TWO_MODEL_METHODS: frozenset[str] = frozenset({"slerp"})
"""Methods that operate on exactly two input models."""

DELTA_BASED_METHODS: frozenset[str] = frozenset(
    {"task_arithmetic", "ties", "dare_ties", "dare_linear"}
)
"""Methods whose weights scale per-model deltas (default 1.0) rather than a simplex."""

DENSITY_METHODS: frozenset[str] = frozenset({"ties", "dare_ties", "dare_linear"})
"""Methods that sparsify each delta by a per-model ``density``."""

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
_WEIGHT_PREFIXES = ("model.", "model-", "pytorch_model.", "pytorch_model-")

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _hub_kwargs_from(load_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "token": load_kwargs.get("token"),
        "cache_folder": load_kwargs.get("cache_folder"),
        "revision": load_kwargs.get("revision"),
        "local_files_only": load_kwargs.get("local_files_only", False),
    }


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_modules_config(model_id: str, **hub_kwargs: Any) -> list[dict[str, Any]]:
    # Plain HF checkpoints (e.g. BAAI/bge-reranker-*) have no modules.json. Treat
    # them as body-only: the empty list makes the pipeline skip the module loop.
    path = load_file_path(model_name_or_path=model_id, filename="modules.json", **hub_kwargs)
    if path is None:
        return []
    return _read_json(path)


def _resolve_module_dir(model_id: str, subfolder: str, **hub_kwargs: Any) -> str | None:
    # load_dir_path can return a non-existent path string (the Hub doesn't track
    # empty dirs, e.g. for a no-op-save module like Normalize), hence the isdir check.
    try:
        path = load_dir_path(model_name_or_path=model_id, subfolder=subfolder, **hub_kwargs)
    except (FileNotFoundError, OSError):
        return None
    if path is None or not os.path.isdir(path):
        return None
    return path


def _has_weight_files(directory: str) -> bool:
    return any(os.path.exists(os.path.join(directory, name)) for name in _WEIGHT_FILES) or any(
        os.path.exists(os.path.join(directory, name))
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    )


def _is_empty_module_dir(directory: str | None) -> bool:
    # File-less stateless modules (e.g. Normalize) have no config and no weights;
    # the Hub doesn't track their empty dir, so they resolve to None for everyone.
    if directory is None:
        return True
    return not (os.path.exists(os.path.join(directory, "config.json")) or _has_weight_files(directory))


def _load_state_dict(directory: str) -> OrderedDict[str, torch.Tensor]:
    """Load a (possibly sharded) safetensors or pytorch state dict from a directory."""
    from safetensors.torch import load_file

    safetensors_index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(safetensors_index):
        weight_map = _read_json(safetensors_index)["weight_map"]
        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(os.path.join(directory, shard)))
        return state

    safetensors_path = os.path.join(directory, "model.safetensors")
    if os.path.exists(safetensors_path):
        return OrderedDict(load_file(safetensors_path))

    pytorch_index = os.path.join(directory, "pytorch_model.bin.index.json")
    if os.path.exists(pytorch_index):
        weight_map = _read_json(pytorch_index)["weight_map"]
        state = OrderedDict()
        for shard in sorted(set(weight_map.values())):
            state.update(torch.load(os.path.join(directory, shard), map_location="cpu", weights_only=True))
        return state

    pytorch_path = os.path.join(directory, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return OrderedDict(torch.load(pytorch_path, map_location="cpu", weights_only=True))

    raise FileNotFoundError(f"No model weights found in {directory!r}.")


def _save_state_dict(
    state_dict: OrderedDict[str, torch.Tensor], directory: str, dtype: torch.dtype
) -> None:
    os.makedirs(directory, exist_ok=True)
    # A single merged shard replaces any sharded/pytorch layout the scaffold left behind.
    for stale in ("model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        stale_path = os.path.join(directory, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)
    cast = {k: (v.to(dtype) if v.is_floating_point() else v).contiguous() for k, v in state_dict.items()}
    save_safetensors_file(cast, os.path.join(directory, "model.safetensors"))


# --------------------------------------------------------------------------- #
# Merge algorithms (operate on lists of state dicts)
# --------------------------------------------------------------------------- #
def _stack(key: str, state_dicts: Sequence[OrderedDict[str, torch.Tensor]], device: torch.device) -> torch.Tensor:
    base_shape = state_dicts[0][key].shape
    tensors = []
    for i, sd in enumerate(state_dicts):
        if key not in sd:
            raise KeyError(f"Key {key!r} present in model 0 but missing in model {i}.")
        t = sd[key]
        if t.shape != base_shape:
            raise ValueError(
                f"Shape mismatch for key {key!r}: model[0]={tuple(base_shape)}, model[{i}]={tuple(t.shape)}"
            )
        tensors.append(t.to(device=device, dtype=torch.float32))
    return torch.stack(tensors, dim=0)


def _weight_view(weights: Sequence[float], ndim: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(weights, dtype=torch.float32, device=device).view((-1,) + (1,) * (ndim - 1))


def _merge_linear(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]], weights: Sequence[float], device: torch.device
) -> OrderedDict[str, torch.Tensor]:
    weight_sum = float(sum(weights))
    if weight_sum == 0:
        raise ValueError("Sum of weights must be non-zero.")
    norm = [w / weight_sum for w in weights]
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, base in state_dicts[0].items():
        if not base.is_floating_point():
            out[key] = base.clone()
            continue
        stacked = _stack(key, state_dicts, device)
        merged = (stacked * _weight_view(norm, stacked.dim(), device)).sum(dim=0)
        out[key] = merged.cpu()
    return out


def _slerp_tensor(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8, dot_threshold: float = 0.9995) -> torch.Tensor:
    v0f, v1f = v0.flatten(), v1.flatten()
    n0, n1 = v0f.norm(), v1f.norm()
    if n0 < eps or n1 < eps:
        return torch.lerp(v0, v1, t)
    dot = torch.dot(v0f / n0, v1f / n1).clamp(-1.0, 1.0)
    if dot.abs() > dot_threshold:
        # Nearly (anti)colinear: slerp degenerates, fall back to linear interpolation.
        return torch.lerp(v0, v1, t)
    theta_0 = torch.acos(dot)
    sin_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    s0 = torch.sin(theta_0 - theta_t) / sin_0
    s1 = torch.sin(theta_t) / sin_0
    return (s0 * v0f + s1 * v1f).reshape(v0.shape)


def _merge_slerp(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]], weights: Sequence[float], device: torch.device
) -> OrderedDict[str, torch.Tensor]:
    t = float(weights[1])  # interpolation factor from model[0] (t=0) towards model[1] (t=1)
    a, b = state_dicts[0], state_dicts[1]
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, base in a.items():
        if not base.is_floating_point():
            out[key] = base.clone()
            continue
        if key not in b:
            raise KeyError(f"Key {key!r} present in model 0 but missing in model 1.")
        merged = _slerp_tensor(t, base.to(device=device, dtype=torch.float32), b[key].to(device=device, dtype=torch.float32))
        out[key] = merged.cpu()
    return out


def _trim(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top ``density`` fraction of ``delta`` by magnitude, zero the rest (TIES)."""
    if density >= 1.0:
        return delta
    flat = delta.abs().flatten()
    k = int(flat.numel() * density)
    if k <= 0:
        return torch.zeros_like(delta)
    threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
    return delta * (delta.abs() >= threshold)


def _dare_drop(delta: torch.Tensor, density: float, generator: torch.Generator) -> torch.Tensor:
    """Randomly drop each delta element with prob ``1-density``, rescale survivors by ``1/density`` (DARE)."""
    if density <= 0.0:
        return torch.zeros_like(delta)  # drop everything; avoids a 0/0 NaN in the rescale below
    if density >= 1.0:
        return delta
    mask = torch.rand(delta.shape, generator=generator, device=delta.device) < density
    return delta * mask / density


def _merge_delta(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]],
    weights: Sequence[float],
    base_state: OrderedDict[str, torch.Tensor],
    method: str,
    densities: Sequence[float],
    device: torch.device,
    generator: torch.Generator,
) -> OrderedDict[str, torch.Tensor]:
    """task_arithmetic / ties / dare_* : merge per-model deltas on top of a base."""
    sparsify = method in DENSITY_METHODS
    is_dare = method in ("dare_ties", "dare_linear")
    elect_sign = method in ("ties", "dare_ties")

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, base in base_state.items():
        if not base.is_floating_point():
            out[key] = base.clone()
            continue
        base_f = base.to(device=device, dtype=torch.float32)
        deltas, used_weights = [], []
        for i, (sd, w) in enumerate(zip(state_dicts, weights)):
            if key not in sd:
                continue
            delta = sd[key].to(device=device, dtype=torch.float32) - base_f
            if sparsify:
                delta = _dare_drop(delta, densities[i], generator) if is_dare else _trim(delta, densities[i])
            deltas.append(delta)
            used_weights.append(w)
        if not deltas:
            out[key] = base.clone()
            continue

        stacked = torch.stack(deltas, dim=0)
        weight_view = _weight_view(used_weights, stacked.dim(), device)
        if elect_sign:
            # TIES disjoint merge: the elected delta is the (weighted) MEAN over the
            # models whose sign agrees, not the sum. Dividing by the per-element count
            # of agreeing models keeps the merged magnitude in range (Yadav et al., 2023).
            elected = torch.sign((stacked * weight_view).sum(dim=0))
            agrees = torch.sign(stacked) == elected
            numerator = (stacked * weight_view * agrees).sum(dim=0)
            count = agrees.sum(dim=0).clamp(min=1.0)
            merged_delta = numerator / count
        else:
            merged_delta = (stacked * weight_view).sum(dim=0)
        out[key] = (base_f + merged_delta).cpu()
    return out


def _merge_directory(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]],
    weights: Sequence[float],
    method: str,
    base_state: OrderedDict[str, torch.Tensor] | None,
    densities: Sequence[float],
    device: torch.device,
    generator: torch.Generator,
    context: str,
) -> OrderedDict[str, torch.Tensor]:
    if method == "linear":
        return _merge_linear(state_dicts, weights, device)
    if method == "slerp":
        return _merge_slerp(state_dicts, weights, device)
    # Delta methods need a base state. ST weight submodules may have no matching
    # base weights (e.g. base_model lacks a Dense head) — fall back to linear.
    if base_state is None:
        logger.warning("No base weights for %s; falling back to linear averaging.", context)
        return _merge_linear(state_dicts, weights, device)
    return _merge_delta(state_dicts, weights, base_state, method, densities, device, generator)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _resolve_weights(weights: Sequence[float] | None, n: int, method: str) -> list[float]:
    """Default to [1.0]*n for delta methods, [1/n]*n otherwise."""
    if weights is None:
        return [1.0] * n if method in DELTA_BASED_METHODS else [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}.")
    return [float(w) for w in weights]


def _resolve_densities(densities: Sequence[float] | None, n: int) -> list[float]:
    if densities is None:
        return [1.0] * n
    if len(densities) != n:
        raise ValueError(f"Expected {n} densities, got {len(densities)}.")
    return [float(d) for d in densities]


def _validate_modules_consistency(
    modules_per_model: list[list[dict[str, Any]]], model_ids: Sequence[str]
) -> None:
    present = [bool(m) for m in modules_per_model]
    if any(present) and not all(present):
        with_json = [m for m, p in zip(model_ids, present) if p]
        without_json = [m for m, p in zip(model_ids, present) if not p]
        raise ValueError(
            f"Inputs disagree on modules.json presence.\n"
            f"  with modules.json:    {with_json}\n"
            f"  without modules.json: {without_json}\n"
            "Re-save the body-only input(s) via the matching ST class (e.g. "
            "`CrossEncoder(path).save(new_path)`) to give them a modules.json, "
            "or merge only body-only models together."
        )

    base = modules_per_model[0]
    for i, mods in enumerate(modules_per_model[1:], start=1):
        if len(mods) != len(base):
            raise ValueError(
                f"Module count mismatch: {model_ids[0]!r} has {len(base)} modules, "
                f"{model_ids[i]!r} has {len(mods)}."
            )
        for j, (m_base, m_other) in enumerate(zip(base, mods)):
            if m_base["type"] != m_other["type"]:
                raise ValueError(
                    f"Module #{j} class mismatch:\n"
                    f"  {model_ids[0]!r}: {m_base['type']}\n"
                    f"  {model_ids[i]!r}: {m_other['type']}"
                )
            if m_base.get("path", "") != m_other.get("path", ""):
                raise ValueError(
                    f"Module #{j} path mismatch (different save_in_root settings?):\n"
                    f"  {model_ids[0]!r}: {m_base.get('path')!r}\n"
                    f"  {model_ids[i]!r}: {m_other.get('path')!r}"
                )

    unsupported = {
        "sentence_transformers.base.modules.router.Asym",
        "sentence_transformers.base.modules.router.Router",
        "sentence_transformers.models.Asym",
        "sentence_transformers.models.Router",
    }
    for j, m in enumerate(base):
        if m["type"] in unsupported:
            raise NotImplementedError(
                f"Merging models containing {m['type'].rsplit('.', 1)[-1]!r} (module #{j}) "
                "is not supported yet — its child modules would need recursive merging."
            )


def _validate_body_compatibility(
    body_dirs: Sequence[str], body_ids: Sequence[str]
) -> None:
    """Reject architecturally incompatible bodies up front with a clear message.

    Compares the root ``config.json`` of every input (and base model) so a hidden
    size or architecture mismatch surfaces here, rather than as a low-level tensor
    shape/key error during the body merge. Missing keys are skipped, not enforced.
    """
    configs: list[dict[str, Any] | None] = []
    for d in body_dirs:
        cfg_path = os.path.join(d, "config.json")
        configs.append(_read_json(cfg_path) if os.path.exists(cfg_path) else None)

    ref_idx = next((i for i, c in enumerate(configs) if c is not None), None)
    if ref_idx is None:
        return
    ref = configs[ref_idx]
    for i, cfg in enumerate(configs):
        if cfg is None or i == ref_idx:
            continue
        if ref.get("model_type") and cfg.get("model_type") and cfg["model_type"] != ref["model_type"]:
            raise ValueError(
                f"Architecture mismatch: {body_ids[ref_idx]!r} is a {ref['model_type']!r} model, "
                f"{body_ids[i]!r} is a {cfg['model_type']!r} model. Merging requires the same architecture."
            )
        if ref.get("hidden_size") and cfg.get("hidden_size") and cfg["hidden_size"] != ref["hidden_size"]:
            raise ValueError(
                f"Hidden size mismatch: {body_ids[ref_idx]!r} has hidden_size={ref['hidden_size']}, "
                f"{body_ids[i]!r} has hidden_size={cfg['hidden_size']}. Merging requires the same hidden size."
            )


def _validate_module_configs(
    module_dirs: list[str], module_type: str, module_index: int, model_ids: Sequence[str]
) -> None:
    # Compare configs canonicalized through the Module class so older saves with
    # missing-but-default-equivalent keys compare equal. None acts as a wildcard.
    from sentence_transformers.util import import_from_string

    base_cfg_path = os.path.join(module_dirs[0], "config.json")
    if not os.path.exists(base_cfg_path):
        return

    module_cls: Any = None
    try:
        module_cls = import_from_string(module_type)
    except Exception:
        pass

    def _canonical(directory: str) -> dict[str, Any]:
        if module_cls is not None and hasattr(module_cls, "load_config"):
            try:
                cfg = module_cls.load_config(directory)
                try:
                    return module_cls(**cfg).get_config_dict()
                except Exception:
                    return cfg
            except Exception:
                pass
        return _read_json(os.path.join(directory, "config.json"))

    def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
        for key in set(a) | set(b):
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            if va != vb:
                return False
        return True

    base_cfg = _canonical(module_dirs[0])
    for i, d in enumerate(module_dirs[1:], start=1):
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        other_cfg = _canonical(d)
        if not _compatible(base_cfg, other_cfg):
            raise ValueError(
                f"Module #{module_index} config mismatch between {model_ids[0]!r} and {model_ids[i]!r}:\n"
                f"  {model_ids[0]!r}: {base_cfg}\n"
                f"  {model_ids[i]!r}: {other_cfg}"
            )


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #
def _copy_scaffold(src_root: str, output_path: str) -> None:
    """Copy every file from the first model into output_path, minus weight files.

    Configs, tokenizer files, module subfolders and stateless-module weights are
    reproduced verbatim; weight files are skipped because the merged versions are
    written on top afterwards.
    """
    for dirpath, _dirnames, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        target_dir = output_path if rel == "." else os.path.join(output_path, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in filenames:
            lower = name.lower()
            if lower.endswith(_WEIGHT_EXTS) or lower.startswith(_WEIGHT_PREFIXES):
                continue
            # Don't inherit the first input's model card - the merged model would
            # otherwise republish it verbatim on the next save/push_to_hub.
            if rel == "." and lower == "readme.md":
                continue
            shutil.copy2(os.path.join(dirpath, name), os.path.join(target_dir, name))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def merge_models(
    cls: type[BaseModel],
    models: Sequence[str],
    weights: Sequence[float] | None = None,
    method: str = "linear",
    base_model: str | None = None,
    output_path: str | None = None,
    dtype: str = "float16",
    device: str = "cpu",
    densities: Sequence[float] | None = None,
    seed: int = 0,
    **load_kwargs: Any,
) -> BaseModel:
    """Internal entry point for ``BaseModel.merge``. See that classmethod for the
    user-facing docstring with full argument descriptions."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported merge method {method!r}. Supported: {sorted(SUPPORTED_METHODS)}")
    if len(models) < 2:
        raise ValueError(f"At least two models are required for merging, got {len(models)}.")
    if method in TWO_MODEL_METHODS and len(models) != 2:
        raise ValueError(f"Method {method!r} requires exactly 2 input models, got {len(models)}.")
    if method in TWO_MODEL_METHODS and len(set(models)) < 2:
        raise ValueError(f"Method {method!r} requires two distinct input models. Got duplicates: {list(models)!r}.")
    if method in REQUIRES_BASE_MODEL and base_model is None:
        raise ValueError(f"Method {method!r} requires `base_model`.")
    if base_model is not None and method not in REQUIRES_BASE_MODEL:
        raise ValueError(
            f"`base_model` is only used by delta-based methods {sorted(REQUIRES_BASE_MODEL)}, not {method!r}. "
            "Remove `base_model` or pick a delta-based method."
        )
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported: {sorted(_DTYPES)}")
    if output_path is None:
        raise ValueError("`output_path` is required.")

    weights = _resolve_weights(weights, len(models), method)
    densities = _resolve_densities(densities, len(models))
    torch_dtype = _DTYPES[dtype]
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device).manual_seed(seed)

    hub_kwargs = _hub_kwargs_from(load_kwargs)
    modules_per_model = [_load_modules_config(m, **hub_kwargs) for m in models]
    _validate_modules_consistency(modules_per_model, models)
    base_modules = modules_per_model[0]

    if base_modules and base_modules[0].get("path", "") != "":
        raise NotImplementedError(
            "Merging models whose first (Transformer) module is saved in a subfolder "
            "(save_in_root=False) is not yet supported."
        )

    model_dirs = [_resolve_module_dir(m, subfolder="", **hub_kwargs) for m in models]
    if any(d is None for d in model_dirs):
        missing = [models[i] for i, d in enumerate(model_dirs) if d is None]
        raise FileNotFoundError(f"Could not resolve a local directory for {missing}.")
    base_dir = _resolve_module_dir(base_model, subfolder="", **hub_kwargs) if base_model else None
    if base_model and base_dir is None:
        # Delta methods require the base body weights; a silent linear fallback here
        # would merge with a different algorithm than the user asked for.
        raise FileNotFoundError(f"Could not resolve a local directory for base_model {base_model!r}.")

    body_dirs = list(model_dirs) + ([base_dir] if base_dir is not None else [])
    body_ids = list(models) + ([base_model] if base_dir is not None else [])
    _validate_body_compatibility(body_dirs, body_ids)

    os.makedirs(output_path, exist_ok=True)
    _copy_scaffold(model_dirs[0], output_path)

    # 1. Merge the transformer body (root weights).
    body_state_dicts = [_load_state_dict(d) for d in model_dirs]
    body_base = _load_state_dict(base_dir) if base_dir is not None else None
    merged_body = _merge_directory(
        body_state_dicts, weights, method, body_base, densities, torch_device, generator, context="transformer body"
    )
    _save_state_dict(merged_body, output_path, torch_dtype)

    # 2. Merge each non-body module.
    for idx, module_cfg in enumerate(base_modules[1:], start=1):
        subfolder = module_cfg["path"]
        target_dir = os.path.join(output_path, subfolder)
        per_model_dirs = [_resolve_module_dir(m, subfolder=subfolder, **hub_kwargs) for m in models]

        # File-less stateless module (e.g. Normalize): no dir on the Hub for any
        # input. Ensure the folder exists in the output so the module reloads.
        if all(_is_empty_module_dir(d) for d in per_model_dirs):
            os.makedirs(target_dir, exist_ok=True)
            continue

        if any(d is None for d in per_model_dirs):
            missing = [models[i] for i, d in enumerate(per_model_dirs) if d is None]
            raise FileNotFoundError(f"Subfolder {subfolder!r} missing for {missing} but present for others.")

        _validate_module_configs(per_model_dirs, module_cfg["type"], idx, models)

        # Stateless module with a config only (e.g. Pooling): already copied by the scaffold.
        if not _has_weight_files(per_model_dirs[0]):
            continue

        module_state_dicts = [_load_state_dict(d) for d in per_model_dirs]
        module_base = None
        if base_dir is not None:
            base_module_dir = _resolve_module_dir(base_model, subfolder=subfolder, **hub_kwargs)
            if base_module_dir is not None and _has_weight_files(base_module_dir):
                module_base = _load_state_dict(base_module_dir)
        merged_module = _merge_directory(
            module_state_dicts, weights, method, module_base, densities, torch_device, generator,
            context=f"module {subfolder!r}",
        )
        _save_state_dict(merged_module, os.path.join(output_path, subfolder), torch_dtype)

    return cls(output_path, **load_kwargs)
