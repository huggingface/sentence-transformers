"""Model merging via mergekit, with handling for ST's surrounding modules.

The transformer body of each input is merged by mergekit. Weight-bearing
ST modules (Dense, LayerNorm, ...) are merged at the state-dict level.
Stateless modules (Pooling, Normalize, ...) are validated and copied from
the first model.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
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
    "nuslerp",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_linear",
    "breadcrumbs",
    "breadcrumbs_ties",
    "della",
    "della_linear",
    "model_stock",
    "multislerp",
    "nearswap",
    "sce",
    "arcee_fusion",
    "karcher",
)
"""Merge methods exposed via mergekit."""

REQUIRES_BASE_MODEL: frozenset[str] = frozenset(
    {
        "ties",
        "dare_ties",
        "dare_linear",
        "task_arithmetic",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
        "model_stock",
        "sce",
    }
)
"""Methods that require an explicit ``base_model``."""

TWO_MODEL_METHODS: frozenset[str] = frozenset({"slerp", "nuslerp", "nearswap"})
"""Methods that operate on exactly two input models."""

DENSITY_METHODS: frozenset[str] = frozenset(
    {"ties", "dare_ties", "dare_linear", "breadcrumbs", "breadcrumbs_ties", "della", "della_linear"}
)

DELTA_BASED_METHODS: frozenset[str] = frozenset(
    {
        "task_arithmetic",
        "ties",
        "dare_ties",
        "dare_linear",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
    }
)
# Methods whose weights are raw scaling factors on per-model deltas (rather
# than a simplex). For these, weights=None defaults to [1.0]*n and mergekit's
# `normalize` is left off so the user's scaling carries through.


def _require_mergekit() -> None:
    try:
        from mergekit.config import MergeConfiguration  # noqa: F401
        from mergekit.merge import run_merge  # noqa: F401
        from mergekit.options import MergeOptions  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Model merging requires the `mergekit` package. Install it with:\n"
            "    pip install sentence-transformers[merge]\n"
            "or directly with:\n"
            "    pip install mergekit"
        ) from exc

    # mergekit 0.1.4 + pydantic 2.10.x: a few pydantic models hold forward refs
    # to torch types that aren't resolved at import time. See arcee-ai/mergekit#681.
    try:
        import torch  # noqa: F401

        from mergekit.architecture.base import (
            ConfiguredModelArchitecture,
            ConfiguredModuleArchitecture,
        )

        ConfiguredModuleArchitecture.model_rebuild()
        ConfiguredModelArchitecture.model_rebuild()
    except Exception:
        pass

    # mergekit 0.1.4 bug: plan.py:172 does `[w_in.name] + (w_in.aliases or [])`
    # but `aliases` is typed as Tuple[str, ...] (so the pydantic model stays
    # hashable). The list+tuple concat raises TypeError on any arch that uses
    # aliases — mostly BERT variants, which is exactly what ST users merge.
    # Patch the offending method at runtime with `*` unpacking instead of `+`.
    try:
        import inspect as _inspect
        import textwrap as _textwrap

        from mergekit.plan import MergePlanner

        if not getattr(MergePlanner, "_st_plan_tensor_patched", False):
            _src = _inspect.getsource(MergePlanner.plan_tensor)
            _buggy = "[w_in.name] + (w_in.aliases or [])"
            _fixed = "[w_in.name, *(w_in.aliases or ())]"
            if _buggy in _src:
                _new_src = _textwrap.dedent(_src).replace(_buggy, _fixed)
                _ns: dict[str, Any] = {}
                _mod = _inspect.getmodule(MergePlanner)
                exec(_new_src, _mod.__dict__, _ns)
                MergePlanner.plan_tensor = _ns["plan_tensor"]
                logger.warning(
                    "Patched mergekit MergePlanner.plan_tensor to work around "
                    "an upstream tuple/list bug. Drop this patch once mergekit "
                    "fixes it."
                )
            MergePlanner._st_plan_tensor_patched = True
    except Exception as _patch_err:
        logger.warning("Could not apply mergekit plan_tensor patch: %s", _patch_err)


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
    # Plain HF AutoModelForSequenceClassification checkpoints (e.g. BAAI/bge-reranker-*)
    # have no modules.json. Treat them as body-only models — the empty list
    # tells the rest of the pipeline to skip the non-body module loop.
    path = load_file_path(model_name_or_path=model_id, filename="modules.json", **hub_kwargs)
    if path is None:
        return []
    return _read_json(path)


def _resolve_module_dir(model_id: str, subfolder: str, **hub_kwargs: Any) -> str | None:
    # Returns None for missing-or-empty subdirs. Note that load_dir_path can
    # return a non-existent path string (HF Hub doesn't track empty dirs, e.g.
    # for a no-op-save module like Normalize), hence the explicit isdir check.
    try:
        path = load_dir_path(model_name_or_path=model_id, subfolder=subfolder, **hub_kwargs)
    except (FileNotFoundError, OSError):
        return None
    if path is None or not os.path.isdir(path):
        return None
    return path


def _has_weight_files(directory: str) -> bool:
    return any(
        os.path.exists(os.path.join(directory, name)) for name in ("model.safetensors", "pytorch_model.bin")
    )


def _load_state_dict(directory: str) -> OrderedDict[str, torch.Tensor]:
    safetensors_path = os.path.join(directory, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return OrderedDict(load_file(safetensors_path))
    pytorch_path = os.path.join(directory, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return OrderedDict(torch.load(pytorch_path, map_location="cpu", weights_only=True))
    raise FileNotFoundError(f"No model weights found in {directory!r}.")


def _save_state_dict(state_dict: OrderedDict[str, torch.Tensor], directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    save_safetensors_file(dict(state_dict), os.path.join(directory, "model.safetensors"))


def _resolve_weights(weights: Sequence[float] | None, n: int, method: str) -> list[float]:
    """Default to [1.0]*n for delta methods, [1/n]*n otherwise."""
    if weights is None:
        if method in DELTA_BASED_METHODS:
            return [1.0] * n
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}.")
    return [float(w) for w in weights]


def _validate_modules_consistency(
    modules_per_model: list[list[dict[str, Any]]], model_ids: Sequence[str]
) -> None:
    # Reject mixing body-only inputs (no modules.json) with ST-style ones.
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

    # Reject module types that are explicitly out of scope.
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


def _validate_module_configs(
    module_dirs: list[str | None],
    module_type: str,
    module_index: int,
    model_ids: Sequence[str],
) -> dict[str, Any] | None:
    # Compare module configs after canonicalizing through the Module class
    # (load_config -> __init__ -> get_config_dict). This way older saves with
    # missing-but-default-equivalent keys compare equal to newer saves.
    # Falls back to raw JSON if the class can't be imported/instantiated.
    from sentence_transformers.util import import_from_string

    base_dir = module_dirs[0]
    if base_dir is None:
        return None
    base_cfg_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(base_cfg_path):
        return None

    module_cls: Any = None
    try:
        module_cls = import_from_string(module_type)
    except Exception:
        pass

    def _canonical_config(directory: str) -> dict[str, Any]:
        # Prefer Module-class canonicalization; fall back to raw JSON.
        if module_cls is not None and hasattr(module_cls, "load_config"):
            try:
                cfg = module_cls.load_config(directory)
                # Instantiate to fill in defaults for keys not present in older saves.
                try:
                    instance = module_cls(**cfg)
                    return instance.get_config_dict()
                except Exception:
                    return cfg
            except Exception:
                pass
        return _read_json(os.path.join(directory, "config.json"))

    def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
        # Treat None as a wildcard. Some modules (e.g. SpladePooling) leave
        # ``embedding_dimension=None`` until the first forward pass, so saved
        # checkpoints may differ in whether that field is None or filled in.
        for key in set(a) | set(b):
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                continue
            if va != vb:
                return False
        return True

    base_cfg = _canonical_config(base_dir)
    for i, d in enumerate(module_dirs[1:], start=1):
        if d is None:
            continue
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        other_cfg = _canonical_config(d)
        if not _compatible(base_cfg, other_cfg):
            raise ValueError(
                f"Module #{module_index} config mismatch between "
                f"{model_ids[0]!r} and {model_ids[i]!r}:\n"
                f"  {model_ids[0]!r}: {base_cfg}\n"
                f"  {model_ids[i]!r}: {other_cfg}"
            )
    return base_cfg


def _merge_state_dict_linear(
    state_dicts: list[OrderedDict[str, torch.Tensor]], weights: Sequence[float]
) -> OrderedDict[str, torch.Tensor]:
    weight_sum = float(sum(weights))
    if weight_sum == 0:
        raise ValueError("Sum of weights must be non-zero.")
    norm_weights = [w / weight_sum for w in weights]

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    base = state_dicts[0]
    keys = list(base.keys())
    for key in keys:
        tensors = []
        for i, sd in enumerate(state_dicts):
            if key not in sd:
                raise KeyError(f"Key {key!r} present in model 0 but missing in model {i}.")
            t = sd[key]
            if t.shape != base[key].shape:
                raise ValueError(
                    f"Shape mismatch for key {key!r}: model[0]={tuple(base[key].shape)}, "
                    f"model[{i}]={tuple(t.shape)}"
                )
            tensors.append(t)
        stacked = torch.stack([t.float() for t in tensors], dim=0)
        view_shape = (-1,) + (1,) * (stacked.dim() - 1)
        weight_tensor = torch.tensor(norm_weights, dtype=stacked.dtype).view(view_shape)
        merged = (stacked * weight_tensor).sum(dim=0)
        result[key] = merged.to(base[key].dtype)
    return result


def _merge_state_dict_task_arithmetic(
    state_dicts: list[OrderedDict[str, torch.Tensor]],
    weights: Sequence[float],
    base_state: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, base_tensor in base_state.items():
        delta_sum = torch.zeros_like(base_tensor, dtype=torch.float32)
        for i, (sd, w) in enumerate(zip(state_dicts, weights)):
            if key not in sd:
                continue
            t = sd[key]
            if t.shape != base_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for key {key!r}: base={tuple(base_tensor.shape)}, "
                    f"model[{i}]={tuple(t.shape)}"
                )
            delta_sum = delta_sum + w * (t.float() - base_tensor.float())
        merged = base_tensor.float() + delta_sum
        result[key] = merged.to(base_tensor.dtype)
    return result


def _merge_state_dict_dispatch(
    state_dicts: list[OrderedDict[str, torch.Tensor]],
    weights: Sequence[float],
    method: str,
    base_state: OrderedDict[str, torch.Tensor] | None,
    module_subfolder: str = "",
) -> OrderedDict[str, torch.Tensor]:
    if method == "task_arithmetic" and base_state is not None:
        return _merge_state_dict_task_arithmetic(state_dicts, weights, base_state)
    if method != "linear":
        # Body is merged with the user's method via mergekit, but ST-side
        # weight modules (Dense, LayerNorm, ...) currently only have a linear
        # implementation here. Warn so the user isn't surprised.
        logger.warning(
            "Method %r applies to the transformer body only; ST weight modules%s "
            "(Dense / LayerNorm / WeightedLayerPooling) fall back to linear averaging.",
            method,
            f" at {module_subfolder!r}" if module_subfolder else "",
        )
    return _merge_state_dict_linear(state_dicts, weights)


def _build_mergekit_config(
    models: Sequence[str],
    weights: Sequence[float],
    method: str,
    base_model: str | None,
    dtype: str,
    densities: Sequence[float] | None = None,
):
    from mergekit.config import InputModelDefinition, MergeConfiguration

    config_dict: dict[str, Any] = {"merge_method": method, "dtype": dtype}
    # Don't normalize for delta methods — that would halve the user's scaling.
    parameters: dict[str, Any] = {
        "normalize": method not in DELTA_BASED_METHODS,
        "int8_mask": True,
    }

    if method == "slerp":
        if base_model is None:
            base = models[0]
            other = models[1]
            t_value = float(weights[1])
        else:
            base = base_model
            other = models[0] if models[0] != base_model else models[1]
            t_value = float(weights[0] if models[0] != base_model else weights[1])
        config_dict["base_model"] = base
        config_dict["models"] = [InputModelDefinition(model=other, parameters={})]
        # slerp wants `t` at the top level (tensor-parameter), not per-model.
        parameters["t"] = t_value
    else:
        input_models = []
        for idx, (model_id, w) in enumerate(zip(models, weights)):
            params: dict[str, Any] = {"weight": float(w)}
            if method in DENSITY_METHODS:
                params["density"] = (
                    float(densities[idx]) if densities is not None else 1.0
                )
            input_models.append(InputModelDefinition(model=model_id, parameters=params))
        config_dict["models"] = input_models
        if base_model is not None:
            config_dict["base_model"] = base_model

    config_dict["parameters"] = parameters
    return MergeConfiguration.model_validate(config_dict)


def _collect_input_tensor_keys(models: Sequence[str], hub_kwargs: dict[str, Any]) -> set[str]:
    """Return the union of tensor names across all input checkpoints."""
    keys: set[str] = set()
    for m in models:
        # Prefer the sharded safetensors index — gives us all keys without loading tensors.
        idx_path = load_file_path(
            model_name_or_path=m, filename="model.safetensors.index.json", **hub_kwargs
        )
        if idx_path is not None:
            with open(idx_path, encoding="utf-8") as f:
                keys.update(json.load(f).get("weight_map", {}).keys())
            continue
        sf_path = load_file_path(model_name_or_path=m, filename="model.safetensors", **hub_kwargs)
        if sf_path is not None:
            from safetensors import safe_open

            with safe_open(sf_path, framework="pt") as f:
                keys.update(f.keys())
            continue
        bin_idx = load_file_path(
            model_name_or_path=m, filename="pytorch_model.bin.index.json", **hub_kwargs
        )
        if bin_idx is not None:
            with open(bin_idx, encoding="utf-8") as f:
                keys.update(json.load(f).get("weight_map", {}).keys())
            continue
        # Last resort: load the full pytorch_model.bin keys.
        bin_path = load_file_path(model_name_or_path=m, filename="pytorch_model.bin", **hub_kwargs)
        if bin_path is not None:
            sd = torch.load(bin_path, map_location="cpu", weights_only=True)
            keys.update(sd.keys())
    return keys


_LAYER_KEY_HINTS = (".layer.", ".layers.", ".block.", ".blocks.", ".h.")


def _relax_required_weights(models: Sequence[str], hub_kwargs: dict[str, Any]) -> None:
    """Reconcile mergekit's arch registry with what's actually in the inputs.

    Two failure modes are covered:

    * Some required weights in mergekit's hand-crafted arch JSON aren't saved
      by every variant of the architecture — e.g. ``bert-masked-lm`` requires
      ``bert.pooler.dense.{weight,bias}``, but ``BertForMaskedLM`` checkpoints
      typically don't save the pooler. Mark them optional.
    * mergekit's hand-crafted arch may simply be incomplete — e.g. that same
      ``bert-masked-lm`` entry doesn't list ``cls.predictions.transform.*``,
      so those weights would silently be dropped from the merged output. When
      we detect such uncovered non-layer keys, we delete the arch entry to
      force mergekit's auto-inference path, which lists every input tensor.
    """
    try:
        from mergekit.architecture.json_definitions import NAME_TO_ARCH
        from transformers import AutoConfig
    except Exception:
        return

    try:
        cfg = AutoConfig.from_pretrained(models[0], trust_remote_code=False)
    except Exception:
        return
    if not cfg.architectures or cfg.architectures[0] not in NAME_TO_ARCH:
        return
    arch_name = cfg.architectures[0]

    present = _collect_input_tensor_keys(models, hub_kwargs)
    if not present:
        return

    # Figure out which non-layer input keys this arch doesn't cover. Layer
    # weights live behind templated names that we can't expand without the
    # full layer count, so we exclude anything that looks layer-shaped from
    # the coverage check.
    arch = NAME_TO_ARCH[arch_name][0]
    covered: set[str] = set()
    for module_def in arch.modules.values():
        definition = getattr(module_def.architecture, "definition", None)
        if definition is None:
            continue
        for w in list(definition.pre_weights) + list(definition.post_weights):
            covered.add(w.name)
            covered.update(w.aliases or ())
    non_layer_present = {k for k in present if not any(h in k for h in _LAYER_KEY_HINTS)}
    uncovered = non_layer_present - covered
    if uncovered:
        logger.warning(
            "mergekit's %s architecture does not list %d input weight(s) (e.g. %s); "
            "falling back to auto-inference so the merge includes every tensor.",
            arch_name,
            len(uncovered),
            ", ".join(sorted(uncovered)[:3]),
        )
        NAME_TO_ARCH.pop(arch_name, None)
        return

    def _patch(weights):
        out = []
        for w in weights:
            if w.optional:
                out.append(w)
                continue
            if w.name in present or any(a in present for a in (w.aliases or ())):
                out.append(w)
            else:
                out.append(w.model_copy(update={"optional": True}))
        return out

    new_arches = []
    for arch in NAME_TO_ARCH[arch_name]:
        new_modules = {}
        for module_name, module_def in arch.modules.items():
            mod_arch = module_def.architecture
            definition = getattr(mod_arch, "definition", None)
            if definition is None:
                new_modules[module_name] = module_def
                continue
            new_def = definition.model_copy(
                update={
                    "pre_weights": _patch(definition.pre_weights),
                    "post_weights": _patch(definition.post_weights),
                }
            )
            new_mod_arch = mod_arch.model_copy(update={"definition": new_def})
            new_modules[module_name] = module_def.model_copy(update={"architecture": new_mod_arch})
        new_arches.append(arch.model_copy(update={"modules": new_modules}))
    NAME_TO_ARCH[arch_name] = new_arches


def _run_mergekit(
    models: Sequence[str],
    weights: Sequence[float],
    method: str,
    base_model: str | None,
    output_path: str,
    dtype: str,
    device: str,
    densities: Sequence[float] | None,
    mergekit_options: dict[str, Any] | None,
    hub_kwargs: dict[str, Any] | None = None,
) -> None:
    _require_mergekit()
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    if hub_kwargs is not None:
        _relax_required_weights(models, hub_kwargs)

    config = _build_mergekit_config(
        models=models,
        weights=weights,
        method=method,
        base_model=base_model,
        dtype=dtype,
        densities=densities,
    )
    options_dict: dict[str, Any] = {
        "cuda": device.startswith("cuda"),
        "lazy_unpickle": False,
        "trust_remote_code": False,
        "low_cpu_memory": True,
    }
    if mergekit_options:
        options_dict.update(mergekit_options)
    options = MergeOptions(**options_dict)

    run_merge(config, out_path=output_path, options=options)


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
    mergekit_options: dict[str, Any] | None = None,
    **load_kwargs: Any,
) -> BaseModel:
    """Internal entry point for ``BaseModel.merge``. See that classmethod for the
    user-facing docstring with full argument descriptions."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported merge method {method!r}. Supported: {sorted(SUPPORTED_METHODS)}"
        )
    if len(models) < 2:
        raise ValueError(
            f"At least two models are required for merging, got {len(models)}."
        )
    if method in TWO_MODEL_METHODS and len(models) != 2:
        raise ValueError(f"Method {method!r} requires exactly 2 input models, got {len(models)}.")
    if method in TWO_MODEL_METHODS and len(set(models)) < 2:
        raise ValueError(
            f"Method {method!r} requires two distinct input models. Got duplicates: {list(models)!r}."
        )
    if method == "slerp" and base_model is not None and base_model not in models:
        raise ValueError(
            f"For `slerp` with an explicit `base_model`, the base must be one of `models`. "
            f"Got base_model={base_model!r}, models={list(models)!r}."
        )
    if method in REQUIRES_BASE_MODEL and base_model is None:
        raise ValueError(f"Method {method!r} requires `base_model`.")
    if output_path is None:
        raise ValueError("`output_path` is required.")

    weights = _resolve_weights(weights, len(models), method)
    _require_mergekit()

    hub_kwargs = _hub_kwargs_from(load_kwargs)

    modules_per_model = [_load_modules_config(m, **hub_kwargs) for m in models]
    _validate_modules_consistency(modules_per_model, models)
    base_modules = modules_per_model[0]

    # When every input is body-only (no modules.json — e.g. plain HF reranker
    # checkpoints like BAAI/bge-reranker-v2-m3), there's no ST module stack to
    # validate or copy. mergekit handles the body and cls(output_path) auto-
    # initializes a default module structure on reload.
    if base_modules:
        body_subfolder = base_modules[0].get("path", "")
        if body_subfolder != "":
            raise NotImplementedError(
                "Merging models whose first (Transformer) module is saved in a subfolder "
                "(save_in_root=False) is not yet supported. The first module of all input "
                f"models must be saved at the root, but got path={body_subfolder!r}."
            )

    os.makedirs(output_path, exist_ok=True)

    # 1. Run mergekit on the transformer body to a temp dir, then move into output_path.
    #    Using a temp dir avoids any conflict if output_path is non-empty.
    with tempfile.TemporaryDirectory(prefix="st-merge-") as tmp_root:
        body_out = os.path.join(tmp_root, "body")
        _run_mergekit(
            models=list(models),
            weights=weights,
            method=method,
            base_model=base_model,
            output_path=body_out,
            dtype=dtype,
            device=device,
            densities=densities,
            mergekit_options=mergekit_options,
            hub_kwargs=hub_kwargs,
        )
        # Copy mergekit outputs into output_path root
        for entry in os.listdir(body_out):
            src = os.path.join(body_out, entry)
            dst = os.path.join(output_path, entry)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    # 2. Handle each non-body module.
    for idx, module_cfg in enumerate(base_modules[1:], start=1):
        subfolder = module_cfg["path"]
        target_dir = os.path.join(output_path, subfolder)
        per_model_dirs = [_resolve_module_dir(m, subfolder=subfolder, **hub_kwargs) for m in models]

        # Stateless modules (e.g. Normalize) may be missing on disk for some
        # inputs and present-but-empty for others. If every input is empty in
        # this sense, just mkdir the target and move on.
        def _is_empty(d: str | None) -> bool:
            if d is None:
                return True
            cfg = os.path.exists(os.path.join(d, "config.json"))
            return not (cfg or _has_weight_files(d))

        if all(_is_empty(d) for d in per_model_dirs):
            os.makedirs(target_dir, exist_ok=True)
            continue

        if any(d is None for d in per_model_dirs):
            missing = [models[i] for i, d in enumerate(per_model_dirs) if d is None]
            raise FileNotFoundError(
                f"Subfolder {subfolder!r} missing for {missing} but present for others."
            )

        _validate_module_configs(per_model_dirs, module_cfg["type"], idx, models)

        if not _has_weight_files(per_model_dirs[0]):
            # Stateless module → copy directory from first model.
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(per_model_dirs[0], target_dir)
            continue

        # Weighted module → state_dict-level merge.
        state_dicts = [_load_state_dict(d) for d in per_model_dirs]
        base_state: OrderedDict[str, torch.Tensor] | None = None
        if method == "task_arithmetic" and base_model is not None:
            base_dir = _resolve_module_dir(base_model, subfolder=subfolder, **hub_kwargs)
            if base_dir is not None and _has_weight_files(base_dir):
                base_state = _load_state_dict(base_dir)
        merged_sd = _merge_state_dict_dispatch(
            state_dicts, weights, method, base_state, module_subfolder=subfolder
        )

        os.makedirs(target_dir, exist_ok=True)
        cfg_path = os.path.join(per_model_dirs[0], "config.json")
        if os.path.exists(cfg_path):
            shutil.copy2(cfg_path, os.path.join(target_dir, "config.json"))
        _save_state_dict(merged_sd, target_dir)

    # 3. Copy top-level sidecar files from the first input that mergekit didn't
    #    already write. This picks up the ST-specific configs (modules.json,
    #    config_sentence_transformers.json, sentence_bert_config.json — the
    #    last one holds the `transformer_task` needed to restore CE/Sparse
    #    heads on load) and the multimodal sidecars
    #    (preprocessor_config.json, processor_config.json, chat_template.json,
    #    ...). Weight files are explicitly skipped — the merged weights
    #    written by mergekit must not be shadowed by the originals.
    _WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
    _WEIGHT_PREFIXES = ("model.", "model-", "pytorch_model.", "pytorch_model-")
    first_root = _resolve_module_dir(models[0], subfolder="", **hub_kwargs)
    if first_root is not None:
        existing = set(os.listdir(output_path))
        for name in os.listdir(first_root):
            if name in existing:
                continue
            lower = name.lower()
            if lower.endswith(_WEIGHT_EXTS) or lower.startswith(_WEIGHT_PREFIXES):
                continue
            src = os.path.join(first_root, name)
            if not os.path.isfile(src):
                continue
            shutil.copy2(src, os.path.join(output_path, name))

    return cls(output_path, **load_kwargs)
