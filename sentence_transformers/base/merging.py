"""Model merging through mergekit's public in-memory API."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from packaging.version import Version
from transformers import PreTrainedModel

from sentence_transformers.base.modules import Module, Transformer
from sentence_transformers.sparse_encoder.modules import MLMTransformer, SpladePooling

if TYPE_CHECKING:
    from mergekit.merge_methods import BatchOptions

    from sentence_transformers.base.model import BaseModel


def _validate_models(models: Sequence[BaseModel]) -> None:
    """Check input compatibility before merging, using the first model as reference.

    Require torch, no quantized modules or PEFT adapters, matching module names/types,
    configurations and extra_repr settings, and matching tokenizer vocabularies and
    special tokens where exposed. Transformer metadata, private config fields, dtypes,
    and model-level prompts may differ.
    Accept legacy MLMTransformer wrappers and ignore derived SPLADE pooling dimensions.
    State keys, tied weights, shapes, and buffers are checked later. This does not
    verify tokenizer preprocessing, common training ancestry, or merge quality.
    """
    reference = dict(models[0].named_modules())
    type_aliases = {MLMTransformer: Transformer}

    for index, model in enumerate(models):
        if model.backend != "torch":
            raise ValueError("Model merging requires the torch backend.")

        modules = dict(model.named_modules())
        if any(getattr(module, "peft_config", None) for module in modules.values()):
            raise ValueError(
                "Merging models with PEFT adapters is not supported. "
                "Merge the adapters into the base weights and unload them first."
            )
        if modules.keys() != reference.keys():
            raise ValueError(f"Model {index} has a different module structure from model 0.")

        for name, module in modules.items():
            other = reference[name]
            if type_aliases.get(type(module), type(module)) is not type_aliases.get(type(other), type(other)):
                raise ValueError(f"Module {name!r} has a different type in model {index}.")
            if getattr(module, "is_quantized", False):
                raise ValueError("Merging quantized models is not supported. Load unquantized models first.")
            if module.extra_repr() != other.extra_repr():
                raise ValueError(f"Module {name!r} has a different configuration in model {index}.")

            if isinstance(module, Module):
                configs = [item.get_config_dict() for item in (module, other)]
                if isinstance(module, SpladePooling):
                    for config in configs:
                        config.pop("embedding_dimension", None)
                if configs[0] != configs[1]:
                    raise ValueError(f"Module {name!r} has a different configuration in model {index}.")

            if isinstance(module, PreTrainedModel):
                # Concrete module types were checked above. Saved provenance and dtype need not match.
                ignored = {"architectures", "transformers_version", "torch_dtype", "dtype"}
                configs = [
                    {
                        key: value
                        for key, value in item.config.to_dict().items()
                        if key not in ignored and not key.startswith("_")
                    }
                    for item in (module, other)
                ]
                if configs[0] != configs[1]:
                    raise ValueError(f"Transformer {name!r} has a different configuration in model {index}.")

            tokenizer = getattr(module, "tokenizer", None)
            other_tokenizer = getattr(other, "tokenizer", None)
            if tokenizer is not None and hasattr(tokenizer, "get_vocab"):
                if (
                    other_tokenizer is None
                    or tokenizer.get_vocab() != other_tokenizer.get_vocab()
                    or getattr(tokenizer, "special_tokens_map", None)
                    != getattr(other_tokenizer, "special_tokens_map", None)
                ):
                    raise ValueError(
                        f"Tokenizer vocabulary or special tokens differ in model {index}, module {name!r}."
                    )


def _unique_state_dict(model: BaseModel, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Return one entry per tensor object and a map from every original key to its representative.

    Deduplicating tied weights prevents stochastic methods from merging aliases differently.
    Tensors are detached and moved to ``device``, but may share input storage, so do not
    modify them in place. Sharing is detected by object identity, not overlapping storage.
    Non-tensor entries and meta tensors are rejected.
    """
    state = {}
    aliases = {}
    names = {}

    # Keep parameter objects until aliases are identified. Detaching first can hide their identity.
    for name, tensor in model.state_dict(keep_vars=True).items():
        if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
            raise ValueError(f"State entry {name!r} must be a materialized tensor.")

        canonical_name = names.setdefault(id(tensor), name)
        aliases[name] = canonical_name
        if canonical_name == name:
            state[name] = tensor.detach().to(device)

    return state, aliases


@torch.no_grad()
def merge_models(
    cls: type[BaseModel],
    models: Sequence[str | Path | BaseModel],
    method: str = "linear",
    *,
    base_model: str | Path | BaseModel | None = None,
    parameters: dict[str, Any] | None = None,
    dtype: str | torch.dtype | None = torch.float32,
    out_dtype: str | torch.dtype | None = None,
    device: str | torch.device = "cpu",
    batch_options: BatchOptions | None = None,
    **load_kwargs: Any,
) -> BaseModel:
    """Implement ``BaseModel.merge``. See that method for the public arguments.

    Load and validate inputs, merge each shared tensor once through mergekit, then restore
    aliases into a copy of the first model. Return it in evaluation mode with fresh model-card
    metadata. Caller-owned inputs are left unchanged.
    """
    try:
        if Version(version("mergekit")) < Version("0.1.5"):
            raise ImportError("mergekit is too old")

        from mergekit.merge_methods import BasePolicy, merge_state_dicts
        from mergekit.merge_methods import get as get_merge_method
    except ImportError as exc:
        raise ImportError(
            "Model merging requires mergekit>=0.1.5 with merge_state_dicts. Install sentence-transformers[merge]."
        ) from exc

    if isinstance(models, (str, Path)) or len(models) < 2:
        raise ValueError("At least two models are required for merging.")

    dtypes = {"dtype": dtype, "out_dtype": out_dtype}
    for name, value in dtypes.items():
        if isinstance(value, str):
            value = getattr(torch, value, value)
        if value not in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"{name} must be None, float16, bfloat16, float32, or float64.")
        dtypes[name] = value

    device = torch.device(device)
    merge_method = get_merge_method(method)

    if base_model is not None and merge_method.spec.contract.base == BasePolicy.IGNORED:
        raise ValueError(f"Method {method!r} does not use a base_model.")

    inputs = list(models)
    base_index = None
    if base_model is not None:
        # Reuse an existing input's ID so it is not counted twice and per-input parameter maps stay valid.
        base_index = next(
            (
                index
                for index, model in enumerate(inputs)
                if model is base_model
                or isinstance(model, (str, Path))
                and isinstance(base_model, (str, Path))
                and str(model) == str(base_model)
            ),
            len(inputs),
        )
        if base_index == len(inputs):
            inputs.append(base_model)
    elif method == "slerp":
        base_index = 0
    merge_method.validate_inputs(list(range(len(inputs))), base_index)

    parameters = dict(parameters or {})
    specs = {parameter.name: parameter for parameter in merge_method.spec.parameters}
    if "weight" in specs:
        parameters.setdefault("weight", 1.0)
    if method == "slerp":
        parameters.setdefault("t", 0.5)

    loaded = []
    for model in inputs:
        if isinstance(model, (str, Path)):
            model = cls(str(model), device=device, **load_kwargs)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a {cls.__name__} instance or model path, got {type(model).__name__}.")
        loaded.append(model)

    _validate_models(loaded)

    states = {}
    aliases = None
    for index, model in enumerate(loaded):
        states[index], model_aliases = _unique_state_dict(model, device)
        if aliases is not None and model_aliases != aliases:
            raise ValueError(f"Model {index} has different state keys or tied weights from model 0.")
        aliases = model_aliases

    merged_state_dict = merge_state_dicts(
        states,
        merge_method,
        base=base_index,
        parameters=parameters,
        dtype=dtypes["dtype"],
        out_dtype=dtypes["out_dtype"],
        batch_options=batch_options,
    )

    # Substitute merged tensors during deepcopy instead of copying the first model's weights.
    memo = {}
    for name, tensor in loaded[0].state_dict(keep_vars=True).items():
        if id(tensor) not in memo:
            # Consume outputs one at a time. Cloning also isolates methods that return input storage.
            value = merged_state_dict.pop(aliases[name]).clone().requires_grad_(tensor.requires_grad)
            if isinstance(tensor, torch.nn.Parameter):
                value = torch.nn.Parameter(value, requires_grad=tensor.requires_grad)
            memo[id(tensor)] = value

    merged = deepcopy(loaded[0], memo).to(device=device)

    # A parent's training history and evaluation scores do not describe the newly merged model.
    merged._model_card_text = None
    merged.model_card_data = merged.model_card_data_class(local_files_only=True)
    merged.model_card_data.register_model(merged)
    merged.eval()

    return merged
