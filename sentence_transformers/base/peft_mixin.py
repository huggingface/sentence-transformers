from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

from transformers.integrations.peft import PeftAdapterMixin as PeftAdapterMixinTransformers
from transformers.utils.import_utils import is_peft_available

if TYPE_CHECKING:
    from peft import PeftModel


def as_peft_model(model: Any) -> PeftModel | None:
    """Return ``model`` if it is a ``peft.PeftModel`` wrapper, and ``None`` otherwise.

    A checkpoint that already carries adapters can end up wrapped by ``peft`` itself, e.g. after
    ``model[0].model = get_peft_model(model[0].model, peft_config)``, which is the only way to
    attach a prompt learning method. Such a model is PEFT capable, but it does not subclass the
    ``transformers`` PEFT mixin.
    """
    if model is None or not is_peft_available():
        return None

    from peft import PeftModel

    return model if isinstance(model, PeftModel) else None


def _peft_model_load_adapter(
    model: PeftModel, peft_model_id: str | None = None, adapter_name: str | None = None, **kwargs
) -> Any:
    """``PeftModel.load_adapter`` takes ``adapter_name`` as a required positional argument."""
    return model.load_adapter(peft_model_id, adapter_name or "default", **kwargs)


def _peft_model_add_adapter(model: PeftModel, adapter_config: Any, adapter_name: str | None = None, **kwargs) -> None:
    """``PeftModel.add_adapter`` takes the name before the config, the reverse of the transformers order."""
    return model.add_adapter(adapter_name or "default", adapter_config, **kwargs)


def _peft_model_active_adapters(model: PeftModel) -> list[str]:
    """``PeftModel.active_adapters`` is a property rather than a method."""
    return list(model.active_adapters)


def _peft_model_active_adapter(model: PeftModel) -> str:
    """``PeftModel.active_adapter`` is a property rather than a method."""
    return model.active_adapter


def _peft_model_get_adapter_state_dict(model: PeftModel, adapter_name: str | None = None, **kwargs) -> dict:
    """``PeftModel`` has no ``get_adapter_state_dict``; ``peft`` exposes it as a free function."""
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(model, adapter_name=adapter_name or model.active_adapter, **kwargs)


def _peft_model_toggle_adapters(model: PeftModel, enable: bool) -> None:
    method_name = "enable_adapter_layers" if enable else "disable_adapter_layers"
    method = getattr(model.base_model, method_name, None)
    if method is None:
        raise ValueError(
            f"Adapters of type {type(model.active_peft_config).__name__} cannot be enabled or disabled in place. "
            "Prompt learning methods such as prompt tuning, prefix tuning and P-tuning always run their virtual "
            "tokens, so there are no adapter layers to toggle."
        )
    method()


def _peft_model_enable_adapters(model: PeftModel) -> None:
    """``PeftModel`` toggles adapter layers on the tuner rather than on the model itself."""
    _peft_model_toggle_adapters(model, enable=True)


def _peft_model_disable_adapters(model: PeftModel) -> None:
    """``PeftModel`` toggles adapter layers on the tuner rather than on the model itself."""
    _peft_model_toggle_adapters(model, enable=False)


def _single_adapter_name(adapter_name: Any, method_name: str) -> str:
    """The transformers mixin accepts a list of adapter names, ``PeftModel`` addresses exactly one."""
    if isinstance(adapter_name, str):
        return adapter_name
    if isinstance(adapter_name, (list, tuple)) and len(adapter_name) == 1 and isinstance(adapter_name[0], str):
        return adapter_name[0]
    raise ValueError(
        f"`{method_name}` accepts a list of adapter names on a plain transformers model, but the underlying model "
        f"is a peft.PeftModel, which addresses a single adapter at a time. Pass one adapter name instead, calling "
        f"`{method_name}` once per adapter if needed. Got {adapter_name!r}."
    )


def _peft_model_set_adapter(model: PeftModel, adapter_name: str | list[str], **kwargs) -> None:
    """``PeftModel`` activates exactly one adapter, so a list of several names cannot be applied."""
    return model.set_adapter(_single_adapter_name(adapter_name, "set_adapter"), **kwargs)


def _peft_model_delete_adapter(model: PeftModel, adapter_names: str | list[str] | None = None, **kwargs) -> None:
    """``PeftModel.delete_adapter`` takes a single ``adapter_name``, not the transformers ``adapter_names``.

    ``PeftModel.delete_adapter`` forwards to ``self.base_model.delete_adapter(adapter_name=...)``. For the tuner
    based methods ``base_model`` is the tuner, which accepts that keyword, but for prompt learning ``base_model``
    is the raw transformers model, whose own mixin takes ``adapter_names``, so the forwarded call raises a
    ``TypeError``. ``peft`` implements no deletion path for prompt learning at all, so reject it with an
    explanation rather than letting that ``TypeError`` surface.
    """
    if adapter_names is None:
        adapter_names = kwargs.pop("adapter_name", None)
    adapter_name = _single_adapter_name(adapter_names, "delete_adapter")
    config = model.peft_config.get(adapter_name)
    if config is not None and config.is_prompt_learning:
        raise ValueError(
            f"Adapters of type {type(config).__name__} cannot be deleted from the underlying peft.PeftModel. "
            "peft only implements deletion for the tuner based methods; for prompt learning methods such as "
            "prompt tuning, prefix tuning and P-tuning it forwards the call to the base transformers model, "
            "which does not hold the prompt adapter."
        )
    return model.delete_adapter(adapter_name, **kwargs)


# ``peft.PeftModel`` and ``transformers.integrations.peft.PeftAdapterMixin`` expose overlapping but
# differently shaped APIs. Methods listed here are translated to their ``peft`` equivalent.
PEFT_MODEL_TRANSLATIONS = {
    "load_adapter": _peft_model_load_adapter,
    "add_adapter": _peft_model_add_adapter,
    "active_adapters": _peft_model_active_adapters,
    "active_adapter": _peft_model_active_adapter,
    "get_adapter_state_dict": _peft_model_get_adapter_state_dict,
    "enable_adapters": _peft_model_enable_adapters,
    "disable_adapters": _peft_model_disable_adapters,
    "set_adapter": _peft_model_set_adapter,
    "delete_adapter": _peft_model_delete_adapter,
}


def peft_wrapper(func):
    """Wrapper to call the method on the auto_model with a check for PEFT compatibility."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.check_peft_compatible_model()
        method_name = func.__name__
        model = self.transformers_model
        if (peft_model := as_peft_model(model)) is not None:
            # ``PeftModel.__getattr__`` forwards unknown attributes down to the wrapped transformers
            # model, so several of these methods resolve to the transformers PEFT mixin and then fail
            # because the adapters are registered on the wrapper instead. Translate them explicitly.
            if (translation := PEFT_MODEL_TRANSLATIONS.get(method_name)) is not None:
                return translation(peft_model, *args, **kwargs)
        if not hasattr(model, method_name):
            raise AttributeError(
                f"The underlying transformers model ({type(model).__name__}) does not have a "
                f"`{method_name}` method. This may indicate an incompatible or outdated version of transformers or peft."
            )
        method = getattr(model, method_name)
        return method(*args, **kwargs)

    return wrapper


class PeftAdapterMixin:
    """
    Wrapper Mixin that adds the functionality to easily load and use adapters on the model. For
    more details about adapters check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods follow those supported by transformers library,
    you can find more information on:
    https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin

    Models whose underlying module is already wrapped in a ``peft.PeftModel`` are supported as well,
    with the arguments translated to the ``peft`` equivalents where the two interfaces differ.
    """

    def has_peft_compatible_model(self) -> bool:
        model = self.transformers_model
        return isinstance(model, PeftAdapterMixinTransformers) or as_peft_model(model) is not None

    def check_peft_compatible_model(self) -> None:
        if not self.has_peft_compatible_model():
            raise ValueError(
                "PEFT methods are only supported for Sentence Transformer models that use the Transformer module."
            )

    @peft_wrapper
    def load_adapter(self, *args, **kwargs) -> None:
        """
        Load adapter weights from file or remote Hub folder." If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `load_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.load_adapter
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def add_adapter(self, *args, **kwargs) -> None:
        """
        Adds a fresh new adapter to the current model for training purposes. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Requires peft as a backend to load the adapter weights and the underlying model to be compatible with PEFT.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `add_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.add_adapter

        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def set_adapter(self, *args, **kwargs) -> None:
        """
        Sets a specific adapter by forcing the model to use that adapter and disable the other adapters.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `set_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.set_adapter
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def disable_adapters(self) -> None:
        """
        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def active_adapters(self) -> list[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def active_adapter(self) -> str: ...  # Implementation handled by the wrapper

    @peft_wrapper
    def get_adapter_state_dict(self, *args, **kwargs) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `get_adapter_state_dict` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.get_adapter_state_dict
        """
        ...  # Implementation handled by the wrapper

    @peft_wrapper
    def delete_adapter(self, *args, **kwargs) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Delete an adapter's LoRA layers from the underlying model.

        Args:
            *args:
                Positional arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter
            **kwargs:
                Keyword arguments to pass to the underlying AutoModel `delete_adapter` function. More information can be found in the transformers documentation
                https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.delete_adapter
        """
