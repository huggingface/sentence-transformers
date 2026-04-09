from __future__ import annotations

import functools

from transformers.utils import logging as transformers_logging

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


def deprecated_kwargs(**renames: str):
    """Decorator factory that transparently renames deprecated keyword arguments.

    Emits a deprecation warning when a caller uses the old name. If both old and
    new names are provided, the old name is silently dropped in favor of the new.

    Usage::

        @deprecated_kwargs(sentence_embedding_dimension="embedding_dimension")
        def __init__(self, embedding_dimension: int, ...):
            ...

    Note: for backward-compatible loading of saved configs without warnings,
    use ``config_key_renames`` on the :class:`Module` subclass. This
    decorator is for warning users who explicitly pass the old keyword argument.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_name, new_name in renames.items():
                if old_name in kwargs:
                    logger.warning_once(
                        f"The `{old_name}` argument was renamed and is now deprecated. "
                        f"Please use `{new_name}` instead."
                    )
                    if new_name not in kwargs:
                        kwargs[new_name] = kwargs.pop(old_name)
                    else:
                        kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def transformer_kwargs_decorator(func):
    """Decorator for :class:`Transformer.__init__` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``model_args`` -> ``model_kwargs``
    * ``tokenizer_args`` -> ``processor_kwargs``
    * ``config_args`` -> ``config_kwargs``
    * ``cache_dir`` -> distributed into ``model_kwargs``, ``processor_kwargs``, and ``config_kwargs``
    """
    _RENAMED_KWARGS = {
        "model_args": "model_kwargs",
        "tokenizer_args": "processor_kwargs",
        "config_args": "config_kwargs",
    }

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for old_name, new_name in _RENAMED_KWARGS.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning(
                    f"The Transformer `{old_name}` argument was renamed and is now deprecated, "
                    f"please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        if "cache_dir" in kwargs:
            cache_dir = kwargs.pop("cache_dir")
            if cache_dir is not None:
                logger.warning(
                    "The Transformer `cache_dir` argument is deprecated. "
                    "Please pass `cache_dir` via `model_kwargs`, `processor_kwargs`, and/or `config_kwargs` instead."
                )
                for dict_name in ("model_kwargs", "processor_kwargs", "config_kwargs"):
                    kwargs.setdefault(dict_name, {})
                    kwargs[dict_name].setdefault("cache_dir", cache_dir)

        return func(*args, **kwargs)

    return wrapper


def cross_encoder_init_args_decorator(func):
    """Decorator for :class:`CrossEncoder.__init__` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``model_name`` -> ``model_name_or_path``
    * ``automodel_args`` -> ``model_kwargs``
    * ``tokenizer_args`` -> ``processor_kwargs``
    * ``tokenizer_kwargs`` -> ``processor_kwargs``
    * ``config_args`` -> ``config_kwargs``
    * ``cache_dir`` -> ``cache_folder``
    * ``default_activation_function`` -> ``activation_fn``
    * ``classifier_dropout`` -> ``config_kwargs["classifier_dropout"]``

    Also handles legacy positional arguments (``num_labels``, ``max_length``,
    ``activation_fn``, ``device``), which are now keyword-only.
    """

    # Old positional order after model_name_or_path (which is still positional):
    # num_labels, max_length, activation_fn, device
    _POSITIONAL_ARGS = ("num_labels", "max_length", "activation_fn", "device")

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Handle old-style positional arguments: in v5.3 and earlier, num_labels,
        # max_length, activation_fn, and device could be passed positionally.
        if args:
            # First positional arg is model_name_or_path (still positional), rest are legacy
            new_args = args[:1]
            for i, value in enumerate(args[1:]):
                if i < len(_POSITIONAL_ARGS):
                    name = _POSITIONAL_ARGS[i]
                    logger.warning(
                        f"Passing `{name}` as a positional argument to CrossEncoder is deprecated. "
                        f"Please use `{name}={value!r}` as a keyword argument instead."
                    )
                    if name not in kwargs:
                        kwargs[name] = value
            args = new_args

        kwargs_renamed_mapping = {
            "model_name": "model_name_or_path",
            "automodel_args": "model_kwargs",
            "tokenizer_args": "processor_kwargs",
            "tokenizer_kwargs": "processor_kwargs",
            "config_args": "config_kwargs",
            "cache_dir": "cache_folder",
            "default_activation_function": "activation_fn",
        }
        for old_name, new_name in kwargs_renamed_mapping.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning(
                    f"The CrossEncoder `{old_name}` argument was renamed and is now deprecated. Please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        if "classifier_dropout" in kwargs:
            classifier_dropout = kwargs.pop("classifier_dropout")
            logger.warning(
                f"The CrossEncoder `classifier_dropout` argument is deprecated. Please use `config_kwargs={{'classifier_dropout': {classifier_dropout}}}` instead."
            )
            if "config_kwargs" not in kwargs:
                kwargs["config_kwargs"] = {"classifier_dropout": classifier_dropout}
            else:
                kwargs["config_kwargs"]["classifier_dropout"] = classifier_dropout

        return func(self, *args, **kwargs)

    return wrapper


def cross_encoder_predict_rank_args_decorator(func):
    """Decorator for :class:`CrossEncoder.predict` / :class:`CrossEncoder.rank` that handles deprecated keyword arguments.

    Handles the following legacy kwargs:

    * ``sentences`` -> ``inputs`` (via :func:`deprecated_kwargs`)
    * ``activation_fct`` -> ``activation_fn``
    * ``num_workers`` -> removed (no-op)
    """

    @deprecated_kwargs(sentences="inputs", activation_fct="activation_fn")
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if "num_workers" in kwargs:
            kwargs.pop("num_workers")
            logger.warning(
                "The CrossEncoder.predict `num_workers` argument is deprecated and has no effect. "
                "It will be removed in a future version."
            )

        return func(self, *args, **kwargs)

    return wrapper


def save_to_hub_args_decorator(func):
    """
    A decorator to update the signature of the :class:`~sentence_transformers.base.model.BaseModel.save_to_hub` method
    to replace the deprecated `repo_name` argument with `repo_id`, and to introduce backwards compatibility for
    positional arguments despite a newly added `token` argument.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # If repo_id not already set, use repo_name
        repo_name = kwargs.pop("repo_name", None)
        if repo_name and "repo_id" not in kwargs:
            logger.warning(
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated. Please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper
