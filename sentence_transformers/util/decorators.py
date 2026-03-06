from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


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


def deprecated_tokenizer_kwargs_decorator(func):
    """Decorator that renames the deprecated ``tokenizer_kwargs`` parameter to ``processor_kwargs``.

    Apply this decorator to public-facing ``__init__`` methods (e.g.
    :class:`SentenceTransformer`, :class:`SparseEncoder`, :class:`CrossEncoder`)
    so that callers who still pass ``tokenizer_kwargs`` receive a deprecation
    warning and have the value transparently forwarded as ``processor_kwargs``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "tokenizer_kwargs" in kwargs:
            tokenizer_kwargs = kwargs.pop("tokenizer_kwargs")
            logger.warning(
                "The `tokenizer_kwargs` argument was renamed and is now deprecated. "
                "Please use `processor_kwargs` instead."
            )
            if "processor_kwargs" not in kwargs:
                kwargs["processor_kwargs"] = tokenizer_kwargs

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
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
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

    * ``activation_fct`` -> ``activation_fn``
    * ``num_workers`` -> removed (no-op)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        kwargs_renamed_mapping = {
            "activation_fct": "activation_fn",
        }
        for old_name, new_name in kwargs_renamed_mapping.items():
            if old_name in kwargs:
                kwarg_value = kwargs.pop(old_name)
                logger.warning(
                    f"The CrossEncoder.predict `{old_name}` argument was renamed and is now deprecated. Please use `{new_name}` instead."
                )
                if new_name not in kwargs:
                    kwargs[new_name] = kwarg_value

        deprecated_args = ["num_workers"]

        for deprecated_arg in deprecated_args:
            if deprecated_arg in kwargs:
                kwargs.pop(deprecated_arg)
                logger.warning(
                    f"The CrossEncoder.predict `{deprecated_arg}` argument is deprecated and has no effect. It will be removed in a future version."
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
