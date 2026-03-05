from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def deprecated_kwargs_decorator(kwargs_renamed_mapping: dict[str, str], class_name: str) -> callable:
    """A decorator that renames deprecated keyword arguments with a warning.

    Args:
        kwargs_renamed_mapping: Mapping of ``{old_name: new_name}`` for kwargs to rename.
        class_name: Name of the class to include in the deprecation warning.

    Example::

        @deprecated_kwargs_decorator({"tokenizer_args": "processor_kwargs"}, "Transformer")
        def __init__(self, model_name_or_path, processor_kwargs=None):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old_name, new_name in kwargs_renamed_mapping.items():
                if old_name in kwargs:
                    kwarg_value = kwargs.pop(old_name)
                    logger.warning(
                        f"The {class_name} `{old_name}` argument was renamed and is now deprecated, "
                        f"please use `{new_name}` instead."
                    )
                    if new_name not in kwargs:
                        kwargs[new_name] = kwarg_value
            return func(*args, **kwargs)

        return wrapper

    return decorator


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
                "Providing a `repo_name` keyword argument to `save_to_hub` is deprecated, please use `repo_id` instead."
            )
            kwargs["repo_id"] = repo_name

        # If positional args are used, adjust for the new "token" keyword argument
        if len(args) >= 2:
            args = (*args[:2], None, *args[2:])

        return func(self, *args, **kwargs)

    return wrapper
