from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


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
