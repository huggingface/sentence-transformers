from __future__ import annotations

import csv
import importlib
import logging
import os
import warnings
from contextlib import contextmanager
from inspect import isclass


def fullname(obj) -> str:
    """
    Gives a full name (package_name.class_name) for a class / object in Python. Will
    be used to load the correct classes from JSON files

    Args:
        obj: The object for which to get the full name, e.g. an instance of a class or the class itself.

    Returns:
        str: The full name of the object.

    Example:
        >>> from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
        >>> from sentence_transformers import SentenceTransformer
        >>> from sentence_transformers.util import fullname
        >>> model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        >>> loss = MultipleNegativesRankingLoss(model)
        >>> fullname(loss)
        'sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking.MultipleNegativesRankingLoss'
    """
    if not isclass(obj):
        obj = obj.__class__
    module = obj.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__name__  # Avoid reporting __builtin__
    return module + "." + obj.__name__


def import_from_string(dotted_path: str) -> type:
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.

    Args:
        dotted_path (str): The dotted module path.

    Returns:
        Any: The attribute/class designated by the last name in the path.

    Raises:
        ImportError: If the import failed.

    Example:
        >>> import_from_string('sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking.MultipleNegativesRankingLoss')
        <class 'sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking.MultipleNegativesRankingLoss'>
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)

    # Suppress deprecation warnings: these imports come from model configs, not user code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            module = importlib.import_module(dotted_path)
        except Exception:
            module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        raise ImportError(msg)


def import_module_class(
    class_ref: str,
    model_name_or_path: str | None = None,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    code_revision: str | None = None,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    local_files_only: bool = False,
) -> type:
    """
    Resolve a module class reference to a class object.

    For class refs in the ``sentence_transformers.*`` namespace, this imports directly via
    :func:`import_from_string`. For other class refs (e.g. repository-local custom classes
    like ``modeling_my_model.CustomTransformer``), it first tries
    :func:`transformers.dynamic_module_utils.get_class_from_dynamic_module` to fetch the
    modeling file from the model directory, then falls back to :func:`import_from_string`
    if dynamic loading is not applicable or fails.

    Importing a class outside the ``sentence_transformers.*`` namespace runs code selected by the model's
    configuration (repository-hosted modeling code via dynamic loading, or a locally installed package via
    direct import). Loading such a class is fully trusted only when ``trust_remote_code`` is set, or when
    ``model_name_or_path`` resolves to a local directory (i.e. the user already has the file on disk and is
    implicitly trusted).

    .. deprecated:: 5.6
        The implicit trust of local directories is deprecated and will be removed in v6.0.
        From v6.0, loading repository-local custom code will require ``trust_remote_code=True``,
        matching the behavior for models loaded from the Hugging Face Hub. A ``FutureWarning`` is
        emitted whenever a local model is loaded via this short-circuit without
        ``trust_remote_code=True``.

    .. deprecated:: 5.7
        Importing a module class outside the ``sentence_transformers.*`` namespace from an untrusted,
        non-local model is deprecated. It currently emits a ``FutureWarning`` and will be refused from
        v6.0, when ``trust_remote_code=True`` becomes required for these imports.

    Args:
        class_ref: Dotted class path. Either a fully-qualified ``sentence_transformers.*``
            path or a repository-local reference like ``modeling_<name>.<ClassName>``.
        model_name_or_path: Hub repo id or local directory used to source repository-local
            modeling files. Required for dynamic loading.
        trust_remote_code: Whether to trust and import code selected by the model configuration. Permits
            dynamic loading from an unverified Hub repo, and (from v6.0) is required to import a
            non-``sentence_transformers.*`` class.
        revision: Hub revision to fetch the modeling file from.
        code_revision: Optional separate revision pinning for the modeling code (overrides
            ``revision`` when set).
        token: Hugging Face Hub authentication token. Required for fetching modeling files
            from private repositories.
        cache_folder: Optional override for the Hugging Face Hub cache directory.
        local_files_only: If True, only use cached files and never reach out to the Hub.

    Returns:
        The resolved class.
    """
    if class_ref.startswith("sentence_transformers."):
        return import_from_string(class_ref)

    is_local_dir = model_name_or_path is not None and os.path.isdir(model_name_or_path)
    trusted = trust_remote_code or is_local_dir
    if model_name_or_path is not None and trusted:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            module_class = get_class_from_dynamic_module(
                class_ref,
                model_name_or_path,
                revision=revision,
                code_revision=code_revision,
                token=token,
                cache_dir=cache_folder,
                local_files_only=local_files_only,
            )
        except (OSError, ValueError):
            # 1) the file does not exist, or 2) the class_ref is not correctly formatted/found
            pass
        else:
            # TODO(v6.0): remove the `or is_local_dir` short-circuit above and this warning,
            # requiring trust_remote_code for local custom code (#3801).
            if not trust_remote_code:
                warnings.warn(
                    f"Loading custom module {class_ref!r} from local path {model_name_or_path!r} without "
                    f"`trust_remote_code=True` is deprecated. It will require `trust_remote_code=True` from "
                    f"Sentence Transformers v6.0.",
                    FutureWarning,
                    stacklevel=2,
                )
            return module_class

    if model_name_or_path is not None and not trusted:
        # TODO(v6.0): raise here instead of warning, refusing untrusted module-class imports (#3801).
        warnings.warn(
            f"The model {model_name_or_path!r} references the module class {class_ref!r}, which is not part "
            f"of Sentence Transformers. Importing it executes third-party code, and will start requiring "
            f"`trust_remote_code=True` from Sentence Transformers v6.0. Pass `trust_remote_code=True` if you "
            f"trust {model_name_or_path!r} to reference a safe class.",
            FutureWarning,
            stacklevel=2,
        )
    return import_from_string(class_ref)


@contextmanager
def disable_datasets_caching():
    """
    A context manager that will disable caching in the datasets library.
    """
    from datasets import disable_caching, enable_caching, is_caching_enabled

    is_originally_enabled = is_caching_enabled()

    try:
        if is_originally_enabled:
            disable_caching()
        yield
    finally:
        if is_originally_enabled:
            enable_caching()


@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    Args:
        highest_level: the maximum logging level allowed.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def append_to_last_row(csv_path, additional_data):
    # Read the entire CSV file
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) > 1:  # Make sure there's at least one data row (after the header)
        # Append the additional data to the last row
        rows[-1].extend(additional_data)

        # Write the entire file back
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return True
    return False


# This is a list of edge cases that we don't want to prefix with "sentence-transformers/", "cross-encoder/", etc.
# despite not having a "/" in their name.
ORIGINAL_TRANSFORMER_MODELS = [
    "albert-base-v1",
    "albert-base-v2",
    "albert-large-v1",
    "albert-large-v2",
    "albert-xlarge-v1",
    "albert-xlarge-v2",
    "albert-xxlarge-v1",
    "albert-xxlarge-v2",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-uncased",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking",
    "bert-large-cased",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-uncased-whole-word-masking",
    "bert-large-uncased",
    "camembert-base",
    "ctrl",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert-base-uncased",
    "distilgpt2",
    "distilroberta-base",
    "gpt2-large",
    "gpt2-medium",
    "gpt2-xl",
    "gpt2",
    "openai-gpt",
    "roberta-base-openai-detector",
    "roberta-base",
    "roberta-large-mnli",
    "roberta-large-openai-detector",
    "roberta-large",
    "t5-11b",
    "t5-3b",
    "t5-base",
    "t5-large",
    "t5-small",
    "transfo-xl-wt103",
    "xlm-clm-ende-1024",
    "xlm-clm-enfr-1024",
    "xlm-mlm-100-1280",
    "xlm-mlm-17-1280",
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-roberta-base",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
]
