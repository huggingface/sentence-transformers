from __future__ import annotations

import csv
import importlib
import logging
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
        >>> model = SentenceTransformer('all-MiniLM-L6-v2')
        >>> loss = MultipleNegativesRankingLoss(model)
        >>> fullname(loss)
        'sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'
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
        >>> import_from_string('sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss')
        <class 'sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss'>
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except Exception:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f'Module "{module_path}" does not define a "{class_name}" attribute/class'
        raise ImportError(msg)


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
