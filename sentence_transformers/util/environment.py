from __future__ import annotations

import contextlib
import importlib
import logging
import os
from collections.abc import Generator
from importlib.metadata import PackageNotFoundError, metadata

import torch
from transformers import is_torch_npu_available

logger = logging.getLogger(__name__)

# Maps keywords found in dependency errors to the recommended sentence-transformers extra(s).
_DEPENDENCY_EXTRA_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("pillow", "pil"), 'pip install -U "sentence-transformers[image]"'),
    # When torchcodec is missing, transformers falls back to torchvision's removed `read_video`.
    # This must be checked before the generic "torchvision" hint below.
    (("read_video",), 'pip install -U "sentence-transformers[video]"  # or [audio] for audio-only models'),
    (("torchvision",), 'pip install -U "sentence-transformers[image]"'),
    (("torchcodec",), "pip install -U torchcodec"),
    (("soundfile", "librosa"), 'pip install -U "sentence-transformers[audio]"'),
]


@contextlib.contextmanager
def suggest_extra_on_exception() -> Generator[None, None, None]:
    """Re-raise ImportError/AttributeError with an install hint when a multimodal dependency is missing."""
    try:
        yield
    except (ImportError, AttributeError) as e:
        msg = str(e).lower()
        for keywords, hint in _DEPENDENCY_EXTRA_HINTS:
            if any(kw in msg for kw in keywords):
                raise type(e)(f"{str(e).strip()}\n\nTo install the required dependencies, run:\n{hint}") from e
        raise


def get_device_name() -> str:
    """
    Returns the name of the device where this module is running on.

    This function only supports single device or basic distributed training setups.
    In distributed mode for cuda device, it uses the rank to assign a specific CUDA device.

    Returns:
        str: Device name, like 'cuda:2', 'mps', 'npu', 'xpu', 'hpu', or 'cpu'
    """
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        elif torch.distributed.is_initialized() and torch.cuda.device_count() > torch.distributed.get_rank():
            local_rank = torch.distributed.get_rank()
        else:
            local_rank = 0
        return f"cuda:{local_rank}"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif importlib.util.find_spec("habana_frameworks") is not None:
        import habana_frameworks.torch.hpu as hthpu

        if hthpu.is_available():
            return "hpu"
    return "cpu"


def check_package_availability(package_name: str, owner: str) -> bool:
    """
    Checks if a package is available from the correct owner.
    """
    try:
        meta = metadata(package_name)
        home_page = meta["Home-page"]
        return meta["Name"] == package_name and home_page is not None and owner in home_page
    except PackageNotFoundError:
        return False


def is_accelerate_available() -> bool:
    """
    Returns True if the Huggingface accelerate library is available.
    """
    return check_package_availability("accelerate", "huggingface")


def is_datasets_available() -> bool:
    """
    Returns True if the Huggingface datasets library is available.
    """
    return check_package_availability("datasets", "huggingface")


def is_training_available() -> bool:
    """
    Returns True if we have the required dependencies for training Sentence
    Transformers models, i.e. Huggingface datasets and Huggingface accelerate.
    """
    return is_accelerate_available() and is_datasets_available()
