from __future__ import annotations

import contextlib
import importlib
import logging
import os
import platform
import sys
from collections.abc import Generator
from importlib.metadata import PackageNotFoundError, metadata, version
from typing import Any

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version
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


def is_dist_initialized() -> bool:
    """
    Returns whether ``torch.distributed`` is available and has been initialized.

    The availability check must come first: some PyTorch builds (e.g. ROCm or CPU-only) report
    ``torch.distributed.is_available() == False`` and do not expose APIs like ``is_initialized``, so calling
    them directly raises ``AttributeError``.
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


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
        elif is_dist_initialized() and torch.cuda.device_count() > torch.distributed.get_rank():
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


def get_installed_version(package_name: str) -> str | None:
    """
    Returns the version of an installed package, or None if the package can't be found.

    The ``__version__`` of an already imported module takes precedence over the distribution
    metadata, as the metadata of editable installs lags behind the code that actually runs.
    """
    if package_name == "python":
        return platform.python_version()

    module = sys.modules.get(package_name.replace("-", "_"))
    if isinstance(module_version := getattr(module, "__version__", None), str):
        return module_version

    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def check_version_requirements(requirements: dict[str, Any] | None, source: str | None = None) -> None:
    """
    Verifies that the installed packages satisfy the version requirements declared by a model.

    Requirements that can't be interpreted (e.g. an invalid specifier) are logged as a warning
    and skipped, so that a typo in a model configuration doesn't make the model unloadable.

    Args:
        requirements (dict[str, Any], optional): Mapping of package name to a `PEP 440
            <https://peps.python.org/pep-0440/#version-specifiers>`_ version specifier, e.g.
            ``{"transformers": ">=5.15", "peft": ">=0.18,<0.20"}``. Instead of a specifier, a value
            may also be a dictionary with a ``"specifier"`` and a ``"reason"`` describing what goes
            wrong if the requirement isn't met. The special ``"python"`` package name is matched
            against the running Python version.
        source (str, optional): The model name or path that declared the requirements, used in the
            error message. Defaults to None.

    Raises:
        ImportError: If one or more requirements aren't satisfied.
    """
    if not requirements:
        return
    if not isinstance(requirements, dict):
        logger.warning(f"Ignoring the `requirements` of {source!r}: expected a dictionary of version specifiers.")
        return

    # The `__version__` block in config_sentence_transformers.json spells torch as "pytorch",
    # so model authors are likely to use that name in their requirements too.
    aliases = {"pytorch": "torch"}

    unmet: list[str] = []
    install_specs: list[str] = []
    for package_name, requirement in requirements.items():
        package_name = aliases.get(package_name, package_name)
        is_dict = isinstance(requirement, dict)
        specifier = requirement.get("specifier", "") if is_dict else requirement
        reason = requirement.get("reason", "") if is_dict else ""

        if not isinstance(specifier, str):
            logger.warning(
                f"Could not verify the {specifier!r} requirement of {source!r}: expected a version specifier string."
            )
            continue

        installed_version = get_installed_version(package_name)
        # Parse both sides up front: anything unparsable must warn rather than raise.
        try:
            specifier_set = SpecifierSet(specifier, prereleases=True)
            if installed_version is not None and specifier_set.contains(Version(installed_version)):
                continue
        except Exception as e:
            logger.warning(f"Could not verify the {specifier!r} requirement of {source!r}: {e}")
            continue

        found = f"{package_name}=={installed_version} is installed" if installed_version else "it is not installed"
        unmet.append(f"- {package_name}{specifier}, but {found}. {reason}".rstrip())
        if package_name != "python":
            install_specs.append(f'"{package_name}{specifier}"')

    if unmet:
        lines = [f"The model {source!r} requires:" if source else "This model requires:", *unmet]
        if install_specs:
            lines.append(f"Install compatible versions with:\n    pip install -U {' '.join(install_specs)}")
        raise ImportError("\n".join(lines))
