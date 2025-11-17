from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
from huggingface_hub import list_repo_files

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

logger = logging.getLogger(__name__)


def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    """
    Wraps the save_pretrained method of a model to save to a subfolder.

    Args:
        _save_pretrained_fn: The original save_pretrained function
        subfolder: The subfolder to save to

    Returns:
        A wrapped function that saves to the specified subfolder
    """

    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper


def backend_should_export(
    load_path: Path,
    is_local: bool,
    model_kwargs: dict[str, Any],
    target_file_name: str,
    target_file_glob: str,
    backend_name: str,
) -> tuple[bool, dict[str, Any]]:
    """
    Determines whether the model should be exported to the backend, or if it can be loaded directly.
    Also update the `file_name` and `subfolder` model_kwargs if necessary.

    These are the cases:

    1. If export is set in model_kwargs, just return export
    2. If `<subfolder>/<file_name>` exists; set export to False
    3. If `<backend>/<file_name>` exists; set export to False and set subfolder to the backend (e.g. "onnx")
    4. If `<file_name>` contains a folder, add those folders to the subfolder and set the file_name to the last part

    We will warn if:

    1. The expected file does not exist in the model directory given the optional file_name and subfolder.
        If there are valid files for this backend, but they're don't align with file_name, then we give a useful warning.
    2. Multiple files are found in the model directory that match the target file name and the user did not
        specify the desired file name via `model_kwargs={"file_name": "<file_name>"}`

    Args:
        load_path: The model repository or directory, as a Path instance
        is_local: Whether the model is local or remote, i.e. whether load_path is a local directory
        model_kwargs: The model_kwargs dictionary. Notable keys are "export", "file_name", and "subfolder"
        target_file_name: The expected file name in the model directory, e.g. "model.onnx" or "openvino_model.xml"
        target_file_glob: The glob pattern to match the target file name, e.g. "*.onnx" or "openvino*.xml"
        backend_name: The human-readable name of the backend for use in warnings, e.g. "ONNX" or "OpenVINO"

    Returns:
        Tuple[bool, dict[str, Any]]: A tuple of the export boolean and the updated model_kwargs dictionary.
            Notable keys in model_kwargs are "export", "file_name", and "subfolder".
    """
    export = model_kwargs.pop("export", None)
    if export:
        return export, model_kwargs

    backend = backend_name.lower()
    file_name = model_kwargs.get("file_name", target_file_name)
    subfolder = model_kwargs.get("subfolder", None)
    primary_full_path = Path(subfolder, file_name).as_posix() if subfolder else Path(file_name).as_posix()
    secondary_full_path = (
        Path(subfolder, backend, file_name).as_posix() if subfolder else Path(backend, file_name).as_posix()
    )
    glob_pattern = f"{subfolder}/**/{target_file_glob}" if subfolder else f"**/{target_file_glob}"

    # Get the list of files in the model directory that match the target file name
    if is_local:
        model_file_names = [path.relative_to(load_path).as_posix() for path in load_path.glob(glob_pattern)]
    else:
        all_files = list_repo_files(
            load_path.as_posix(),
            repo_type="model",
            revision=model_kwargs.get("revision", None),
            token=model_kwargs.get("token", None),
        )
        model_file_names = [fname for fname in all_files if fnmatch(fname, glob_pattern)]

    # First check if the expected file exists in the root of the model directory
    # If it doesn't, check if it exists in the backend subfolder.
    # If it does, set the subfolder to include the backend
    model_found = primary_full_path in model_file_names
    if not model_found:
        model_found = secondary_full_path in model_file_names
        if model_found:
            if len(model_file_names) > 1 and "file_name" not in model_kwargs:
                logger.warning(
                    f"Multiple {backend_name} files found in {load_path.as_posix()!r}: {model_file_names}, defaulting to {secondary_full_path!r}. "
                    f'Please specify the desired file name via `model_kwargs={{"file_name": "<file_name>"}}`.'
                )
            model_kwargs["subfolder"] = Path(subfolder, backend).as_posix() if subfolder else backend
            model_kwargs["file_name"] = file_name
    if export is None:
        export = not model_found

    # If the file_name contains subfolders, set it as the subfolder instead
    file_name_parts = Path(file_name).parts
    if len(file_name_parts) > 1:
        model_kwargs["file_name"] = file_name_parts[-1]
        model_kwargs["subfolder"] = Path(model_kwargs.get("subfolder", ""), *file_name_parts[:-1]).as_posix()

    if export:
        logger.warning(f"No {file_name!r} found in {load_path.as_posix()!r}. Exporting the model to {backend_name}.")

        if model_file_names:
            logger.warning(
                f"If you intended to load one of the {model_file_names} {backend_name} files, "
                f'please specify the desired file name via `model_kwargs={{"file_name": "{model_file_names[0]}"}}`.'
            )

    return export, model_kwargs


def backend_warn_to_save(model_name_or_path: str, is_local: bool, backend_name: str) -> None:
    """
    Warns the user to save the model if they just exported it.

    Args:
        model_name_or_path: The model name or path
        is_local: Whether the model is local
        backend_name: The name of the backend (ONNX or OpenVINO)
    """
    to_log = f"Saving the exported {backend_name} model is heavily recommended to avoid having to export it again."
    if is_local:
        to_log += f" Do so with `model.save_pretrained({model_name_or_path!r})`."
    else:
        to_log += f" Do so with `model.push_to_hub({model_name_or_path!r}, create_pr=True)`."
    logger.warning(to_log)


def save_or_push_to_hub_model(
    export_function: Callable,
    export_function_name: str,
    config,
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
    backend: str = "onnx",
    model: SentenceTransformer | SparseEncoder | CrossEncoder | None = None,
):
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    if backend == "onnx":
        file_name = f"model_{file_suffix}.onnx"
    elif backend == "openvino":
        file_name = f"openvino_model_{file_suffix}.xml"

    with tempfile.TemporaryDirectory() as save_dir:
        export_function(save_dir)

        # OpenVINO models are saved in a nested directory
        if backend == "openvino":
            save_dir = Path(save_dir) / backend
            # and we need to attach the file_suffix for both the .xml and .bin files
            shutil.move(save_dir / "openvino_model.xml", save_dir / file_name)
            shutil.move(save_dir / "openvino_model.bin", (save_dir / file_name).with_suffix(".bin"))
            save_dir = save_dir.as_posix()

        # Because we upload folders and save_dir now has unnecessary files (tokenizer.json, config.json, etc.),
        # we move the main file to a nested directory
        if backend == "onnx":
            dst_dir = Path(save_dir) / backend
            dst_dir.mkdir(parents=True, exist_ok=True)
            source = Path(save_dir) / file_name
            destination = dst_dir / file_name
            shutil.move(source, destination)
            save_dir = dst_dir.as_posix()

        if push_to_hub:
            commit_description = ""
            if create_pr:
                opt_config_string = repr(config).replace("(", "(\n\t").replace(", ", ",\n\t").replace(")", "\n)")
                if isinstance(model, SparseEncoder):
                    commit_description = f"""\
Hello!

*This pull request has been automatically generated from the [`{export_function_name}`](https://sbert.net/docs/package_reference/util.html#sentence_transformers.backend.{export_function_name}) function from the Sentence Transformers library.*

## Config
```python
{opt_config_string}
```

## Tip:
Consider testing this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import SparseEncoder

# TODO: Fill in the PR number
pr_number = 2
model = SparseEncoder(
    "{model_name_or_path}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
    model_kwargs={{"file_name": "{file_name}"}},
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```
"""
                elif model is None or isinstance(model, SentenceTransformer):
                    commit_description = f"""\
Hello!

*This pull request has been automatically generated from the [`{export_function_name}`](https://sbert.net/docs/package_reference/util.html#sentence_transformers.backend.{export_function_name}) function from the Sentence Transformers library.*

## Config
```python
{opt_config_string}
```

## Tip:
Consider testing this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import SentenceTransformer

# TODO: Fill in the PR number
pr_number = 2
model = SentenceTransformer(
    "{model_name_or_path}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
    model_kwargs={{"file_name": "{file_name}"}},
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```
"""
                elif isinstance(model, CrossEncoder):
                    commit_description = f"""\
Hello!

*This pull request has been automatically generated from the [`{export_function_name}`](https://sbert.net/docs/package_reference/util.html#sentence_transformers.backend.{export_function_name}) function from the Sentence Transformers library.*

## Config
```python
{opt_config_string}
```

## Tip:
Consider testing this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import CrossEncoder

# TODO: Fill in the PR number
pr_number = 2
model = CrossEncoder(
    "{model_name_or_path}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
    model_kwargs={{"file_name": "{file_name}"}},
)

# Verify that everything works as expected
query = "Which planet is known as the Red Planet?"
passages = [
	"Venus is often called Earth's twin because of its similar size and proximity.",
	"Mars, known for its reddish appearance, is often referred to as the Red Planet.",
	"Jupiter, the largest planet in our solar system, has a prominent red spot.",
	"Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]

scores = model.predict([(query, passage) for passage in passages])
print(scores)
```
"""

            huggingface_hub.upload_folder(
                folder_path=save_dir,
                path_in_repo=backend,
                repo_id=model_name_or_path,
                repo_type="model",
                commit_message=f"Add exported {backend} model {file_name!r}",
                commit_description=commit_description,
                create_pr=create_pr,
            )

        else:
            dst_dir = Path(model_name_or_path) / backend
            # Create destination if it does not exist
            dst_dir.mkdir(parents=True, exist_ok=True)

            source = Path(save_dir) / file_name
            destination = dst_dir / file_name
            shutil.copy(source, destination)

            # OpenVINO has a second file to save: the .bin file
            if backend == "openvino":
                bin_source = (Path(save_dir) / file_name).with_suffix(".bin")
                bin_destination = (Path(dst_dir) / file_name).with_suffix(".bin")
                shutil.copy(bin_source, bin_destination)
