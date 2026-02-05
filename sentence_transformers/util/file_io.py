from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.autonotebook import tqdm


class disabled_tqdm(tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/huggingface_hub/issues/1603"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


def is_sentence_transformer_model(
    model_name_or_path: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> bool:
    """
    Checks if the given model name or path corresponds to a SentenceTransformer model.

    Args:
        model_name_or_path (str): The name or path of the model.
        token (Optional[Union[bool, str]]): The token to be used for authentication. Defaults to None.
        cache_folder (Optional[str]): The folder to cache the model files. Defaults to None.
        revision (Optional[str]): The revision of the model. Defaults to None.
        local_files_only (bool): Whether to only use local files for the model. Defaults to False.

    Returns:
        bool: True if the model is a SentenceTransformer model, False otherwise.
    """
    return bool(
        load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
    )


def load_file_path(
    model_name_or_path: str,
    filename: str | Path,
    subfolder: str = "",
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    """
    Loads a file from a local or remote location.

    Args:
        model_name_or_path (str): The model name or path.
        filename (str): The name of the file to load.
        subfolder (str): The subfolder within the model subfolder (if applicable).
        token (Optional[Union[bool, str]]): The token to access the remote file (if applicable).
        cache_folder (Optional[str]): The folder to cache the downloaded file (if applicable).
        revision (Optional[str], optional): The revision of the file (if applicable). Defaults to None.
        local_files_only (bool, optional): Whether to only consider local files. Defaults to False.

    Returns:
        Optional[str]: The path to the loaded file, or None if the file could not be found or loaded.
    """
    # If file is local
    file_path = Path(model_name_or_path, subfolder, filename)
    if file_path.exists():
        return str(file_path)

    # If file is remote
    file_path = Path(subfolder, filename)
    try:
        return hf_hub_download(
            model_name_or_path,
            filename=file_path.name,
            subfolder=file_path.parent.as_posix(),
            revision=revision,
            library_name="sentence-transformers",
            token=token,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
    except Exception:
        return None


def load_dir_path(
    model_name_or_path: str,
    subfolder: str,
    token: bool | str | None = None,
    cache_folder: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> str | None:
    """
    Loads the subfolder path for a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        subfolder (str): The subfolder to load.
        token (Optional[Union[bool, str]]): The token for authentication.
        cache_folder (Optional[str]): The folder to cache the downloaded files.
        revision (Optional[str], optional): The revision of the model. Defaults to None.
        local_files_only (bool, optional): Whether to only use local files. Defaults to False.

    Returns:
        Optional[str]: The subfolder path if it exists, otherwise None.
    """
    if isinstance(subfolder, Path):
        subfolder = subfolder.as_posix()

    # If file is local
    dir_path = Path(model_name_or_path, subfolder)
    if dir_path.exists():
        return str(dir_path)

    download_kwargs = {
        "repo_id": model_name_or_path,
        "revision": revision,
        "allow_patterns": f"{subfolder}/**" if subfolder not in ["", "."] else None,
        "library_name": "sentence-transformers",
        "token": token,
        "cache_dir": cache_folder,
        "local_files_only": local_files_only,
        "tqdm_class": disabled_tqdm,
    }
    # Try to download from the remote
    try:
        repo_path = snapshot_download(**download_kwargs)
    except Exception:
        # Otherwise, try local (i.e. cache) only
        download_kwargs["local_files_only"] = True
        repo_path = snapshot_download(**download_kwargs)
    return Path(repo_path, subfolder)


def http_get(url: str, path: str) -> None:
    """Download a URL to a local file with a progress bar.

    The content is streamed in chunks and first written to a temporary
    ``"<path>_part"`` file, which is atomically moved to ``path`` once the
    download has completed successfully. Parent directories of ``path`` are
    created automatically if they do not exist.

    Args:
        url (str): The HTTP(S) URL to download.
        path (str): Destination file path on the local filesystem.

    Raises:
        ImportError: If the optional ``httpx`` dependency is not installed.
        httpx.HTTPStatusError: If the HTTP request returns a non-success status code.
        OSError: If the file cannot be written to ``path``.

    Returns:
        None
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx is required to use this function. Please install it via `pip install httpx`.")

    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)

    download_filepath = path + "_part"
    with httpx.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(
            unit="B", total=total, unit_scale=True, leave=False, desc=f"Downloading {os.path.basename(path)}"
        )

        try:
            with open(download_filepath, "wb") as file_binary:
                for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        progress.update(len(chunk))
                        file_binary.write(chunk)
            os.replace(download_filepath, path)
        except Exception:
            if os.path.exists(download_filepath):
                os.remove(download_filepath)
            raise
        finally:
            progress.close()
