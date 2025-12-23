from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from sentence_transformers.util import load_dir_path, load_file_path


class Module(ABC, torch.nn.Module):
    """
    Base class for all modules in the Sentence Transformers library.

    This class provides a common interface for all modules, including methods for loading and saving the module's
    configuration and weights. It also provides a method for performing the forward pass of the module.

    Two abstract methods are defined in this class, which must be implemented by subclasses:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.forward`: The forward pass of the module.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save`: Save the module to disk.

    Optionally, you may also have to override:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.load`: Load the module from disk.

    To assist with loading and saving the module, several utility methods are provided:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_config`: Load the module's configuration from a JSON file.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_file_path`: Load a file from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_dir_path`: Load a directory from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_torch_weights`: Load the PyTorch weights of the module, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save_config`: Save the module's configuration to a JSON file.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save_torch_weights`: Save the PyTorch weights of the module.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.get_config_dict`: Get the module's configuration as a dictionary.

    And several class variables are defined to assist with loading and saving the module:

    - :attr:`sentence_transformers.sentence_transformer.models.Module.config_file_name`: The name of the configuration file used to save the module's configuration.
    - :attr:`sentence_transformers.sentence_transformer.models.Module.config_keys`: A list of keys used to save the module's configuration.
    - :attr:`sentence_transformers.sentence_transformer.models.Module.save_in_root`: Whether to save the module's configuration in the root directory of the model or in a subdirectory named after the module.
    """

    config_file_name: str = "config.json"
    """
    The name of the configuration file used to save the module's configuration. This file is used to initialize the
    module when loading it from a pre-trained model.
    """
    config_keys: list[str] = []
    """
    A list of keys used to save the module's configuration. These keys are used to save the module's configuration
    when saving the model to disk.
    """
    save_in_root: bool = False
    """
    Whether to save the module's configuration in the root directory of the model or in a subdirectory named after the module.
    """
    forward_kwargs: set[str] = set()
    """
    A set of keyword arguments that can be passed to the forward method of the module. These arguments are used to
    pass additional information from the model's encode method to the module's forward method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor | Any], **kwargs) -> dict[str, torch.Tensor | Any]:
        """
        Forward pass of the module. This method should be overridden by subclasses to implement the specific behavior of the module.

        The forward method takes a dictionary of features as input and returns a dictionary of features as output.
        The keys in the ``features`` dictionary depend on the position of the module in the model pipeline, as
        the ``features`` dictionary is passed from one module to the next. Common keys in the ``features`` dictionary
        are:

            - ``input_ids``: The input IDs of the tokens in the input text.
            - ``attention_mask``: The attention mask for the input tokens.
            - ``token_type_ids``: The token type IDs for the input tokens.
            - ``token_embeddings``: The token embeddings for the input tokens.
            - ``sentence_embedding``: The sentence embedding for the input text, i.e. pooled token embeddings.

        Optionally, the ``forward`` method can accept additional keyword arguments (``**kwargs``) that can be used to
        pass additional information from ``model.encode`` to this module.

        Args:
            features (dict[str, torch.Tensor | Any]): A dictionary of features to be processed by the module.
            **kwargs: Additional keyword arguments that can be used to pass additional information from ``model.encode``.

        Returns:
            dict[str, torch.Tensor | Any]: A dictionary of features after processing by the module.
        """

    def get_config_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary of the configuration parameters of the module.

        These parameters are used to save the module's configuration when saving the model to disk, and again used
        to initialize the module when loading it from a pre-trained model. The keys used in the dictionary are defined in the
        ``config_keys`` class variable.

        Returns:
            dict[str, Any]: A dictionary of the configuration parameters of the module.
        """
        return {key: getattr(self, key) for key in self.config_keys}

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        """
        Load this module from a model checkpoint. The checkpoint can be either a local directory or a model id on Hugging Face.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model on Hugging Face.
            subfolder (str, optional): The subfolder within the model directory to load from, e.g. ``"1_Pooling"``.
                Defaults to ``""``.
            token (bool | str | None, optional): The token to use for authentication when loading from Hugging Face.
                If None, tries to use a token saved using ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
                Defaults to None.
            cache_folder (str | None, optional): The folder to use for caching the model files.
                If None, uses the default cache folder for Hugging Face, ``~/.cache/huggingface``. Defaults to None.
            revision (str | None, optional): The revision of the model to load.
                If None, uses the latest revision. Defaults to None.
            local_files_only (bool, optional): Whether to only load local files. Defaults to False.
            **kwargs: Additional module-specific arguments used in an overridden ``load`` method, such as ``trust_remote_code``,
                ``model_kwargs``, ``tokenizer_kwargs``, ``config_kwargs``, ``backend``, etc.

        Returns:
            Self: The loaded module.
        """
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(**config)

    @classmethod
    def load_config(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        config_filename: str | None = None,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        """
        Load the configuration of the module from a model checkpoint. The checkpoint can be either a local directory or a model id on Hugging Face.
        The configuration is loaded from a JSON file, which contains the parameters used to initialize the module.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model on Hugging Face.
            subfolder (str, optional): The subfolder within the model directory to load from, e.g. ``"1_Pooling"``.
                Defaults to ``""``.
            config_filename (str | None, optional): The name of the configuration file to load.
                If None, uses the default configuration file name defined in the ``config_file_name`` class variable.
                Defaults to None.
            token (bool | str | None, optional): The token to use for authentication when loading from Hugging Face.
                If None, tries to use a token saved using ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
                Defaults to None.
            cache_folder (str | None, optional): The folder to use for caching the model files.
                If None, uses the default cache folder for Hugging Face, ``~/.cache/huggingface``. Defaults to None.
            revision (str | None, optional): The revision of the model to load.
                If None, uses the latest revision. Defaults to None.
            local_files_only (bool, optional): Whether to only load local files. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary of the configuration parameters of the module.
        """
        config_path = load_file_path(
            model_name_or_path=model_name_or_path,
            filename=config_filename or cls.config_file_name,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_path is None:
            return {}

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config

    @staticmethod
    def load_file_path(
        model_name_or_path: str,
        filename: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str | None:
        """
        A utility function to load a file from a model checkpoint. The checkpoint can be either a local directory or a model id on Hugging Face.
        The file is loaded from the specified subfolder within the model directory.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model on Hugging Face.
            filename (str): The name of the file to load.
            subfolder (str, optional): The subfolder within the model directory to load from, e.g. ``"1_Pooling"``.
                Defaults to ``""``.
            token (bool | str | None, optional): The token to use for authentication when loading from Hugging Face.
                If None, tries to use a token saved using ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
                Defaults to None.
            cache_folder (str | None, optional): The folder to use for caching the model files.
                If None, uses the default cache folder for Hugging Face, ``~/.cache/huggingface``. Defaults to None.
            revision (str | None, optional): The revision of the model to load.
                If None, uses the latest revision. Defaults to None.
            local_files_only (bool, optional): Whether to only load local files. Defaults to False.

        Returns:
            str | None: The path to the loaded file, or None if the file was not found.
        """
        return load_file_path(
            model_name_or_path=model_name_or_path,
            filename=filename,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @staticmethod
    def load_dir_path(
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        """
        A utility function to load a directory from a model checkpoint. The checkpoint can be either a local directory or a model id on Hugging Face.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model on Hugging Face.
            subfolder (str, optional): The subfolder within the model directory to load from, e.g. ``"1_Pooling"``.
                Defaults to ``""``.
            token (bool | str | None, optional): The token to use for authentication when loading from Hugging Face.
                If None, tries to use a token saved using ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
                Defaults to None.
            cache_folder (str | None, optional): The folder to use for caching the model files.
                If None, uses the default cache folder for Hugging Face, ``~/.cache/huggingface``. Defaults to None.
            revision (str | None, optional): The revision of the model to load.
                If None, uses the latest revision. Defaults to None.
            local_files_only (bool, optional): Whether to only load local files. Defaults to False.

        Returns:
            str: The path to the loaded directory.
        """
        return load_dir_path(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    @classmethod
    def load_torch_weights(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        model: Self | None = None,
    ):
        """
        A utility function to load the PyTorch weights of a model from a checkpoint. The checkpoint can be either a
        local directory or a model id on Hugging Face. The weights are loaded from either a ``model.safetensors``
        file or a ``pytorch_model.bin`` file, depending on which one is available. This method either loads the
        weights into the model or returns the weights as a state dictionary.

        Args:
            model_name_or_path (str): The path to the model directory or the name of the model on Hugging Face.
            subfolder (str, optional): The subfolder within the model directory to load from, e.g. ``"2_Dense"``.
                Defaults to ``""``.
            token (bool | str | None, optional): The token to use for authentication when loading from Hugging Face.
                If None, tries to use a token saved using ``huggingface-cli login`` or the ``HF_TOKEN`` environment variable.
                Defaults to None.
            cache_folder (str | None, optional): The folder to use for caching the model files.
                If None, uses the default cache folder for Hugging Face, ``~/.cache/huggingface``. Defaults to None.
            revision (str | None, optional): The revision of the model to load.
                If None, uses the latest revision. Defaults to None.
            local_files_only (bool, optional): Whether to only load local files. Defaults to False.
            model (Self | None, optional): The model to load the weights into. If None, returns the weights as a state
                dictionary. Defaults to None.

        Raises:
            ValueError: If neither a ``model.safetensors`` file nor a ``pytorch_model.bin`` file is found in the model
                checkpoint in the ``subfolder``.

        Returns:
            Self | dict[str, torch.Tensor]: The model with the loaded weights or the weights as a state dictionary,
                depending on the value of the ``model`` argument.
        """
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        # 1. Attempt to load a safetensors file from the local or remote directory
        safetensors_path = cls.load_file_path(model_name_or_path, filename="model.safetensors", **hub_kwargs)
        if safetensors_path is not None:
            # Either load the weights into the model or return the weights
            if model is not None:
                missing, unexpected = load_safetensors_model(model, safetensors_path, strict=False)
                missing_keys = ", ".join(f'"{k}"' for k in sorted(missing))
                unexpected_keys = ", ".join(f'"{k}"' for k in sorted(unexpected))
                error_lines = [f"Error(s) in loading state_dict for {model.__class__.__name__}:"]
                if missing:
                    error_lines.append(f"    Missing key(s) in state_dict: {missing_keys}")
                if unexpected:
                    error_lines.append(f"    Unexpected key(s) in state_dict: {unexpected_keys}")
                if missing or unexpected:
                    raise RuntimeError("\n".join(error_lines))
                return model
            else:
                weights = load_safetensors_file(safetensors_path)
                return weights

        # 2. If safetensors file is not found, attempt to load a pytorch model file
        # from the local or remote directory
        pytorch_model_path = cls.load_file_path(model_name_or_path, filename="pytorch_model.bin", **hub_kwargs)
        if pytorch_model_path is None:
            raise ValueError(f"Could not find 'model.safetensors' or 'pytorch_model.bin' in {model_name_or_path}.")

        weights = torch.load(pytorch_model_path, map_location=torch.device("cpu"), weights_only=True)
        if model is not None:
            model.load_state_dict(weights)
            return model
        return weights

    @abstractmethod
    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        """
        Save the module to disk. This method should be overridden by subclasses to implement the specific behavior of the module.

        Args:
            output_path (str): The path to the directory where the module should be saved.
            *args: Additional arguments that can be used to pass additional information to the save method.
            safe_serialization (bool, optional): Whether to use the safetensors format for saving the model weights.
                Defaults to True.
            **kwargs: Additional keyword arguments that can be used to pass additional information to the save method.
        """

    def save_config(self, output_path: str, filename: str | None = None) -> None:
        """
        Save the configuration of the module to a JSON file.

        Args:
            output_path (str): The path to the directory where the configuration file should be saved.
            filename (str | None, optional): The name of the configuration file. If None, uses the default configuration
                file name defined in the ``config_file_name`` class variable. Defaults to None.

        Returns:
            None
        """
        config = self.get_config_dict()
        config_output_path = os.path.join(output_path, filename or self.config_file_name)
        with open(config_output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def save_torch_weights(self, output_path: str, safe_serialization: bool = True) -> None:
        """
        Save the PyTorch weights of the module to disk.

        Args:
            output_path (str): The path to the directory where the weights should be saved.
            safe_serialization (bool, optional): Whether to use the safetensors format for saving the model weights.
                Defaults to True.

        Returns:
            None
        """
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_config_dict()})"
