from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Literal

import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import CardData, HfApi
from packaging import version
from torch import Tensor, nn
from transformers import PreTrainedModel, is_datasets_available, is_torch_npu_available
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_relative_import_files

from sentence_transformers import __version__
from sentence_transformers.base.evaluation import SentenceEvaluator
from sentence_transformers.base.modality import format_modality, infer_batch_modality
from sentence_transformers.base.modality_types import Modality, PairInput, SingleInput
from sentence_transformers.base.model_card import BaseModelCardData, generate_model_card
from sentence_transformers.base.modules import Module, Router, Transformer
from sentence_transformers.base.peft_mixin import PeftAdapterMixin
from sentence_transformers.util import (
    get_device_name,
    import_from_string,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
)
from sentence_transformers.util.misc import ORIGINAL_TRANSFORMER_MODELS

logger = logging.getLogger(__name__)


class BaseModel(nn.Sequential, PeftAdapterMixin, ABC):
    """
    Base class for SentenceTransformer, SparseEncoder, and CrossEncoder models.

    This class provides common functionality for:

    - Model loading (from Hub, local paths, or creating new models)
    - Model saving (to disk and Hub)
    - Device management
    - Module architecture (sequential composition)
    - Configuration management
    - Tokenizer/processor access

    All models inherit from nn.Sequential and are composed of a sequence of modules
    that are called sequentially in the forward pass.
    """

    # The dataclass used to generate the model card when saving the model.
    model_card_data_class = BaseModelCardData
    # The default Hugging Face organization to prepend to short model names.
    default_huggingface_organization: str | None = None
    # Default prompts to initialize for new model instances (e.g. {"query": "", "document": ""}).
    _default_prompts: dict[str, str] = {}
    # The placeholder model ID in model card templates that gets replaced with the actual model ID.
    _model_card_model_id_placeholder: str = "sentence_transformers_model_id"

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        modules: list[nn.Module] | OrderedDict[str, nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: CardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        """
        Initialize a BaseModel instance.

        Args:
            model_name_or_path: Model name or path to load from
            modules: List of modules to compose the model from
            device: Device to use for computation
            cache_folder: Folder to cache downloaded models
            trust_remote_code: Whether to trust remote code
            revision: Specific model version to use
            local_files_only: Whether to only use local files
            token: HuggingFace authentication token
            use_auth_token: Deprecated, use token instead
            model_kwargs: Additional model configuration parameters
            processor_kwargs: Additional processor/tokenizer configuration parameters
            config_kwargs: Additional config configuration parameters
            model_card_data: Model card data object
            backend: Backend to use for inference (torch, onnx, openvino)
            prompts: Dictionary of named prompts for the model
            default_prompt_name: Default prompt name to use if none is specified
        """
        default_prompts = dict(self._default_prompts)
        if prompts:
            default_prompts.update(prompts)
        self.prompts = default_prompts
        self.default_prompt_name = default_prompt_name
        self.trust_remote_code = trust_remote_code
        self.model_card_data = model_card_data or self.model_card_data_class(local_files_only=local_files_only)
        self.module_kwargs = None
        self._model_card_vars = {}
        self._model_card_text = None
        self.model_type = self.__class__.__name__
        self.backend = backend

        # Handle deprecated use_auth_token
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v4 of SentenceTransformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "Both `token` and `use_auth_token` are specified. Please only specify the `token` argument."
                )
            token = use_auth_token

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")

        # Determine device
        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device_name: {device}")

        if device == "hpu" and importlib.util.find_spec("optimum") is not None:
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()

        # Load model
        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(f"Load pretrained {self.__class__.__name__}: {model_name_or_path}")

            if not os.path.exists(model_name_or_path):
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise FileNotFoundError(f"Path {model_name_or_path} not found")

                if (
                    self.default_huggingface_organization is not None
                    and "/" not in model_name_or_path
                    and model_name_or_path.lower() not in ORIGINAL_TRANSFORMER_MODELS
                ):
                    model_name_or_path = f"{self.default_huggingface_organization}/{model_name_or_path}"

            modules, self.module_kwargs = self._load_modules(
                model_name_or_path,
                token=token,
                cache_folder=cache_folder,
                revision=revision,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                config_kwargs=config_kwargs,
            )

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        # Ensure all tensors in the model are of the same dtype as the first tensor
        try:
            first_parameter_dtype = next(self.parameters()).dtype
            self.to(first_parameter_dtype)
        except StopIteration:
            pass

        self.to(device)
        self.is_hpu_graph_enabled = False

        # Pass the model to the model card data for later use
        self.model_card_data.register_model(self)

        # Validate prompts after model loading (which may have merged config prompts)
        self._validate_prompts()

    def _validate_prompts(self) -> None:
        """Validate prompt configuration and log prompt information."""
        if self.default_prompt_name is not None and self.default_prompt_name not in self.prompts:
            raise ValueError(
                f"Default prompt name '{self.default_prompt_name}' not found in the configured prompts "
                f"dictionary with keys {list(self.prompts.keys())!r}."
            )

        if non_empty_keys := [k for k, v in self.prompts.items() if v != ""]:
            if len(non_empty_keys) == 1:
                logger.info(f"1 prompt is loaded, with the key: {non_empty_keys[0]}")
            else:
                logger.info(f"{len(non_empty_keys)} prompts are loaded, with the keys: {non_empty_keys}")
        if self.default_prompt_name:
            logger.warning_once(
                f"Default prompt name is set to '{self.default_prompt_name}'. "
                f"This prompt will be applied to all inference calls, except if "
                f"a `prompt` or `prompt_name` parameter is provided."
            )

    def _resolve_prompt(self, prompt: str | None, prompt_name: str | None) -> str | None:
        """Resolve a prompt from a prompt name or the default prompt name.

        Args:
            prompt: An explicit prompt string, or None.
            prompt_name: A key into ``self.prompts``, or None.

        Returns:
            The resolved prompt string, or None if no prompt applies.
        """
        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        elif prompt_name is not None:
            logger.warning(
                "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                "Ignoring the `prompt_name` in favor of `prompt`."
            )
        return prompt

    def get_backend(self) -> Literal["torch", "onnx", "openvino"]:
        """Return the backend used for inference, which can be one of "torch", "onnx", or "openvino".

        Returns:
            str: The backend used for inference.
        """
        return self.backend

    @property
    def modalities(self) -> list[Modality]:
        """Return the list of modalities supported by this model."""
        return getattr(self[0], "modalities", ["text"])

    def supports(self, modality: Modality) -> bool:
        """Check if the model supports the given modality.

        A modality is supported if:
        1. It is directly listed in :attr:`modalities`, or
        2. It is a tuple of modalities whose individual parts are all in :attr:`modalities`
           **and** ``"message"`` is in :attr:`modalities` (since the model uses message format
           to combine multiple modalities).

        Args:
            modality: A single modality string (e.g. ``"text"``, ``"image"``) or a tuple
                of modality strings (e.g. ``("image", "text")``).

        Returns:
            bool: Whether the model supports the given modality.

        Example::

            >>> from sentence_transformers import SentenceTransformer
            >>> model = SentenceTransformer("all-MiniLM-L6-v2")
            >>> model.supports("text")
            True
            >>> model.supports("image")
            False
        """
        supported = self.modalities
        if modality in supported:
            return True
        if isinstance(modality, tuple) and "message" in supported:
            return all(part in supported for part in modality)
        return False

    def get_model_kwargs(self) -> list[str]:
        """
        Get the keyword arguments specific to this model for the `encode`, `encode_query`, or `encode_document` methods.

        Example:

            >>> from sentence_transformers import SentenceTransformer, SparseEncoder
            >>> SentenceTransformer("all-MiniLM-L6-v2").get_model_kwargs()
            []
            >>> SentenceTransformer("jinaai/jina-embeddings-v4", trust_remote_code=True).get_model_kwargs()
            ['task', 'truncate_dim']
            >>> SparseEncoder("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill").get_model_kwargs()
            ['task']

        Returns:
            list[str]: A list of keyword arguments for the forward pass.
        """
        modules = list(self.named_children())
        forward_kwargs = set()
        while modules:
            module_name, module = modules.pop()
            if isinstance(module, Router):
                for route_modules in module.sub_modules.values():
                    modules.extend(list(route_modules.named_children()))
            if self.module_kwargs and module_name in self.module_kwargs:
                forward_kwargs.update(self.module_kwargs[module_name])
            if hasattr(module, "forward_kwargs"):
                forward_kwargs.update(module.forward_kwargs)
        return list(forward_kwargs)

    def get_max_seq_length(self) -> int | None:
        """
        Returns the maximal sequence length that the model accepts. Longer inputs will be truncated.

        Returns:
            Optional[int]: The maximal sequence length that the model accepts, or None if it is not defined.
        """
        if hasattr(self._first_module(), "max_seq_length"):
            return self._first_module().max_seq_length

        return None

    def _first_module(self) -> torch.nn.Module:
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self) -> torch.nn.Module:
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def _text_length(self, text: list[int] | list[list[int]]) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(text["input_ids"])
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Forward pass through all modules in the model."""
        for module_name, module in self.named_children():
            module_kwargs = {}
            if isinstance(module, Router):
                module_kwargs = kwargs
            else:
                module_kwarg_keys = []
                if self.module_kwargs is not None:
                    module_kwarg_keys = self.module_kwargs.get(module_name, [])
                module_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in module_kwarg_keys or (hasattr(module, "forward_kwargs") and key in module.forward_kwargs)
                }
            input = module(input, **module_kwargs)
        return input

    def preprocess(
        self,
        inputs: list[SingleInput | PairInput],
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Tensor | Any]:
        """
        Preprocesses the texts.

        Args:
            inputs (list[str | list[str] | tuple[str, str] | dict[str, Any] | PIL.Image | np.ndarray | torch.Tensor]): A list of inputs to be preprocessed.
                If a single input is provided, it must be wrapped in a list.

        Returns:
            Dict[str, Tensor]: A dictionary of tensors with the preprocessed texts.
        """
        # Validate that the inputs match a supported modality.
        # If "message" is supported, any modality is allowed since the input module
        # can convert it to message format (e.g. wrapping images in chat messages).
        modality = None
        if inputs:
            try:
                modality = infer_batch_modality(inputs)
            except (ValueError, TypeError):
                pass
            else:
                if not self.supports(modality):
                    supported = ", ".join(format_modality(m) for m in self.modalities)
                    message = (
                        f"Modality '{format_modality(modality)}' is not supported by this {type(self).__name__} model. "
                        f"Supported modalities: {supported}"
                    )
                    if isinstance(modality, tuple) and all(part in self.modalities for part in modality):
                        message += (
                            f"\nThis model supports {' and '.join(modality)} individually, "
                            "but not in the same input. Please encode each modality separately."
                        )
                    raise ValueError(message)

        # Backwards compatibility: fall back to preprocess/tokenize without prompt if the
        # input module doesn't accept it. Only the main path (preprocess with prompt) will
        # be supported in the future.
        try:
            preprocessed = self[0].preprocess(inputs, prompt=prompt, **kwargs)
        except TypeError:
            if prompt and modality == "text":
                inputs = [prompt + inp for inp in inputs]  # type: ignore[operator]
            preprocessed = self[0].preprocess(inputs, **kwargs)
        except AttributeError:
            if prompt and modality == "text":
                inputs = [prompt + inp for inp in inputs]  # type: ignore[operator]
            try:
                preprocessed = self[0].tokenize(inputs, **kwargs)
            except TypeError:
                preprocessed = self[0].tokenize(inputs)

        return preprocessed

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, Tensor]:
        """
        .. deprecated::
            `tokenize` is deprecated and will be removed in a future version. Use `preprocess` instead.
        """
        import warnings

        warnings.warn(
            "The `tokenize` method is deprecated and will be removed in a future version. "
            "Please use `preprocess` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.preprocess(inputs=texts, **kwargs)

    def is_singular_input(self, inputs: Any) -> bool:
        """
        Check if the input represents a single example or a batch of examples.

        Args:
            inputs: The input to check.
        Returns:
            bool: True if the input is a single example, False if it is a batch.
        """
        list_types = (list, tuple)
        if is_datasets_available():
            from datasets import Column

            list_types += (Column,)
        return not isinstance(inputs, list_types)

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded again.

        Args:
            path (str): Path on disk where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        logger.info(f"Save model to {path}")
        modules_config = []

        # Save model-level configuration options
        config = self._get_model_config()
        with open(os.path.join(path, "config_sentence_transformers.json"), "w", encoding="utf8") as fOut:
            json.dump(config, fOut, indent=2)

        # Save modules
        for idx, name in enumerate(self._modules):
            module: Module = self._modules[name]
            if (
                idx == 0 and hasattr(module, "save_in_root") and module.save_in_root
            ):  # Save first module in the main folder
                model_path = os.path.join(path, "")
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            # Try to save with safetensors, but fall back to the traditional PyTorch way if the module doesn't support it
            try:
                module.save(model_path, safe_serialization=safe_serialization)
            except TypeError:
                module.save(model_path)

            class_ref = type(module).__module__
            # For remote modules, we want to remove "transformers_modules.{repo_name}":
            if class_ref.startswith("transformers_modules."):
                class_file = sys.modules[class_ref].__file__

                # Save the custom module file
                dest_file = Path(model_path) / (Path(class_file).name)
                shutil.copy(class_file, dest_file)

                # Save all files imported in the custom module file
                for needed_file in get_relative_import_files(class_file):
                    dest_file = Path(model_path) / (Path(needed_file).name)
                    shutil.copy(needed_file, dest_file)

                # For remote modules, we want to ignore the "transformers_modules.{repo_id}" part,
                # i.e. we only want the filename
                class_ref = f"{class_ref.split('.')[-1]}.{type(module).__name__}"
            else:
                class_ref = f"{class_ref}.{type(module).__name__}"

            module_config = {"idx": idx, "name": name, "path": os.path.basename(model_path), "type": class_ref}
            if self.module_kwargs and name in self.module_kwargs and (module_kwargs := self.module_kwargs[name]):
                module_config["kwargs"] = module_kwargs
            modules_config.append(module_config)

        with open(os.path.join(path, "modules.json"), "w", encoding="utf8") as fOut:
            json.dump(modules_config, fOut, indent=2)

        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def _get_model_config(self) -> dict[str, Any]:
        # TODO: Consider sorting alphabetically
        return {
            "model_type": self.model_type,
            "__version__": {
                "sentence_transformers": __version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            },
            "prompts": self.prompts,
            "default_prompt_name": self.default_prompt_name,
        }

    def save_pretrained(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded again.

        Args:
            path (str): Path on disk where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        self.save(
            path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    def _update_default_model_id(self, model_card: str) -> str:
        """Update the default model ID in the model card."""
        if self.model_card_data.model_id:
            model_card = model_card.replace(
                f'model = {self.__class__.__name__}("{self._model_card_model_id_placeholder}"',
                f'model = {self.__class__.__name__}("{self.model_card_data.model_id}"',
            )
        return model_card

    def _create_model_card(
        self, path: str, model_name: str | None = None, train_datasets: list[str] | None = "deprecated"
    ) -> None:
        """
        Create an automatic model card and store it in the specified path.

        Args:
            path (str): The path where the model card will be stored.
            model_name (Optional[str], optional): The name of the model. Defaults to None.
            train_datasets (Optional[List[str]], optional): Deprecated argument. Defaults to "deprecated".

        Returns:
            None
        """
        if model_name:
            model_path = Path(model_name)
            if not model_path.exists() and not self.model_card_data.model_id:
                self.model_card_data.model_id = model_name

        # If we loaded a model from the Hub, and no training was done, then
        # we don't generate a new model card, but reuse the old one instead.
        if self._model_card_text and "generated_from_trainer" not in self.model_card_data.tags:
            model_card = self._model_card_text
            model_card = self._update_default_model_id(model_card)
        else:
            try:
                model_card = generate_model_card(self)
            except Exception:
                logger.error(
                    f"Error while generating model card:\n{traceback.format_exc()}"
                    "Consider opening an issue on https://github.com/huggingface/sentence-transformers/issues with this traceback.\n"
                    "Skipping model card creation."
                )
                return

        with open(os.path.join(path, "README.md"), "w", encoding="utf8") as fOut:
            fOut.write(model_card)

    @save_to_hub_args_decorator
    def save_to_hub(
        self,
        repo_id: str,
        organization: str | None = None,
        token: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        commit_message: str = "Add new model.",
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
    ) -> str:
        """
        DEPRECATED, use `push_to_hub` instead.

        Uploads all elements of this model to a new HuggingFace Hub repository.

        Args:
            repo_id (str): Repository name for your model in the Hub, including the user or organization.
            token (str, optional): An authentication token (See https://huggingface.co/settings/token)
            private (bool, optional): Set to true, for hosting a private model
            safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way
            commit_message (str, optional): Message to commit while pushing.
            local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
            exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
            replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card
            train_datasets (List[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.

        Returns:
            str: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        logger.warning(
            "The `save_to_hub` method is deprecated and will be removed in a future version of SentenceTransformers."
            " Please use `push_to_hub` instead for future model uploads."
        )

        if organization:
            if "/" not in repo_id:
                logger.warning(
                    f'Providing an `organization` to `save_to_hub` is deprecated. Please use `repo_id="{organization}/{repo_id}"` instead.'
                )
                repo_id = f"{organization}/{repo_id}"
            elif repo_id.split("/")[0] != organization:
                raise ValueError(
                    "Providing an `organization` to `save_to_hub` is deprecated. Please only use `repo_id`."
                )
            else:
                logger.warning(
                    f'Providing an `organization` to `save_to_hub` is deprecated. Please only use `repo_id="{repo_id}"` instead.'
                )

        return self.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            safe_serialization=safe_serialization,
            commit_message=commit_message,
            local_model_path=local_model_path,
            exist_ok=exist_ok,
            replace_model_card=replace_model_card,
            train_datasets=train_datasets,
        )

    def push_to_hub(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        commit_message: str | None = None,
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
        revision: str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Uploads all elements of this model to a new HuggingFace Hub repository.

        Args:
            repo_id (str): Repository name for your model in the Hub, including the user or organization.
            token (str, optional): An authentication token (See https://huggingface.co/settings/token)
            private (bool, optional): Set to true, for hosting a private model
            safe_serialization (bool, optional): If true, save the model using safetensors. If false, save the model the traditional PyTorch way
            commit_message (str, optional): Message to commit while pushing.
            local_model_path (str, optional): Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
            exist_ok (bool, optional): If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
            replace_model_card (bool, optional): If true, replace an existing model card in the hub with the automatically created model card
            train_datasets (List[str], optional): Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
            revision (str, optional): Branch to push the uploaded files to
            create_pr (bool, optional): If True, create a pull request instead of pushing directly to the main branch

        Returns:
            str: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        api = HfApi(token=token)
        repo_url = api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=None,
            exist_ok=exist_ok or create_pr,
        )
        repo_id = repo_url.repo_id  # Update the repo_id in case the old repo_id didn't contain a user or organization
        self.model_card_data.set_model_id(repo_id)

        if commit_message is None:
            if "generated_from_trainer" in self.model_card_data.tags:
                commit_message = "Add new model."
            else:
                commit_message = f"Uploading {self.__class__.__name__} model."

        commit_description = ""
        if create_pr:
            commit_description += (
                "# Description\n\n"
                + "Add model files with automated model upload from `push_to_hub`.\n\n"
                + "Note that this PR might need to be manually reviewed and merged due to:"
                + "1. Model card may need manual updates\n"
                + "2. Training details may need verification\n"
                + "3. Evaluation results may need validation\n\n"
                + "---\n\n"
            )

            # Create a draft PR with space to add details
            user_input_description = (
                "<details>\n"
                + "<summary>Click here to add additional details about this model</summary>\n\n"
                + "### Model Details\n"
                + "<!-- Add details about the model here -->\n\n"
                + "### Training Details\n"
                + "<!-- Add details about how the model was trained here -->\n\n"
                + "### Evaluation Results\n"
                + "<!-- Add evaluation results here -->\n\n"
                + "</details>"
            )
            commit_description += user_input_description

        if local_model_path:
            folder_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=local_model_path,
                commit_message=commit_message,
                commit_description=commit_description if create_pr else None,
                create_pr=create_pr,
                revision=revision,
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                create_model_card_for_path = not replace_model_card or not os.path.exists(
                    os.path.join(tmp_dir, "README.md")
                )
                self.save(
                    tmp_dir,
                    model_name=repo_id,
                    create_model_card=create_model_card_for_path,
                    train_datasets=train_datasets,
                    safe_serialization=safe_serialization,
                )
                folder_url = api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmp_dir,
                    commit_message=commit_message,
                    commit_description=commit_description if create_pr else None,
                    create_pr=create_pr,
                    revision=revision,
                )

        if create_pr:
            logger.info(f"A pull request has been created at {folder_url.pr_url}")
            # TODO: Check backwards compatibility, this previously returned the commit URL only it seems
            return folder_url.pr_url

        return folder_url.commit_url

    def _load_modules(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        load_kwargs = {
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            "model_kwargs": model_kwargs,
            "processor_kwargs": processor_kwargs,
            "config_kwargs": config_kwargs,
        }

        # Check if this is a Sentence Transformer model
        modules_json_path = load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if modules_json_path is None:
            return self._load_default_modules(model_name_or_path, **load_kwargs)

        model_type_being_loaded = self._get_model_type(
            model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if model_type_being_loaded == self.model_type:
            return self._load_config_modules(model_name_or_path, **load_kwargs)

        return self._load_converted_modules(model_name_or_path, **load_kwargs, model_type=model_type_being_loaded)

    @abstractmethod
    def _load_default_modules(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        """

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            processor_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the processor/tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.
            has_modules (bool, optional): Whether the model has modules.json. Defaults to False.

        Returns:
            List[nn.Module]: A list containing the transformer model and the pooling model.
        """

    def _load_config_modules(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        """
        Loads a full model using the modules.json file.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            processor_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the processor/tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            Tuple[OrderedDict[str, nn.Module], OrderedDict[str, Any]]: An ordered dictionary containing the modules of the model and their kwargs.
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_sentence_transformers_json_path is not None:
            with open(config_sentence_transformers_json_path, encoding="utf8") as fIn:
                model_config = json.load(fIn)

            if (
                "__version__" in model_config
                and "sentence_transformers" in model_config["__version__"]
                and version.parse(model_config["__version__"]["sentence_transformers"]) > version.parse(__version__)
            ):
                logger.warning(
                    f"You are trying to use a model that was created with Sentence Transformers version {model_config['__version__']['sentence_transformers']}, "
                    f"but you're currently using version {__version__}. This might cause unexpected behavior or errors. "
                    "In that case, try to update to the latest version."
                )

            self._parse_model_config(model_config)

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path,
            "README.md",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if model_card_path is not None:
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        # Load the modules
        modules_json_path = load_file_path(
            model_name_or_path,
            "modules.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        with open(modules_json_path, encoding="utf8") as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        module_kwargs = OrderedDict()
        for module_config in modules_config:
            class_ref = module_config["type"]
            module_class: Module = self._load_module_class_from_ref(
                class_ref, model_name_or_path, trust_remote_code, revision, model_kwargs
            )

            # Backwards compatibility: if the module is older and its `load` method only supports one parameter,
            # a path to a local directory containing the module files, then we load it with the old style
            load_signature = inspect.signature(module_class.load)
            # Check if the `load` method only accepts a single parameter (the path to the local directory).
            # This indicates an older module that does not support the newer loading method with multiple arguments.
            if len(load_signature.parameters) == 1:
                signature = inspect.signature(module_class.__init__)
                # Detect Transformer-based modules by checking for model/config kwargs in __init__.
                # Old custom modules (e.g. jinaai/jina-embeddings-v3) use model_args/config_args;
                # new-style modules use model_kwargs/config_kwargs.
                init_params = set(signature.parameters)
                _NEW_TO_OLD = {
                    "model_kwargs": "model_args",
                    "processor_kwargs": "tokenizer_args",
                    "config_kwargs": "config_args",
                }
                uses_old_names = {"model_args", "config_args"} <= init_params
                uses_new_names = {"model_kwargs", "config_kwargs"} <= init_params
                if uses_new_names or uses_old_names:
                    init_kwargs = Transformer._load_init_kwargs(
                        model_name_or_path,
                        # Loading-specific keyword arguments
                        subfolder=module_config["path"],
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                        # Module-specific keyword arguments
                        trust_remote_code=trust_remote_code,
                        model_kwargs=model_kwargs,
                        processor_kwargs=processor_kwargs,
                        config_kwargs=config_kwargs,
                        backend=self.backend,
                    )

                    # Remap new-style keys back to old-style for old custom modules.
                    if uses_old_names and not uses_new_names:
                        for new_name, old_name in _NEW_TO_OLD.items():
                            if new_name in init_kwargs:
                                init_kwargs[old_name] = init_kwargs.pop(new_name)

                    module = module_class(model_name_or_path, **init_kwargs)

                else:
                    # Old modules that don't support the new loading method and don't seem Transformer-based
                    # are loaded by downloading the full directories and calling .load() with the old style
                    # (i.e. only a path to the local directory)
                    local_path = load_dir_path(
                        model_name_or_path=model_name_or_path,
                        subfolder=module_config["path"],
                        token=token,
                        cache_folder=cache_folder,
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                    module = module_class.load(local_path)

            else:
                # Newer modules that support the new loading method are loaded with the new style
                # i.e. with many keyword arguments that can optionally be used by the modules
                module = module_class.load(
                    model_name_or_path,
                    # Loading-specific keyword arguments
                    subfolder=module_config["path"],
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                    # Module-specific keyword arguments
                    trust_remote_code=trust_remote_code,
                    model_kwargs=model_kwargs,
                    processor_kwargs=processor_kwargs,
                    config_kwargs=config_kwargs,
                    backend=self.backend,
                )

            modules[module_config["name"]] = module
            module_kwargs[module_config["name"]] = module_config.get("kwargs", [])

        if revision is None:
            path_parts = Path(modules_json_path)
            if len(path_parts.parts) >= 2:
                revision_path_part = Path(modules_json_path).parts[-2]
                if len(revision_path_part) == 40:
                    revision = revision_path_part
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, module_kwargs

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        # Only update prompts that aren't already set by the user or defaults
        for prompt_name, prompt_text in model_config.get("prompts", {}).items():
            if prompt_name not in self.prompts or not self.prompts[prompt_name]:
                self.prompts[prompt_name] = prompt_text
        if not self.default_prompt_name:
            self.default_prompt_name = model_config.get("default_prompt_name", None)

    def _load_converted_modules(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_type: str | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        return self._load_default_modules(
            model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
        )

    def _get_model_type(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str | None:
        """
        Retrieves the model_type from the config_sentence_transformers.json file.

        This is used to determine whether the model being loaded matches the current class
        (e.g., a SentenceTransformer model loaded with SentenceTransformer, or a SparseEncoder model
        loaded with SparseEncoder). When the model type doesn't match, we switch to a converted
        loading method to ensure compatibility.

        Defaults to "SentenceTransformer" if the config file is missing or has no "model_type" key,
        for backwards compatibility with older models.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.

        Returns:
            str: The model type, e.g. "SentenceTransformer", "SparseEncoder", or "CrossEncoder".
        """
        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        if config_sentence_transformers_json_path is None:
            return "SentenceTransformer"

        with open(config_sentence_transformers_json_path, encoding="utf8") as fIn:
            config = json.load(fIn)
            # Older SentenceTransformer models won't have "model_type", so those default to "SentenceTransformer"
            return config.get("model_type", "SentenceTransformer")

    def _load_module_class_from_ref(
        self,
        class_ref: str,
        model_name_or_path: str,
        trust_remote_code: bool,
        revision: str | None,
        model_kwargs: dict[str, Any] | None,
    ) -> nn.Module:
        """
        Load a module class from a class reference string.

        Args:
            class_ref: The class reference string (e.g., "sentence_transformers.sentence_transformer.modules.Pooling")
            model_name_or_path: The model name or path
            trust_remote_code: Whether to trust remote code
            revision: The model revision
            model_kwargs: Additional model kwargs

        Returns:
            The module class
        """
        # If the class is from sentence_transformers, we can directly import it,
        # otherwise, we try to import it dynamically, and if that fails, we fall back to the default import
        if class_ref.startswith("sentence_transformers."):
            return import_from_string(class_ref)

        if trust_remote_code or os.path.exists(model_name_or_path):
            code_revision = model_kwargs.pop("code_revision", None) if model_kwargs else None
            try:
                return get_class_from_dynamic_module(
                    class_ref,
                    model_name_or_path,
                    revision=revision,
                    code_revision=code_revision,
                )
            except (OSError, ValueError):
                # Ignore the error if 1) the file does not exist, or 2) the class_ref is not correctly formatted/found
                pass

        return import_from_string(class_ref)

    def evaluate(self, evaluator: SentenceEvaluator, output_path: str | None = None) -> dict[str, float] | float:
        """
        Evaluate the model based on an evaluator

        Args:
            evaluator (SentenceEvaluator): The evaluator used to evaluate the model.
            output_path (str, optional): The path where the evaluator can write the results. Defaults to None.

        Returns:
            The evaluation results.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable gradient checkpointing for the model."""
        # Propagate the gradient checkpointing to the transformer model
        for module in self.modules():
            if module is not self and hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @property
    def device(self) -> torch.device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        In case there are no PyTorch parameters, fall back to CPU.
        """
        if (transformers_model := self.transformers_model) is not None and hasattr(transformers_model, "device"):
            return transformers_model.device

        if len(self._modules) and hasattr(self[0], "auto_model") and hasattr(self[0].auto_model, "device"):
            return self[0].auto_model.device

        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> list[tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            try:
                first_tuple = next(gen)
                return first_tuple[1].device
            except StopIteration:
                return torch.device("cpu")

    def start_multi_process_pool(
        self, target_devices: list[str] | None = None
    ) -> dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to infer with several independent processes.

        This method is recommended if you want to predict on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with predict and
        stop_multi_process_pool.

        Args:
            target_devices (List[str], optional): PyTorch target devices, e.g. ["cuda:0", "cuda:1", ...],
                ["npu:0", "npu:1", ...], or ["cpu", "cpu", "cpu", "cpu"]. If target_devices is None and CUDA/NPU
                is available, then all available CUDA/NPU devices will be used. If target_devices is None and
                CUDA/NPU is not available, then 4 CPU devices will be used.

        Returns:
            Dict[str, Any]: A dictionary with the target processes, an input queue, and an output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                target_devices = [f"npu:{i}" for i in range(torch.npu.device_count())]
            else:
                logger.info("CUDA/NPU is not available. Starting 4 CPU workers")
                target_devices = ["cpu"] * 4

        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        self.to("cpu")
        self.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device_id in target_devices:
            p = ctx.Process(
                target=self.__class__._multi_process_worker,
                args=(device_id, self, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    @staticmethod
    def stop_multi_process_pool(pool: dict[Literal["input", "output", "processes"], Any]) -> None:
        """
        Stops all processes started with start_multi_process_pool.

        Args:
            pool (Dict[str, object]): A dictionary containing the input queue, output queue, and process list.

        Returns:
            None
        """
        for p in pool["processes"]:
            p.terminate()

        for p in pool["processes"]:
            p.join()
            p.close()

        pool["input"].close()
        pool["output"].close()

    def _multi_process(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def _multi_process_worker(
        target_device: str,
        model: BaseModel,
        input_queue: Queue,
        results_queue: Queue,
    ) -> None:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @property
    def tokenizer(self) -> Any:
        """
        Property to get the tokenizer that is used by this model
        """
        return self[0].tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        """
        Property to set the tokenizer that should be used by this model
        """
        self[0].tokenizer = value

    @property
    def processor(self) -> Any:
        """
        Property to get the processor that is used by this model
        """
        return self[0].processor

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximal input sequence length for the model. Longer inputs will be truncated.

        Returns:
            int: The maximal input sequence length.
        """
        return self._first_module().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value) -> None:
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        self._first_module().max_seq_length = value

    @property
    def transformers_model(self) -> PreTrainedModel | None:
        """
        Property to get the underlying transformers PreTrainedModel instance, if it exists.
        Note that it's possible for a model to have multiple underlying transformers models, but this property
        will return the first one it finds in the module hierarchy.

        .. note::

            This property can also return e.g. ORTModelForFeatureExtraction or OVModelForFeatureExtraction instances
            from the optimum-intel and optimum-onnx libraries, if the model is loaded using ``backend="onnx"`` or
            ``backend="openvino"``.

        Returns:
            PreTrainedModel or None: The underlying transformers model or None if not found.
        """
        for module in self.modules():
            # The Transformer check allows for returning underlying models with backend="onnx" or "openvino"
            if isinstance(module, Transformer):
                return module.model
            if isinstance(module, PreTrainedModel):
                return module
        return None

    @property
    def _target_device(self) -> torch.device:
        logger.warning(
            f"`{self.__class__.__name__}._target_device` has been deprecated. Please use `{self.__class__.__name__}.device` instead.",
        )
        return self.device

    @_target_device.setter
    def _target_device(self, device: int | str | torch.device | None = None) -> None:
        logger.warning(
            f"`{self.__class__.__name__}._target_device` has been deprecated. Please use `to(device)` instead.",
        )
        self.to(device)

    @property
    def dtype(self) -> torch.dtype | None:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return next(self.parameters()).dtype if len(list(self.parameters())) > 0 else None

    @property
    def _no_split_modules(self) -> list[str]:
        """
        Return the list of modules that should not be split when using model parallelism.
        """
        return []

    @property
    def _keys_to_ignore_on_save(self) -> list[str]:
        """
        Return the list of keys to ignore when saving the model.
        """
        return []
