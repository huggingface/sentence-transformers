from __future__ import annotations

from typing import Any

import torch
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import logging as transformers_logging

from sentence_transformers.base.modality_types import Modality, PairInput, SingleInput
from sentence_transformers.base.modules.module import Module

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


class InputModule(Module):
    """
    Subclass of :class:`sentence_transformers.base.modules.Module`, base class for all input modules in the Sentence
    Transformers library, i.e. modules that are used to process inputs and optionally also perform processing
    in the forward pass.

    This class provides a common interface for all input modules, including methods for loading and saving the module's
    configuration and weights, as well as input processing. It also provides a method for performing the forward pass
    of the module.

    Two abstract methods are inherited from :class:`~sentence_transformers.base.modules.Module` and must be implemented
    by subclasses:

    - :meth:`sentence_transformers.base.modules.Module.forward`: The forward pass of the module.
    - :meth:`sentence_transformers.base.modules.Module.save`: Save the module to disk.

    Additionally, subclasses should override:

    - :meth:`sentence_transformers.base.modules.InputModule.preprocess`: Preprocess the inputs and return a dictionary of preprocessed features.

    Optionally, you may also have to override:

    - :attr:`sentence_transformers.base.modules.InputModule.modalities`: The list of supported input modalities. Defaults to ``["text"]``. Override this to advertise support for non-text modalities (e.g. ``["text", "image"]``).
    - :meth:`sentence_transformers.base.modules.Module.load`: Load the module from disk.

    To assist with loading and saving the module, several utility methods are provided:

    - :meth:`sentence_transformers.base.modules.Module.load_config`: Load the module's configuration from a JSON file.
    - :meth:`sentence_transformers.base.modules.Module.load_file_path`: Load a file from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.base.modules.Module.load_dir_path`: Load a directory from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.base.modules.Module.load_torch_weights`: Load the PyTorch weights of the module, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.base.modules.Module.save_config`: Save the module's configuration to a JSON file.
    - :meth:`sentence_transformers.base.modules.Module.save_torch_weights`: Save the PyTorch weights of the module.
    - :meth:`sentence_transformers.base.modules.InputModule.save_tokenizer`: Save the tokenizer used by the module.
    - :meth:`sentence_transformers.base.modules.Module.get_config_dict`: Get the module's configuration as a dictionary.

    And several class variables are defined to assist with loading and saving the module:

    - :attr:`sentence_transformers.base.modules.Module.config_file_name`: The name of the configuration file used to save the module's configuration.
    - :attr:`sentence_transformers.base.modules.Module.config_keys`: A list of keys used to save the module's configuration.
    - :attr:`sentence_transformers.base.modules.InputModule.save_in_root`: Whether to save the module's configuration in the root directory of the model or in a subdirectory named after the module.
    - :attr:`sentence_transformers.base.modules.InputModule.tokenizer`: The tokenizer used by the module.
    """

    save_in_root: bool = True
    tokenizer: PreTrainedTokenizerBase | Tokenizer
    """
    The tokenizer used for tokenizing the input texts. It can be either a
    :class:`transformers.PreTrainedTokenizerBase` subclass or a Tokenizer from the
    ``tokenizers`` library.
    """

    @property
    def modalities(self) -> list[Modality]:
        """The list of supported input modalities. Defaults to ``["text"]``."""
        return ["text"]

    @staticmethod
    def _prepend_prompt(inputs: list[str], prompt: str) -> list[str]:
        """Prepend a prompt string to each text input."""
        return [prompt + text for text in inputs]

    def preprocess(
        self,
        inputs: list[SingleInput | PairInput],
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """
        Preprocesses the input texts and returns a dictionary of preprocessed features.

        Args:
            inputs (list[SingleInput | PairInput]): List of inputs to preprocess.
            prompt (str | None): Optional prompt to prepend to text inputs.
            **kwargs: Additional keyword arguments for preprocessing, e.g. ``task``.

        Returns:
            dict[str, torch.Tensor | Any]: Dictionary containing preprocessed features, e.g.
                ``{"input_ids": ..., "attention_mask": ...}``, depending on what keys the module's forward method expects.
        """
        # Backward compatibility: if a subclass overrides tokenize() but not preprocess(),
        # delegate to the overridden tokenize(). We check the MRO to avoid calling the base
        # tokenize() which delegates back to preprocess(), causing infinite mutual recursion.
        if type(self).tokenize is not InputModule.tokenize:
            logger.warning_once(
                f"{type(self).__name__} overrides `tokenize` instead of `preprocess`. "
                "`tokenize` is deprecated, please override `preprocess` instead.",
            )
            if prompt:
                inputs = self._prepend_prompt(inputs, prompt)
            return self.tokenize(inputs, **kwargs)
        raise NotImplementedError(f"{type(self).__name__} must implement the `preprocess` method.")

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor | Any]:
        """
        .. deprecated::
            `tokenize` is deprecated. Use `preprocess` instead.

        Tokenizes the input texts and returns a dictionary of tokenized features.

        Args:
            texts (list[str]): List of input texts to tokenize.
            **kwargs: Additional keyword arguments for tokenization, e.g. ``task``.

        Returns:
            dict[str, torch.Tensor | Any]: Dictionary containing tokenized features, e.g.
                ``{"input_ids": ..., "attention_mask": ...}``
        """
        logger.warning_once(
            "The `tokenize` method is deprecated, please use `preprocess` instead.",
        )
        return self.preprocess(texts, **kwargs)

    def save_tokenizer(self, output_path: str, **kwargs) -> None:
        """
        Saves the tokenizer to the specified output path.

        Args:
            output_path (str): Path to save the tokenizer.
            **kwargs: Additional keyword arguments for saving the tokenizer.

        Returns:
            None
        """
        if not hasattr(self, "tokenizer"):
            return

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self.tokenizer.save_pretrained(output_path, **kwargs)
        elif isinstance(self.tokenizer, Tokenizer):
            self.tokenizer.save(output_path, **kwargs)
        return
