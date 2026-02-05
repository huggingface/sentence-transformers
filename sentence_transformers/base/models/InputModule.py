from __future__ import annotations

from typing import Any

import torch
from tokenizers import Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from sentence_transformers.base.models.modality_utils import (
    ArrayInputs,
    DictInputs,
    ImageInputs,
    PairStrInputs,
    StrInputs,
)
from sentence_transformers.base.models.Module import Module


class InputModule(Module):
    """
    Subclass of :class:`sentence_transformers.sentence_transformer.models.Module`, base class for all input modules in the Sentence
    Transformers library, i.e. modules that are used to process inputs and optionally also perform processing
    in the forward pass.

    This class provides a common interface for all input modules, including methods for loading and saving the module's
    configuration and weights, as well as input processing. It also provides a method for performing the forward pass
    of the module.

    Three abstract methods are defined in this class, which must be implemented by subclasses:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.forward`: The forward pass of the module.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save`: Save the module to disk.
    - :meth:`sentence_transformers.sentence_transformer.models.InputModule.tokenize`: Tokenize the input texts and return a dictionary of tokenized features.

    Optionally, you may also have to override:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.load`: Load the module from disk.

    To assist with loading and saving the module, several utility methods are provided:

    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_config`: Load the module's configuration from a JSON file.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_file_path`: Load a file from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_dir_path`: Load a directory from the module's directory, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.load_torch_weights`: Load the PyTorch weights of the module, regardless of whether the module is saved locally or on Hugging Face.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save_config`: Save the module's configuration to a JSON file.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.save_torch_weights`: Save the PyTorch weights of the module.
    - :meth:`sentence_transformers.sentence_transformer.models.InputModule.save_tokenizer`: Save the tokenizer used by the module.
    - :meth:`sentence_transformers.sentence_transformer.models.Module.get_config_dict`: Get the module's configuration as a dictionary.

    And several class variables are defined to assist with loading and saving the module:

    - :attr:`sentence_transformers.sentence_transformer.models.Module.config_file_name`: The name of the configuration file used to save the module's configuration.
    - :attr:`sentence_transformers.sentence_transformer.models.Module.config_keys`: A list of keys used to save the module's configuration.
    - :attr:`sentence_transformers.sentence_transformer.models.InputModule.save_in_root`: Whether to save the module's configuration in the root directory of the model or in a subdirectory named after the module.
    - :attr:`sentence_transformers.sentence_transformer.models.InputModule.tokenizer`: The tokenizer used by the module.
    """

    save_in_root: bool = True
    tokenizer: PreTrainedTokenizerBase | Tokenizer
    """
    The tokenizer used for tokenizing the input texts. It can be either a
    :class:`transformers.PreTrainedTokenizerBase` subclass or a Tokenizer from the
    ``tokenizers`` library.
    """

    def preprocess(
        self,
        inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | PairStrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """
        Preprocesses the input texts and returns a dictionary of preprocessed features.

        Args:
            texts (list[str]): List of input texts to preprocess.
            **kwargs: Additional keyword arguments for preprocessing, e.g. ``task``.

        Returns:
            dict[str, torch.Tensor | Any]: Dictionary containing preprocessed features, e.g.
                ``{"input_ids": ..., "attention_mask": ...}``
        """
        # Should be overridden by subclasses, and otherwise just default to 'tokenize'
        return self.tokenize(inputs, **kwargs)

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor | Any]:
        """
        .. deprecated::
            `tokenize` is deprecated and will be removed in a future version. Use `preprocess` instead.

        Tokenizes the input texts and returns a dictionary of tokenized features.

        Args:
            texts (list[str]): List of input texts to tokenize.
            **kwargs: Additional keyword arguments for tokenization, e.g. ``task``.

        Returns:
            dict[str, torch.Tensor | Any]: Dictionary containing tokenized features, e.g.
                ``{"input_ids": ..., "attention_mask": ...}``
        """
        import warnings

        warnings.warn(
            "The `tokenize` method is deprecated and will be removed in a future version. "
            "Please use `preprocess` instead.",
            FutureWarning,
            stacklevel=2,
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
