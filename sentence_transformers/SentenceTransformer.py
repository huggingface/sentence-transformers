from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import math
import os
import queue
import shutil
import sys
import tempfile
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch
import torch.multiprocessing as mp
import transformers
from huggingface_hub import HfApi
from packaging import version
from torch import Tensor, device, nn
from tqdm.autonotebook import trange
from transformers import PreTrainedModel, is_torch_npu_available
from transformers.dynamic_module_utils import get_class_from_dynamic_module, get_relative_import_files
from typing_extensions import deprecated

from sentence_transformers.model_card import SentenceTransformerModelCardData, generate_model_card
from sentence_transformers.models import Router
from sentence_transformers.models.Module import Module
from sentence_transformers.similarity_functions import SimilarityFunction

from . import __MODEL_HUB_ORGANIZATION__, __version__
from .evaluation import SentenceEvaluator
from .fit_mixin import FitMixin
from .models import Pooling, Transformer
from .peft_mixin import PeftAdapterMixin
from .quantization import quantize_embeddings
from .util import (
    batch_to_device,
    get_device_name,
    import_from_string,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
    truncate_embeddings,
)

logger = logging.getLogger(__name__)


class SentenceTransformer(nn.Sequential, FitMixin, PeftAdapterMixin):
    """
    Loads or creates a SentenceTransformer model that can be used to map sentences / text to embeddings.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disk, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SentenceTransformer models from scratch.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        prompts (Dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text.
            The prompt text will be prepended before any text to encode. For example:
            `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
            titles in "}`.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use. Valid options are "cosine", "dot",
            "euclidean", and "manhattan". If not set, it is automatically set to "cosine" if `similarity` or
            `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.
        cache_folder (str, optional): Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
        use_auth_token (bool or str, optional): Deprecated argument. Please use `token` instead.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. Defaults to None.
        model_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers model.
            Particularly useful options are:

            - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`.
              The different options are:

                    1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified
                    ``dtype``, ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will
                    get loaded in ``torch.float`` (fp32).

                    2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be
                    attempted to be used. If this entry isn't found then next check the ``dtype`` of the first weight in
                    the checkpoint that's of a floating point type and use that as ``dtype``. This will load the model
                    using the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how
                    the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
            - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
              `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
              <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
              or `"flash_attention_2"` (using `Dao-AILab/flash-attention <https://github.com/Dao-AILab/flash-attention>`_).
              By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
              implementation.
            - ``provider``: If backend is "onnx", this is the provider to use for inference, for example "CPUExecutionProvider",
              "CUDAExecutionProvider", etc. See https://onnxruntime.ai/docs/execution-providers/ for all ONNX execution providers.
            - ``file_name``: If backend is "onnx" or "openvino", this is the file name to load, useful for loading optimized
              or quantized ONNX or OpenVINO models.
            - ``export``: If backend is "onnx" or "openvino", then this is a boolean flag specifying whether this model should
              be exported to the backend. If not specified, the model will be exported only if the model repository or directory
              does not already contain an exported model.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details.
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
            on the different backends.

    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[1.0000, 0.6817, 0.0492],
            #         [0.6817, 1.0000, 0.0421],
            #         [0.0492, 0.0421, 1.0000]])
    """

    model_card_data_class = SentenceTransformerModelCardData

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        # Note: self._load_sbert_model can also update `self.prompts` and `self.default_prompt_name`
        self.prompts = {"query": "", "document": ""}
        if prompts:
            self.prompts.update(prompts)
        self.default_prompt_name = default_prompt_name
        self.similarity_fn_name = similarity_fn_name
        self.trust_remote_code = trust_remote_code
        self.truncate_dim = truncate_dim
        self.model_card_data = model_card_data or self.model_card_data_class(local_files_only=local_files_only)
        self.module_kwargs = None
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {"model_type": self.__class__.__name__}
        self._prompt_length_mapping = {}
        self.backend = backend
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v4 of SentenceTransformers.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if cache_folder is None:
            cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME")

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device_name: {device}")

        if device == "hpu" and importlib.util.find_spec("optimum") is not None:
            from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

            adapt_transformers_to_gaudi()

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info(f"Load pretrained {self.__class__.__name__}: {model_name_or_path}")

            # Old models that don't belong to any organization
            basic_transformer_models = [
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

            if not os.path.exists(model_name_or_path):
                # Not a path, load from hub
                if "\\" in model_name_or_path or model_name_or_path.count("/") > 1:
                    raise FileNotFoundError(f"Path {model_name_or_path} not found")

                if "/" not in model_name_or_path and model_name_or_path.lower() not in basic_transformer_models:
                    # A model from sentence-transformers
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + "/" + model_name_or_path
            has_modules = is_sentence_transformer_model(
                model_name_or_path,
                token,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
            )
            if (
                has_modules
                and self._get_model_type(
                    model_name_or_path,
                    token,
                    cache_folder=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                == self._model_config["model_type"]
            ):
                modules, self.module_kwargs = self._load_sbert_model(
                    model_name_or_path,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    config_kwargs=config_kwargs,
                )
            else:
                modules = self._load_auto_model(
                    model_name_or_path,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs,
                    config_kwargs=config_kwargs,
                    has_modules=has_modules,
                )

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        # Ensure all tensors in the model are of the same dtype as the first tensor
        # This is necessary if the first module has been given a lower precision via
        # model_kwargs["torch_dtype"]. The rest of the model should be loaded in the same dtype
        # See #2887 for more details
        try:
            dtype = next(self.parameters()).dtype
            self.to(dtype)
        except StopIteration:
            pass

        self.to(device)
        self.is_hpu_graph_enabled = False

        if self.default_prompt_name is not None and self.default_prompt_name not in self.prompts:
            raise ValueError(
                f"Default prompt name '{self.default_prompt_name}' not found in the configured prompts "
                f"dictionary with keys {list(self.prompts.keys())!r}."
            )

        if self.prompts and (non_empty_keys := [k for k, v in self.prompts.items() if v != ""]):
            if len(non_empty_keys) == 1:
                logger.info(f"1 prompt is loaded, with the key: {non_empty_keys[0]}")
            else:
                logger.info(f"{len(non_empty_keys)} prompts are loaded, with the keys: {non_empty_keys}")
        if self.default_prompt_name:
            logger.warning(
                f"Default prompt name is set to '{self.default_prompt_name}'. "
                "This prompt will be applied to all `encode()` calls, except if `encode()` "
                "is called with `prompt` or `prompt_name` parameters."
            )

        # Ideally, INSTRUCTOR models should set `include_prompt=False` in their pooling configuration, but
        # that would be a breaking change for users currently using the InstructorEmbedding project.
        # So, instead we hardcode setting it for the main INSTRUCTOR models, and otherwise give a warning if we
        # suspect the user is using an INSTRUCTOR model.
        if model_name_or_path in ("hkunlp/instructor-base", "hkunlp/instructor-large", "hkunlp/instructor-xl"):
            self.set_pooling_include_prompt(include_prompt=False)
        elif (
            model_name_or_path
            and "/" in model_name_or_path
            and "instructor" in model_name_or_path.split("/")[1].lower()
        ):
            if any([module.include_prompt for module in self if isinstance(module, Pooling)]):
                logger.warning(
                    "Instructor models require `include_prompt=False` in the pooling configuration. "
                    "Either update the model configuration or call `model.set_pooling_include_prompt(False)` after loading the model."
                )

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)

    def get_backend(self) -> Literal["torch", "onnx", "openvino"]:
        """Return the backend used for inference, which can be one of "torch", "onnx", or "openvino".

        Returns:
            str: The backend used for inference.
        """
        return self.backend

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

    def encode_query(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sentence embeddings specifically optimized for query representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" prompt,
           if available in the model's ``prompts`` dictionary.
        2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.models.Router`
           module, it will use the "query" task type to route the input through the appropriate submodules.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        Args:
            sentences (Union[str, List[str]]): The sentences to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,
                i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.
                Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference
                is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``
                from the model initialization is used. Defaults to None.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                # Load a pre-trained SentenceTransformer model
                model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

                # Encode some queries
                queries = [
                    "What are the effects of climate change?",
                    "History of artificial intelligence",
                    "Technical specifications product XYZ",
                ]

                # Using query-specific encoding
                embeddings = model.encode_query(queries)
                print(embeddings.shape)
                # (3, 768)
        """
        if prompt_name is None and "query" in self.prompts and prompt is None:
            prompt_name = "query"

        return self.encode(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            task="query",
            **kwargs,
        )

    def encode_document(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sentence embeddings specifically optimized for document/passage representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "document" prompt,
           if available in the model's ``prompts`` dictionary.
        2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.models.Router`
           module, it will use the "document" task type to route the input through the appropriate submodules.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        Args:
            sentences (Union[str, List[str]]): The sentences to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,
                i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.
                Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference
                is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``
                from the model initialization is used. Defaults to None.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                # Load a pre-trained SentenceTransformer model
                model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

                # Encode some documents
                documents = [
                    "This research paper discusses the effects of climate change on marine life.",
                    "The article explores the history of artificial intelligence development.",
                    "This document contains technical specifications for the new product line.",
                ]

                # Using document-specific encoding
                embeddings = model.encode_document(documents)
                print(embeddings.shape)
                # (3, 768)
        """
        if prompt_name is None and prompt is None:
            for candidate_prompt_name in ["document", "passage", "corpus"]:
                if candidate_prompt_name in self.prompts:
                    prompt_name = candidate_prompt_name
                    break

        return self.encode(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            task="document",
            **kwargs,
        )

    # Return a single tensor because we're passing a single sentence.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...

    # Return a single array, because convert_to_numpy is True
    # and "sentence_embeddings" is passed
    @overload
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: Literal[True] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> np.ndarray: ...

    # Return a single tensor, because convert_to_tensor is True
    # and "sentence_embeddings" is passed
    @overload
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: Literal[True] = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...

    # Return a list of tensors. Value of convert_ doesn't matter.
    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["sentence_embedding", "token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[Tensor]: ...

    # Return a list of dict of features, ignore the conversion args.
    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> list[dict[str, Tensor]]: ...

    # Return a dict of features, ignore the conversion args.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: None = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> dict[str, Tensor]: ...

    # If "token_embeddings" is True, then the output is a single tensor.
    @overload
    def encode(
        self,
        sentences: str,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        output_value: Literal["token_embeddings"] = ...,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: bool = ...,
        device: str | list[str | torch.device] | None = ...,
        normalize_embeddings: bool = ...,
        truncate_dim: int | None = ...,
        pool: dict[Literal["input", "output", "processes"], Any] | None = ...,
        chunk_size: int | None = ...,
        **kwargs,
    ) -> Tensor: ...

    @torch.inference_mode()
    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sentence embeddings.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        Args:
            sentences (Union[str, List[str]]): The sentences to embed.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return:
                "sentence_embedding" to get sentence embeddings, "token_embeddings" to get wordpiece token embeddings, and `None`,
                to get all output values. Defaults to "sentence_embedding".
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,
                i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.
                Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference
                is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``
                from the model initialization is used. Defaults to None.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                # Load a pre-trained SentenceTransformer model
                model = SentenceTransformer("all-mpnet-base-v2")

                # Encode some texts
                sentences = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 768)
        """
        if self.device.type == "hpu" and not self.is_hpu_graph_enabled:
            import habana_frameworks.torch as ht

            if hasattr(ht, "hpu") and hasattr(ht.hpu, "wrap_in_hpu_graph"):
                ht.hpu.wrap_in_hpu_graph(self, disable_tensor_cache=True)
                self.is_hpu_graph_enabled = True

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        # Cast an individual input to a list with length 1
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        # Throw an error if unused kwargs are passed, except 'task' which is always allowed, even
        # when it does not do anything (as e.g. there's no Router module in the model)
        model_kwargs = self.get_model_kwargs()
        if unused_kwargs := set(kwargs) - set(model_kwargs) - {"task"}:
            raise ValueError(
                f"{self.__class__.__name__}.encode() has been called with additional keyword arguments that this model does not use: {list(unused_kwargs)}. "
                + (
                    f"As per {self.__class__.__name__}.get_model_kwargs(), the valid additional keyword arguments are: {model_kwargs}."
                    if model_kwargs
                    else f"As per {self.__class__.__name__}.get_model_kwargs(), this model does not accept any additional keyword arguments."
                )
            )

        # If pool or a list of devices is provided, use multi-process encoding
        if pool is not None or (isinstance(device, list) and len(device) > 0):
            return self._encode_multi_process(
                sentences,
                # Utility and post-processing parameters
                show_progress_bar=show_progress_bar,
                input_was_string=input_was_string,
                # Multi-process encoding parameters
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                # Encoding parameters
                prompt_name=prompt_name,
                prompt=prompt,
                batch_size=batch_size,
                output_value=output_value,
                precision=precision,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                normalize_embeddings=normalize_embeddings,
                truncate_dim=truncate_dim,
                **kwargs,
            )

        # Original encoding logic when not using multi-process
        allowed_precisions = {"float32", "int8", "uint8", "binary", "ubinary"}
        if precision and precision not in allowed_precisions:
            raise ValueError(f"Precision {precision!r} is not supported")

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
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None and len(prompt) > 0:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            length = self._get_prompt_length(prompt, **kwargs)
            if length is not None:
                extra_features["prompt_length"] = length

        # Here, device is either a single device string (e.g., "cuda:0", "cpu") for single-process encoding or None
        if device is None:
            device = self.device

        self.to(device)

        truncate_dim = truncate_dim if truncate_dim is not None else self.truncate_dim

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, **kwargs)
            if self.device.type == "hpu":
                if "input_ids" in features:
                    curr_tokenize_len = features["input_ids"].shape
                    additional_pad_len = 2 ** math.ceil(math.log2(curr_tokenize_len[1])) - curr_tokenize_len[1]
                    features["input_ids"] = torch.cat(
                        (
                            features["input_ids"],
                            torch.ones((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    features["attention_mask"] = torch.cat(
                        (
                            features["attention_mask"],
                            torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                        ),
                        -1,
                    )
                    if "token_type_ids" in features:
                        features["token_type_ids"] = torch.cat(
                            (
                                features["token_type_ids"],
                                torch.zeros((curr_tokenize_len[0], additional_pad_len), dtype=torch.int8),
                            ),
                            -1,
                        )

            features = batch_to_device(features, device)
            features.update(extra_features)

            with torch.no_grad():
                out_features = self.forward(features, **kwargs)
                if self.device.type == "hpu":
                    out_features = copy.deepcopy(out_features)

                if truncate_dim:
                    out_features["sentence_embedding"] = truncate_embeddings(
                        out_features["sentence_embedding"], truncate_dim
                    )

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for idx in range(len(out_features["sentence_embedding"])):
                        batch_item = {}
                        for name, value in out_features.items():
                            try:
                                batch_item[name] = value[idx]
                            except TypeError:
                                # Handle non-indexable values (like prompt_length)
                                batch_item[name] = value
                        embeddings.append(batch_item)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if all_embeddings and precision and precision != "float32":
            all_embeddings = quantize_embeddings(all_embeddings, precision=precision)

        if convert_to_tensor:
            if len(all_embeddings):
                if isinstance(all_embeddings, np.ndarray):
                    all_embeddings = torch.from_numpy(all_embeddings)
                else:
                    all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.tensor([], device=self.device)
        elif convert_to_numpy:
            if not isinstance(all_embeddings, np.ndarray):
                if all_embeddings and all_embeddings[0].dtype == torch.bfloat16:
                    all_embeddings = np.asarray([emb.float().numpy() for emb in all_embeddings])
                else:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        elif isinstance(all_embeddings, np.ndarray):
            all_embeddings = [torch.from_numpy(embedding) for embedding in all_embeddings]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def forward(self, input: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
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

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function used by :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise`.

        Returns:
            Optional[str]: The name of the similarity function. Can be None if not set, in which case it will
                default to "cosine" when first called.

        Example:
            >>> model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            >>> model.similarity_fn_name
            'dot'
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self, value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    @overload
    def similarity(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor: ...

    @overload
    def similarity(self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]) -> Tensor: ...

    @property
    def similarity(self) -> Callable[[Tensor | npt.NDArray[np.float32], Tensor | npt.NDArray[np.float32]], Tensor]:
        """
        Compute the similarity between two collections of embeddings. The output will be a matrix with the similarity
        scores between all embeddings from the first parameter and all embeddings from the second parameter. This
        differs from `similarity_pairwise` which computes the similarity between each pair of embeddings.
        This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings_1, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings_2, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings_1, num_embeddings_2]-shaped torch tensor with similarity scores.

        Example:
            ::

                >>> model = SentenceTransformer("all-mpnet-base-v2")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences, normalize_embeddings=True)
                >>> model.similarity(embeddings, embeddings)
                tensor([[1.0000, 0.7235, 0.0290, 0.1309],
                        [0.7235, 1.0000, 0.0613, 0.1129],
                        [0.0290, 0.0613, 1.0000, 0.5027],
                        [0.1309, 0.1129, 0.5027, 1.0000]])
                >>> model.similarity_fn_name
                "cosine"
                >>> model.similarity_fn_name = "euclidean"
                >>> model.similarity(embeddings, embeddings)
                tensor([[-0.0000, -0.7437, -1.3935, -1.3184],
                        [-0.7437, -0.0000, -1.3702, -1.3320],
                        [-1.3935, -1.3702, -0.0000, -0.9973],
                        [-1.3184, -1.3320, -0.9973, -0.0000]])
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity

    @overload
    def similarity_pairwise(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor: ...

    @overload
    def similarity_pairwise(
        self, embeddings1: npt.NDArray[np.float32], embeddings2: npt.NDArray[np.float32]
    ) -> Tensor: ...

    @property
    def similarity_pairwise(
        self,
    ) -> Callable[[Tensor | npt.NDArray[np.float32], Tensor | npt.NDArray[np.float32]], Tensor]:
        """
        Compute the similarity between two collections of embeddings. The output will be a vector with the similarity
        scores between each pair of embeddings.
        This method supports only embeddings with fp32 precision and does not accommodate quantized embeddings.

        Args:
            embeddings1 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.
            embeddings2 (Union[Tensor, ndarray]): [num_embeddings, embedding_dim] or [embedding_dim]-shaped numpy array or torch tensor.

        Returns:
            Tensor: A [num_embeddings]-shaped torch tensor with pairwise similarity scores.

        Example:
            ::

                >>> model = SentenceTransformer("all-mpnet-base-v2")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences, normalize_embeddings=True)
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([0.7235, 0.5027])
                >>> model.similarity_fn_name
                "cosine"
                >>> model.similarity_fn_name = "euclidean"
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([-0.7437, -0.9973])
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_pairwise

    def start_multi_process_pool(
        self, target_devices: list[str] | None = None
    ) -> dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the encoding with several independent processes
        via :meth:`SentenceTransformer.encode_multi_process <sentence_transformers.SentenceTransformer.encode_multi_process>`.

        This method is recommended if you want to encode on multiple GPUs or CPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        and stop_multi_process_pool.

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
                target=self.__class__._encode_multi_process_worker,
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

    @deprecated(
        "The `encode_multi_process` method has been deprecated, and its functionality has been integrated into `encode`. "
        "You can now call `encode` with the same parameters to achieve multi-process encoding.",
    )
    def encode_multi_process(
        self,
        sentences: list[str],
        pool: dict[Literal["input", "output", "processes"], Any],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        chunk_size: int | None = None,
        show_progress_bar: bool | None = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        """
        .. warning::
            This method is deprecated. You can now call :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`
            with the same parameters instead, which will automatically handle multi-process encoding using the provided ``pool``.

        Encodes a list of sentences using multiple processes and GPUs via
        :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.
        The sentences are chunked into smaller packages and sent to individual processes, which encode them on different
        GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

        Args:
            sentences (List[str]): List of sentences to encode.
            pool (Dict[Literal["input", "output", "processes"], Any]): A pool of workers started with
                :meth:`SentenceTransformer.start_multi_process_pool <sentence_transformers.SentenceTransformer.start_multi_process_pool>`.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding. Must be a key in the `prompts` dictionary,
                which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (Optional[str], optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored. Defaults to None.
            batch_size (int): Encode sentences with batch size. (default: 32)
            chunk_size (int): Sentences are chunked and sent to the individual processes. If None, it determines a
                sensible size. Defaults to None.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"]): The precision to use for the
                embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions
                are quantized embeddings. Quantized embeddings are smaller in size and faster to compute, but may
                have lower accuracy. They are useful for reducing the size of the embeddings of a corpus for
                semantic search, among other tasks. Defaults to "float32".
            normalize_embeddings (bool): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,
                i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.
                Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference
                is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``
                from the model initialization is used. Defaults to None.

        Returns:
            np.ndarray: A 2D numpy array with shape [num_inputs, output_dimension].

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                def main():
                    model = SentenceTransformer("all-mpnet-base-v2")
                    sentences = ["The weather is so nice!", "It's so sunny outside.", "He's driving to the movie theater.", "She's going to the cinema."] * 1000

                    pool = model.start_multi_process_pool()
                    embeddings = model.encode_multi_process(sentences, pool)
                    model.stop_multi_process_pool(pool)

                    print(embeddings.shape)
                    # => (4000, 768)

                if __name__ == "__main__":
                    main()
        """
        return self.encode(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value="sentence_embedding",
            precision=precision,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
        )

    def _encode_multi_process(
        self,
        inputs: list[str],
        show_progress_bar: bool | None = True,
        input_was_string: bool = False,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **encode_kwargs,
    ):
        convert_to_tensor = encode_kwargs.get("convert_to_tensor", False)
        convert_to_numpy = encode_kwargs.get("convert_to_numpy", False)
        encode_kwargs["show_progress_bar"] = False

        # Create a pool if is not provided, but a list of devices is
        created_pool = False
        if pool is None and isinstance(device, list):
            pool = self.start_multi_process_pool(device)
            created_pool = True

        try:
            # Determine chunk size if not provided. As a default, aim for 10 chunks per process, with a maximum of 5000 sentences per chunk.
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)  # Ensure chunk_size is at least 1

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            # Send inputs to the input queue in chunks
            chunk_id = -1  # We default to -1 to handle empty input gracefully
            for chunk_id, chunk_start in enumerate(range(0, len(inputs), chunk_size)):
                chunk = inputs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, encode_kwargs])

            # Collect results from the output queue
            output_list = sorted(
                [output_queue.get() for _ in trange(chunk_id + 1, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],
            )
            if input_was_string:
                # If input was a single string, return the first (only) result directly
                return output_list[0][1][0]

            # Handle the various output formats: torch tensors, numpy arrays, or
            # list of dictionaries, also when empty.
            embeddings = [output[1] for output in output_list]
            if embeddings:
                if isinstance(embeddings[0], list):
                    embeddings = sum(embeddings, [])
                elif isinstance(embeddings[0], torch.Tensor):
                    embeddings = torch.cat(embeddings)
                elif isinstance(embeddings[0], np.ndarray):
                    embeddings = np.concatenate(embeddings, axis=0)
            elif convert_to_tensor:
                embeddings = torch.Tensor()
            elif convert_to_numpy:
                embeddings = np.array([])
            return embeddings

        finally:
            # Clean up the pool if we created it
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _encode_multi_process_worker(
        target_device: str, model: SentenceTransformer, input_queue: Queue, results_queue: Queue
    ) -> None:
        """
        Internal working process to encode sentences in multi-process setup
        """
        while True:
            try:
                chunk_id, inputs, kwargs = input_queue.get()
                embeddings = model.encode(inputs, device=target_device, **kwargs)
                # If multi-process embeddings are not on CPUs, move them to CPU, so they can
                # all be concatenated later
                if isinstance(embeddings, torch.Tensor) and embeddings.device.type != "cpu":
                    embeddings = embeddings.cpu()
                elif isinstance(embeddings, dict):
                    embeddings = {
                        key: value.cpu() if isinstance(value, torch.Tensor) and value.device.type != "cpu" else value
                        for key, value in embeddings.items()
                    }
                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break

    def set_pooling_include_prompt(self, include_prompt: bool) -> None:
        """
        Sets the `include_prompt` attribute in the pooling layer in the model, if there is one.

        This is useful for INSTRUCTOR models, as the prompt should be excluded from the pooling strategy
        for these models.

        Args:
            include_prompt (bool): Whether to include the prompt in the pooling layer.

        Returns:
            None
        """
        for module in self:
            if isinstance(module, Pooling):
                module.include_prompt = include_prompt
                break

    def _get_prompt_length(self, prompt: str, **kwargs) -> int:
        """
        Return the length of the prompt in tokens, including the BOS token
        """
        if (prompt, *kwargs.values()) in self._prompt_length_mapping:
            return self._prompt_length_mapping[(prompt, *kwargs.values())]

        tokenized_prompt = self.tokenize([prompt], **kwargs)
        if "input_ids" not in tokenized_prompt:
            # If the tokenizer does not return input_ids, we cannot determine the prompt length.
            # This can happen with some tokenizers that do not use input_ids.
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a special EOS token, we do not count it as part of the prompt length.
        # This is to ensure that the prompt length does not include the EOS token.
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if hasattr(self.tokenizer, "all_special_ids") and last_token in self.tokenizer.all_special_ids:
            prompt_length -= 1
        self._prompt_length_mapping[(prompt, *kwargs.values())] = prompt_length
        return prompt_length

    def get_max_seq_length(self) -> int | None:
        """
        Returns the maximal sequence length that the model accepts. Longer inputs will be truncated.

        Returns:
            Optional[int]: The maximal sequence length that the model accepts, or None if it is not defined.
        """
        if hasattr(self._first_module(), "max_seq_length"):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, Tensor]:
        """
        Tokenizes the texts.

        Args:
            texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A list of texts to be tokenized.

        Returns:
            Dict[str, Tensor]: A dictionary of tensors with the tokenized texts. Common keys are "input_ids",
                "attention_mask", and "token_type_ids".
        """
        try:
            return self[0].tokenize(texts, **kwargs)
        except TypeError:
            return self[0].tokenize(texts)

    def get_sentence_features(self, *features) -> dict[Literal["sentence_embedding"], Tensor]:
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.

        Returns:
            Optional[int]: The number of dimensions in the output of `encode`. If it's not known, it's `None`.
        """
        output_dim = None
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(mod, "get_sentence_embedding_dimension", None)
            if callable(sent_embedding_dim_method):
                output_dim = sent_embedding_dim_method()
                break
        if self.truncate_dim is not None:
            # The user requested truncation. If they set it to a dim greater than output_dim,
            # no truncation will actually happen. So return output_dim instead of self.truncate_dim
            return min(output_dim or np.inf, self.truncate_dim)
        return output_dim

    @contextmanager
    def truncate_sentence_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        """
        In this context, :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` outputs
        sentence embeddings truncated at dimension ``truncate_dim``.

        This may be useful when you are using the same model for different applications where different dimensions
        are needed.

        Args:
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to. ``None`` does no truncation.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-mpnet-base-v2")

                with model.truncate_sentence_embeddings(truncate_dim=16):
                    embeddings_truncated = model.encode(["hello there", "hiya"])
                assert embeddings_truncated.shape[-1] == 16
        """
        original_output_dim = self.truncate_dim
        try:
            self.truncate_dim = truncate_dim
            yield
        finally:
            self.truncate_dim = original_output_dim

    def _first_module(self) -> torch.nn.Module:
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self) -> torch.nn.Module:
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SentenceTransformer(path)`` again.

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

        # Save some model info
        self._model_config["__version__"] = {
            "sentence_transformers": __version__,
            "transformers": transformers.__version__,
            "pytorch": torch.__version__,
        }

        with open(os.path.join(path, "config_sentence_transformers.json"), "w", encoding="utf8") as fOut:
            config = self._model_config.copy()
            config["prompts"] = self.prompts
            config["default_prompt_name"] = self.default_prompt_name
            config["similarity_fn_name"] = self.similarity_fn_name
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

            # "module" only works for Sentence Transformers as the modules have the same names as the classes
            class_ref = type(module).__module__
            # For remote modules, we want to remove "transformers_modules.{repo_name}":
            if class_ref.startswith("transformers_modules."):
                class_file = sys.modules[class_ref].__file__

                # Save the custom module file
                dest_file = Path(model_path) / (Path(class_file).name)
                shutil.copy(class_file, dest_file)

                # Save all files importeed in the custom module file
                for needed_file in get_relative_import_files(class_file):
                    dest_file = Path(model_path) / (Path(needed_file).name)
                    shutil.copy(needed_file, dest_file)

                # For remote modules, we want to ignore the "transformers_modules.{repo_id}" part,
                # i.e. we only want the filename
                class_ref = f"{class_ref.split('.')[-1]}.{type(module).__name__}"
            # For other cases, we want to add the class name:
            elif not class_ref.startswith("sentence_transformers."):
                class_ref = f"{class_ref}.{type(module).__name__}"

            module_config = {"idx": idx, "name": name, "path": os.path.basename(model_path), "type": class_ref}
            if self.module_kwargs and name in self.module_kwargs and (module_kwargs := self.module_kwargs[name]):
                module_config["kwargs"] = module_kwargs
            modules_config.append(module_config)

        with open(os.path.join(path, "modules.json"), "w", encoding="utf8") as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)

    def save_pretrained(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Saves a model and its configuration files to a directory, so that it can be loaded
        with ``SentenceTransformer(path)`` again.

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

    def _update_default_model_id(self, model_card):
        if self.model_card_data.model_id:
            model_card = model_card.replace(
                'model = SentenceTransformer("sentence_transformers_model_id"',
                f'model = SentenceTransformer("{self.model_card_data.model_id}"',
            )
        return model_card

    def _create_model_card(
        self, path: str, model_name: str | None = None, train_datasets: list[str] | None = "deprecated"
    ) -> None:
        """
        Create an automatic model and stores it in the specified path. If no training was done and the loaded model
        was a Sentence Transformer model already, then its model card is reused.

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

        # If we loaded a Sentence Transformer model from the Hub, and no training was done, then
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
        commit_message: str = "Add new SentenceTransformer model.",
        local_model_path: str | None = None,
        exist_ok: bool = False,
        replace_model_card: bool = False,
        train_datasets: list[str] | None = None,
    ) -> str:
        """
        DEPRECATED, use `push_to_hub` instead.

        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

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
                    f'Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id="{organization}/{repo_id}"` instead.'
                )
                repo_id = f"{organization}/{repo_id}"
            elif repo_id.split("/")[0] != organization:
                raise ValueError(
                    "Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`."
                )
            else:
                logger.warning(
                    f'Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id="{repo_id}"` instead.'
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
        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

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
        if revision is not None:
            api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

        if commit_message is None:
            backend = self.get_backend()
            if backend == "torch":
                commit_message = f"Add new {self.__class__.__name__} model"
            else:
                commit_message = f"Add new {self.__class__.__name__} model with an {backend} backend"

        commit_description = ""
        if create_pr:
            commit_description = f"""\
Hello!

*This pull request has been automatically generated from the [`push_to_hub`](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.push_to_hub) method from the Sentence Transformers library.*

## Full Model Architecture:
```
{self}
```

## Tip:
Consider testing this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import {self.__class__.__name__}

# TODO: Fill in the PR number
pr_number = 2
model = {self.__class__.__name__}(
    "{repo_id}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{self.get_backend()}",
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```
"""

        if local_model_path:
            folder_url = api.upload_folder(
                repo_id=repo_id,
                folder_path=local_model_path,
                commit_message=commit_message,
                commit_description=commit_description,
                revision=revision,
                create_pr=create_pr,
            )
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, "README.md"))
                self.save_pretrained(
                    tmp_dir,
                    model_name=repo_url.repo_id,
                    create_model_card=create_model_card,
                    train_datasets=train_datasets,
                    safe_serialization=safe_serialization,
                )
                folder_url = api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmp_dir,
                    commit_message=commit_message,
                    commit_description=commit_description,
                    revision=revision,
                    create_pr=create_pr,
                )

        if create_pr:
            return folder_url.pr_url
        return folder_url.commit_url

    def _text_length(self, text: list[int] | list[list[int]]) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

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

    def _load_auto_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        has_modules: bool = False,
    ) -> list[nn.Module]:
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.
            has_modules (bool, optional): Whether the model has modules.json. Defaults to False.

        Returns:
            List[nn.Module]: A list containing the transformer model and the pooling model.
        """
        logger.warning(
            f"No sentence-transformers model found with name {model_name_or_path}. Creating a new one with mean pooling."
        )

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = shared_kwargs if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        tokenizer_kwargs = shared_kwargs if tokenizer_kwargs is None else {**shared_kwargs, **tokenizer_kwargs}
        config_kwargs = shared_kwargs if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
            backend=self.backend,
        )
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), "mean")
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return [transformer_model, pooling_model]

    def _load_module_class_from_ref(
        self,
        class_ref: str,
        model_name_or_path: str,
        trust_remote_code: bool,
        revision: str | None,
        model_kwargs: dict[str, Any] | None,
    ) -> nn.Module:
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

    def _load_sbert_model(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, nn.Module]:
        """
        Loads a full SentenceTransformer model using the modules.json file.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.
            model_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the model. Defaults to None.
            tokenizer_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the tokenizer. Defaults to None.
            config_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for the config. Defaults to None.

        Returns:
            OrderedDict[str, nn.Module]: An ordered dictionary containing the modules of the model.
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
                self._model_config = json.load(fIn)

            if (
                "__version__" in self._model_config
                and "sentence_transformers" in self._model_config["__version__"]
                and version.parse(self._model_config["__version__"]["sentence_transformers"])
                > version.parse(__version__)
            ):
                logger.warning(
                    f"You are trying to use a model that was created with Sentence Transformers version {self._model_config['__version__']['sentence_transformers']}, "
                    f"but you're currently using version {__version__}. This might cause unexpected behavior or errors. "
                    "In that case, try to update to the latest version."
                )

            # Set score functions & prompts if not already overridden by the __init__ calls
            if self._similarity_fn_name is None:
                self.similarity_fn_name = self._model_config.get("similarity_fn_name", None)
            # Only update prompts that aren't already set by the user or defaults
            for prompt_name, prompt_text in self._model_config.get("prompts", {}).items():
                if prompt_name not in self.prompts or not self.prompts[prompt_name]:
                    self.prompts[prompt_name] = prompt_text
            if not self.default_prompt_name:
                self.default_prompt_name = self._model_config.get("default_prompt_name", None)
            if "model_type" not in self._model_config.keys():
                self._model_config["model_type"] = self.__class__.__name__

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

        # Load the modules of sentence transformer
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
                # If the module's `__init__` method contains specific keyword arguments like `model_args` and `config_args`,
                # it is likely Transformer-based. These arguments are commonly used in Transformer models to configure
                # the model and tokenizer during initialization.
                # Example: Models with custom modules on the Hugging Face Hub like
                # https://huggingface.co/jinaai/jina-embeddings-v3 may use this logic.
                if {"model_args", "config_args"} <= set(signature.parameters):
                    # Load initialization arguments specific to Transformer-based modules. This includes
                    # arguments for loading the model, tokenizer, and configuration, as well as any
                    # additional module-specific keyword arguments.
                    common_transformer_init_kwargs = Transformer._load_init_kwargs(
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
                        tokenizer_kwargs=tokenizer_kwargs,
                        config_kwargs=config_kwargs,
                        backend=self.backend,
                    )
                    module = module_class(model_name_or_path, **common_transformer_init_kwargs)

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
                    tokenizer_kwargs=tokenizer_kwargs,
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

    @staticmethod
    @deprecated("SentenceTransformer.load(...) is deprecated, use SentenceTransformer(...) instead.")
    def load(input_path) -> SentenceTransformer:
        return SentenceTransformer(input_path)

    @property
    def device(self) -> device:
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

    @property
    def tokenizer(self) -> Any:
        """
        Property to get the tokenizer that is used by this model
        """
        return self._first_module().tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        """
        Property to set the tokenizer that should be used by this model
        """
        self._first_module().tokenizer = value

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximal input sequence length for the model. Longer inputs will be truncated.

        Returns:
            int: The maximal input sequence length.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-mpnet-base-v2")
                print(model.max_seq_length)
                # => 384
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

        Returns:
            PreTrainedModel or None: The underlying transformers model or None if not found.

        Example:
            ::

                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-mpnet-base-v2")

                # You can now access the underlying transformers model
                transformers_model = model.transformers_model
                print(type(transformers_model))
                # => <class 'transformers.models.mpnet.modeling_mpnet.MPNetModel'>
        """
        for module in self.modules():
            if isinstance(module, PreTrainedModel):
                return module
        return None

    @property
    def _target_device(self) -> torch.device:
        logger.warning(
            "`SentenceTransformer._target_device` has been deprecated, please use `SentenceTransformer.device` instead.",
        )
        return self.device

    @_target_device.setter
    def _target_device(self, device: int | str | torch.device | None = None) -> None:
        self.to(device)

    @property
    def dtype(self) -> torch.dtype | None:
        for child in self.modules():
            if child is not self and hasattr(child, "dtype"):
                return child.dtype
        return None

    @property
    def _no_split_modules(self) -> list[str]:
        try:
            return self._first_module()._no_split_modules
        except AttributeError:
            return []

    @property
    def _keys_to_ignore_on_save(self) -> list[str]:
        try:
            return self._first_module()._keys_to_ignore_on_save
        except AttributeError:
            return []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        # Propagate the gradient checkpointing to the transformer model
        for child in self.modules():
            if isinstance(child, PreTrainedModel):
                return child.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

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

        This is used to determine the appropriate loading method:
        - SentenceTransformer: These models should be loaded with _load_sbert_model when used with SentenceTransformer class
        - SparseEncoder: These models should be loaded with _load_auto_model when used with SentenceTransformer class

        When a model type doesn't match the class being used to load it, we switch loading methods
        to ensure compatibility.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            token (Optional[Union[bool, str]]): The token to use for the model.
            cache_folder (Optional[str]): The folder to cache the model.
            revision (Optional[str], optional): The revision of the model. Defaults to None.
            local_files_only (bool, optional): Whether to use only local files. Defaults to False.

        Returns:
            Optional[str]: The model type (SentenceTransformer or SparseEncoder) if available, None otherwise.
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
            return config.get("model_type", "SentenceTransformer")  # Default to "SentenceTransformer" if not specified
