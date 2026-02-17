from __future__ import annotations

import copy
import logging
import math
import queue
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from multiprocessing import Queue
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from tqdm.autonotebook import trange
from typing_extensions import deprecated

from sentence_transformers.base.model import BaseModel
from sentence_transformers.base.models import Transformer
from sentence_transformers.base.models.modality_utils import ArrayInputs, DictInputs, ImageInputs, StrInputs
from sentence_transformers.sentence_transformer.models import Pooling
from sentence_transformers.util import batch_to_device, truncate_embeddings
from sentence_transformers.util.quantization import quantize_embeddings
from sentence_transformers.util.similarity import SimilarityFunction

from .fit_mixin import FitMixin
from .model_card import SentenceTransformerModelCardData

logger = logging.getLogger(__name__)


class SentenceTransformer(BaseModel, FitMixin):
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
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
        model_card_data (:class:`~sentence_transformers.sentence_transformer.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".

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
    default_huggingface_organization: str | None = "sentence-transformers"

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        modules: list[nn.Module] | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,  # TODO: processor_kwargs?
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        # SentenceTransformer-specific args
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        # Set default prompts for SentenceTransformer
        default_prompts = {"query": "", "document": ""}
        if prompts:
            default_prompts.update(prompts)
        prompts = default_prompts

        # SentenceTransformer-specific attributes
        self.prompts = prompts
        self.default_prompt_name = default_prompt_name
        self.similarity_fn_name = similarity_fn_name
        self.truncate_dim = truncate_dim

        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            use_auth_token=use_auth_token,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )
        self.model_card_data: SentenceTransformerModelCardData

        # Validate and log prompts
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

        # Handle INSTRUCTOR models
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

    def encode_query(
        self,
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
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
        2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.base.models.Router`
           module, it will use the "query" task type to route the input through the appropriate submodules.
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
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
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
        2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.base.models.Router`
           module, it will use the "document" task type to route the input through the appropriate submodules.
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

    # Overload signatures for type hints
    @overload
    def encode(
        self,
        sentences: StrInputs | DictInputs | ImageInputs | ArrayInputs,
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

    @overload
    def encode(
        self,
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
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

    @overload
    def encode(
        self,
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
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

    @overload
    def encode(
        self,
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs],
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

    @overload
    def encode(
        self,
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs],
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

    @overload
    def encode(
        self,
        sentences: StrInputs | DictInputs | ImageInputs | ArrayInputs,
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

    @overload
    def encode(
        self,
        sentences: StrInputs | DictInputs | ImageInputs | ArrayInputs,
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
        sentences: list[StrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,
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
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding.
            prompt (Optional[str], optional): The prompt to use for encoding.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences.
            output_value (Optional[Literal["sentence_embedding", "token_embeddings"]], optional): The type of embeddings to return.
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor.
            device (Union[str, List[str], None], optional): Device(s) to use for computation.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`.
            chunk_size (int, optional): Size of chunks for multi-process encoding.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
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
        is_singular_input = self.is_singular_input(sentences)
        if is_singular_input:
            sentences = [sentences]

        # Validate kwargs
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
            embeddings = self._multi_process(
                sentences,
                # Utility and post-processing parameters
                show_progress_bar=show_progress_bar,
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
            if is_singular_input:
                embeddings = embeddings[0]
            return embeddings

        # Validate precision
        allowed_precisions = {"float32", "int8", "uint8", "binary", "ubinary"}
        if precision and precision not in allowed_precisions:
            raise ValueError(f"Precision {precision!r} is not supported")

        # Handle prompts
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

        # Set device
        if device is None:
            device = self.device
        self.to(device)

        truncate_dim = truncate_dim if truncate_dim is not None else self.truncate_dim

        all_embeddings = []

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index : start_index + batch_size]
            features = self.preprocess(sentences_batch, prompt=prompt, **kwargs)

            # HPU-specific padding
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
                elif output_value is None:
                    embeddings = []
                    for idx in range(len(out_features["sentence_embedding"])):
                        batch_item = {}
                        for name, value in out_features.items():
                            try:
                                batch_item[name] = value[idx]
                            except TypeError:
                                batch_item[name] = value
                        embeddings.append(batch_item)
                else:
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

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

        if is_singular_input:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function."""
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
        scores between all embeddings from the first parameter and all embeddings from the second parameter.
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
        Compute the pairwise similarity between two collections of embeddings.
        """
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_pairwise

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
            This method is deprecated. You can now call :meth:`SentenceTransformer.encode`
            with the same parameters instead, which will automatically handle multi-process encoding using the provided ``pool``.
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

    def _multi_process(
        self,
        inputs: list[StrInputs | DictInputs | ImageInputs | ArrayInputs],
        show_progress_bar: bool | None = True,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **encode_kwargs,
    ):
        """Internal method for multi-process encoding."""
        convert_to_tensor = encode_kwargs.get("convert_to_tensor", False)
        convert_to_numpy = encode_kwargs.get("convert_to_numpy", False)
        encode_kwargs["show_progress_bar"] = False

        # Create a pool if not provided, but a list of devices is
        created_pool = False
        if pool is None and isinstance(device, list):
            pool = self.start_multi_process_pool(device)
            created_pool = True

        try:
            # Determine chunk size
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            # Send inputs to the input queue in chunks
            chunk_id = -1
            for chunk_id, chunk_start in enumerate(range(0, len(inputs), chunk_size)):
                chunk = inputs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, encode_kwargs])

            # Collect results from the output queue
            output_list = sorted(
                [output_queue.get() for _ in trange(chunk_id + 1, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],
            )

            # Handle the various output formats
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
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _multi_process_worker(
        target_device: str, model: SentenceTransformer, input_queue: Queue, results_queue: Queue
    ) -> None:
        """Internal working process to encode sentences in multi-process setup."""
        while True:
            try:
                chunk_id, inputs, kwargs = input_queue.get()
                embeddings = model.encode(inputs, device=target_device, **kwargs)
                # Move embeddings to CPU if needed
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
        """
        for module in self:
            if isinstance(module, Pooling):
                module.include_prompt = include_prompt
                break

    def get_sentence_features(self, *features) -> dict[Literal["sentence_embedding"], Tensor]:
        """Get sentence features from the first module."""
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SentenceTransformer.encode`.

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
            return min(output_dim or np.inf, self.truncate_dim)
        return output_dim

    @contextmanager
    def truncate_sentence_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        """
        In this context, :meth:`SentenceTransformer.encode` outputs
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
        # Call parent save method
        super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

        # Save SentenceTransformer-specific config
        import json
        import os

        config_path = os.path.join(path, "config_sentence_transformers.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf8") as fIn:
                config = json.load(fIn)
        else:
            config = {}

        # Add SentenceTransformer-specific attributes
        if self.similarity_fn_name is not None:
            config["similarity_fn_name"] = self.similarity_fn_name
        if self.truncate_dim is not None:
            config["truncate_dim"] = self.truncate_dim

        with open(config_path, "w", encoding="utf8") as fOut:
            json.dump(config, fOut, indent=2)

    def _update_default_model_id(self, model_card: str) -> str:
        """Update the default model ID in the model card."""
        if self.model_card_data.model_id:
            model_card = model_card.replace(
                'model = SentenceTransformer("sentence_transformers_model_id"',
                f'model = SentenceTransformer("{self.model_card_data.model_id}"',
            )
        return model_card

    @staticmethod
    @deprecated("SentenceTransformer.load(...) is deprecated, use SentenceTransformer(...) instead.")
    def load(input_path) -> SentenceTransformer:
        """Deprecated: Use SentenceTransformer(input_path) instead."""
        return SentenceTransformer(input_path)

    def _load_default_modules(
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
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules.
        Subclasses should override this to provide their own model creation logic.

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
            f"No {self.__class__.__name__} model found with name {model_name_or_path}. Creating a new one with mean pooling."
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
        modules = [transformer_model]
        if transformer_model.module_output_name == "token_embeddings":
            modules.append(Pooling(transformer_model.get_word_embedding_dimension(), "mean"))
        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, {}

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        # Set score functions & prompts if not already overridden by the __init__ calls
        # Only update prompts that aren't already set by the user or defaults
        for prompt_name, prompt_text in model_config.get("prompts", {}).items():
            if prompt_name not in self.prompts or not self.prompts[prompt_name]:
                self.prompts[prompt_name] = prompt_text
        if not self.default_prompt_name:
            self.default_prompt_name = model_config.get("default_prompt_name", None)
        if self._similarity_fn_name is None:
            self.similarity_fn_name = model_config.get("similarity_fn_name", None)

    def _get_model_config(self) -> dict[str, Any]:
        return super()._get_model_config() | {
            "prompts": self.prompts,
            "default_prompt_name": self.default_prompt_name,
            "similarity_fn_name": self.similarity_fn_name,
        }

    def _load_converted_modules(
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
        model_type: str | None = None,
    ) -> tuple[list[nn.Module] | OrderedDict[str, nn.Module], dict[str, Any]]:
        # If we're loading a CrossEncoder or SparseEncoder model, just load it with Transformer + Pooling
        return super()._load_default_modules(
            model_name_or_path,
            token,
            cache_folder,
            revision,
            trust_remote_code,
            local_files_only,
            model_kwargs,
            tokenizer_kwargs,
            config_kwargs,
        )
