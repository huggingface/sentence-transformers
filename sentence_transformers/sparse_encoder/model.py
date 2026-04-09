from __future__ import annotations

import itertools
import logging
import math
import queue
from collections.abc import Callable
from multiprocessing import Queue
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from tqdm import trange
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging as transformers_logging
from typing_extensions import deprecated

from sentence_transformers.base import BaseModel
from sentence_transformers.base.modality_types import TextInput
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder, SpladePooling
from sentence_transformers.util import batch_to_device, select_max_active_dims
from sentence_transformers.util.decorators import deprecated_kwargs
from sentence_transformers.util.similarity import SimilarityFunction

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


class SparseEncoder(BaseModel):
    """
    Loads or creates a SparseEncoder model that can be used to map text to sparse embeddings.

    Args:
        model_name_or_path (str, optional): If a filepath on disk, loads the model from that path. Otherwise, tries
            to download a pre-trained SparseEncoder model. If that fails, tries to construct a model from the
            Hugging Face Hub with that name. Defaults to None.
        modules (list[nn.Module], optional): A list of torch modules that are called sequentially. Can be used to
            create custom SparseEncoder models from scratch. Defaults to None.
        device (str, optional): Device (like ``"cuda"``, ``"cpu"``, ``"mps"``, ``"npu"``) that should be used for
            computation. If None, checks if a GPU can be used. Defaults to None.
        prompts (dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name,
            the value is the prompt text. The prompt text will be prepended before any text to encode. For example:
            ``{"query": "query: ", "passage": "passage: "}``. If a model has saved prompts, you can override
            them by passing your own, or pass ``{"query": "", "document": ""}`` to disable them.
            Defaults to None.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied. Defaults to None.
        cache_folder (str, optional): Path to store models. Can also be set by the ``SENTENCE_TRANSFORMERS_HOME``
            environment variable. Defaults to None.
        trust_remote_code (bool, optional): Whether to allow for custom models defined on the Hub in their own
            modeling files. Only set to ``True`` for repositories you trust and in which you have read the code,
            as it will execute code present on the Hub on your local machine. Defaults to False.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a
            commit id, for a stored model on Hugging Face. Defaults to None.
        local_files_only (bool, optional): Whether to only look at local files (i.e., do not try to download
            the model). Defaults to False.
        token (bool or str, optional): Hugging Face authentication token to download private models.
            Defaults to None.
        model_kwargs (dict[str, Any], optional): Keyword arguments passed to the underlying Hugging Face
            Transformers model via ``AutoModel.from_pretrained``. Particularly useful options include:

            - ``torch_dtype``: Override the default ``torch.dtype`` and load the model under a specific
              dtype. Can be ``torch.float16``, ``torch.bfloat16``, ``torch.float32``, or ``"auto"`` to
              use the dtype from the model's ``config.json``.
            - ``attn_implementation``: The attention implementation to use. For example ``"eager"``,
              ``"sdpa"``, or ``"flash_attention_2"``. If you ``pip install kernels``, then
              ``"flash_attention_2"`` should work without having to install ``flash_attn``. It is
              frequently the fastest option. Defaults to ``"sdpa"`` when available (torch>=2.1.1).
            - ``device_map``: Device map for model parallelism, e.g. ``"auto"``.
            - ``provider``: For ``backend="onnx"``, the ONNX execution provider
              (e.g. ``"CUDAExecutionProvider"``).
            - ``file_name``: For ``backend="onnx"`` or ``"openvino"``, the filename to load
              (e.g. for optimized or quantized models).
            - ``export``: For ``backend="onnx"`` or ``"openvino"``, whether to export the model to the
              backend format. Also set automatically if the exported file doesn't exist.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details. Defaults to None.
        processor_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            processor/tokenizer via ``AutoProcessor.from_pretrained``. See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details. Defaults to None.
        config_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            config via ``AutoConfig.from_pretrained``. See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details. Defaults to None.
        model_card_data (:class:`~sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData`, optional):
            A model card data object that contains information about the model. Used to generate a model card
            when saving the model. If not set, a default model card data object is created. Defaults to None.
        backend (str, optional): The backend to use for inference. Can be ``"torch"`` (default), ``"onnx"``,
            or ``"openvino"``. Defaults to ``"torch"``.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use.
            Valid options are ``"cosine"``, ``"dot"``, ``"euclidean"``, and ``"manhattan"``. If not set, it is
            automatically set to ``"cosine"`` when :attr:`similarity` or :attr:`similarity_pairwise` are first
            accessed. Defaults to None.
        max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the
            model. ``None`` means no limit, which can be slow or memory-intensive if your model wasn't (yet)
            finetuned to high sparsity. Defaults to None.

    Example:
        ::

            from sentence_transformers import SparseEncoder

            # Load a pre-trained SparseEncoder model
            model = SparseEncoder('naver/splade-cocondenser-ensembledistil')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 30522)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[   35.629,     9.154,     0.098],
            #         [    9.154,    27.478,     0.019],
            #         [    0.098,     0.019,    29.553]])
    """

    model_card_data_class = SparseEncoderModelCardData
    default_huggingface_organization: str | None = "sparse-encoder"
    _default_prompts: dict[str, str | None] = {"query": None, "document": None}
    _model_card_model_id_placeholder = "sparse_encoder_model_id"

    @deprecated_kwargs(tokenizer_kwargs="processor_kwargs")
    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        modules: list[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SparseEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        # SparseEncoder-specific args
        similarity_fn_name: str | SimilarityFunction | None = None,
        max_active_dims: int | None = None,
    ) -> None:
        # Set before super().__init__() so _parse_model_config can check these
        self.similarity_fn_name = similarity_fn_name

        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )
        # Narrow the type from BaseModelCardData
        self.model_card_data: SparseEncoderModelCardData

        if max_active_dims is not None and max_active_dims <= 0:
            raise ValueError(f"max_active_dims must be a positive integer, got {max_active_dims}.")
        self.max_active_dims = max_active_dims
        if max_active_dims is None:
            for module in self._modules.values():
                if isinstance(module, SparseAutoEncoder):
                    self.max_active_dims = module.k
                    break

    @deprecated_kwargs(sentences="inputs")
    def encode_query(
        self,
        inputs: list[TextInput] | TextInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | torch.device | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | Tensor:
        """
        Computes embeddings specifically optimized for query representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" prompt,
           if available in the model's ``prompts`` dictionary.
        2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.base.modules.Router`
           module, it will use the "query" task type to route the input through the appropriate submodules.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        All other parameters are identical to :meth:`encode`. See :meth:`encode` for the full parameter documentation.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                # Load a pre-trained SparseEncoder model
                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

                # Encode some texts
                queries = [
                    "What are the effects of climate change?",
                    "History of artificial intelligence",
                    "Technical specifications product XYZ",
                ]
                embeddings = model.encode_query(queries)
                print(embeddings.shape)
                # (3, 30522)
        """
        if prompt_name is None and prompt is None and "query" in self.prompts:
            prompt_name = "query"

        return self.encode(
            inputs=inputs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            device=device,
            max_active_dims=max_active_dims,
            pool=pool,
            chunk_size=chunk_size,
            task="query",
            **kwargs,
        )

    @deprecated_kwargs(sentences="inputs")
    def encode_document(
        self,
        inputs: list[TextInput] | TextInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | torch.device | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | Tensor:
        """
        Computes embeddings specifically optimized for document/passage representation.

        This method is a specialized version of :meth:`encode` that differs in exactly two ways:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses the first available prompt from the following
           candidates: ``"document"``, ``"passage"``, ``"corpus"`` (checked in that order).
        2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.base.modules.Router`
           module, it will use the "document" task type to route the input through the appropriate submodules.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        All other parameters are identical to :meth:`encode`. See :meth:`encode` for the full parameter documentation.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                # Load a pre-trained SparseEncoder model
                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

                # Encode some texts
                sentences = [
                    "This research paper discusses the effects of climate change on marine life.",
                    "The article explores the history of artificial intelligence development.",
                    "This document contains technical specifications for the new product line.",
                ]
                embeddings = model.encode_document(sentences)
                print(embeddings.shape)
                # (3, 30522)
        """
        if prompt_name is None and prompt is None:
            for candidate_prompt_name in ["document", "passage", "corpus"]:
                if candidate_prompt_name in self.prompts:
                    prompt_name = candidate_prompt_name
                    break

        return self.encode(
            inputs=inputs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            convert_to_sparse_tensor=convert_to_sparse_tensor,
            save_to_cpu=save_to_cpu,
            device=device,
            max_active_dims=max_active_dims,
            pool=pool,
            chunk_size=chunk_size,
            task="document",
            **kwargs,
        )

    @deprecated_kwargs(sentences="inputs")
    def encode(
        self,
        inputs: list[TextInput] | TextInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | torch.device | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | Tensor:
        """
        Computes sparse sentence embeddings.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        Args:
            inputs (Union[str, List[str]]): The texts to embed.
            prompt_name (str, optional): The name of the prompt to use for encoding. Must be a key in the ``prompts``
                dictionary, which is either set in the constructor or loaded from the model configuration. For example if
                ``prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What
                is the capital of France?" will be encoded as "query: What is the capital of France?" because the sentence
                is appended to the prompt. If ``prompt`` is also set, this argument is ignored. Defaults to None.
            prompt (str, optional): The prompt to use for encoding. For example, if the prompt is "query: ", then the
                sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?"
                because the sentence is appended to the prompt. If ``prompt`` is set, ``prompt_name`` is ignored.
                Defaults to None.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encoding. Defaults to None,
                in which case the progress bar will be shown if the logger's effective level is INFO or DEBUG.
            convert_to_tensor (bool, optional): Whether the output should be a single stacked tensor (True) or a list
                of individual tensors (False). Sparse tensors may be challenging to slice, so this allows you to
                output lists of tensors instead. Defaults to True.
            convert_to_sparse_tensor (bool, optional): Whether the output should be in the format of a sparse (COO) tensor.
                Defaults to True.
            save_to_cpu (bool, optional): Whether the output should be moved to cpu or stay on the device it has been
                computed on. Defaults to False.
            device (str, torch.device, list, or None, optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding

                If a list is provided, multi-process encoding will be used. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the
                model. ``None`` means the value from the model's config will be used. Defaults to None. If also None in
                the model's config, there will be no limit on the number of active dimensions, which can be slow or
                memory-intensive if your model wasn't (yet) finetuned to high sparsity.
            pool (dict, optional): A pool created by :meth:`start_multi_process_pool` for multi-process encoding.
                If provided, the encoding will be distributed across multiple processes. This is recommended for large
                datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e.
                when ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated.
                Defaults to None.

        Returns:
            Union[list[Tensor], Tensor]: By default, a 2d torch sparse tensor with shape [num_inputs, output_dimension]
            is returned. If only one string input is provided, then the output is a 1d tensor with shape
            [output_dimension]. If ``convert_to_tensor`` is False, a list of individual tensors is returned instead.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                # Load a pre-trained SparseEncoder model
                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

                # Encode some texts
                sentences = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 30522)
        """
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (
                logging.INFO,
                logging.DEBUG,
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        # Cast an individual input to a list with length 1
        is_singular_input = self.is_singular_input(inputs)
        if is_singular_input:
            inputs = [inputs]
        elif not isinstance(inputs, list):
            # Materialize e.g. datasets.Column to avoid slow Arrow deserialization on each index
            inputs = list(inputs)

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
            embeddings = self._multi_process(
                inputs=inputs,
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
                convert_to_tensor=convert_to_tensor,
                convert_to_sparse_tensor=convert_to_sparse_tensor,
                save_to_cpu=True,  # Move all embeddings to CPU to allow for concatenation
                max_active_dims=max_active_dims,
                **kwargs,
            )
            if is_singular_input:
                embeddings = embeddings[0]
            return embeddings

        prompt = self._resolve_prompt(prompt, prompt_name)

        if device is None:
            device = self.device

        self.to(device)
        self.eval()

        max_active_dims = max_active_dims if max_active_dims is not None else self.max_active_dims

        forward_kwargs = dict(kwargs)
        if max_active_dims is not None:
            forward_kwargs["max_active_dims"] = max_active_dims

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._input_length(sen) for sen in inputs])
        if self._can_flatten_inputs():
            length_sorted_idx = self._interleave_sorted_indices(length_sorted_idx)
        inputs_sorted = [inputs[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(inputs), batch_size, desc="Batches", disable=not show_progress_bar):
            inputs_batch = inputs_sorted[start_index : start_index + batch_size]
            features = self.preprocess(inputs_batch, prompt=prompt, **kwargs)
            features = batch_to_device(features, device)

            with torch.inference_mode():
                embeddings = self.forward(features, **forward_kwargs)["sentence_embedding"]

                if max_active_dims is not None:
                    embeddings = select_max_active_dims(embeddings, max_active_dims=max_active_dims)

            if convert_to_sparse_tensor:
                embeddings = embeddings.to_sparse()
            if save_to_cpu:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings) == 0:
                all_embeddings = torch.tensor([], device=self.device)
                if convert_to_sparse_tensor:
                    all_embeddings = all_embeddings.to_sparse()
                if save_to_cpu:
                    all_embeddings = all_embeddings.cpu()
            else:
                all_embeddings = torch.stack(all_embeddings)

        if is_singular_input:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _get_model_config(self) -> dict[str, Any]:
        return super()._get_model_config() | {
            "similarity_fn_name": self._similarity_fn_name,
        }

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        super()._parse_model_config(model_config)
        if self._similarity_fn_name is None:
            self.similarity_fn_name = model_config.get("similarity_fn_name", None)

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function used by :meth:`SparseEncoder.similarity` and :meth:`SparseEncoder.similarity_pairwise`.

        Returns:
            Literal["cosine", "dot", "euclidean", "manhattan"]: The name of the similarity function.
                Defaults to "dot" when first accessed if not explicitly set.

        Example:
            >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
            >>> model.similarity_fn_name
            'dot'
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self,
        value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction | None,
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    def set_pooling_include_prompt(self, include_prompt: bool) -> None:
        """
        Sets the ``include_prompt`` attribute in the pooling layer in the model, if there is one.

        This is useful for models where the prompt should be excluded from the pooling strategy,
        e.g. CSR models with a :class:`~sentence_transformers.sentence_transformer.modules.Pooling` layer.
        """
        for module in self:
            if isinstance(module, Pooling):
                module.include_prompt = include_prompt
                break

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

                >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences)
                >>> model.similarity(embeddings, embeddings)
                tensor([[   30.953,    12.871,     0.000,     0.011],
                        [   12.871,    27.505,     0.580,     0.578],
                        [    0.000,     0.580,    36.068,    15.301],
                        [    0.011,     0.578,    15.301,    39.466]])
                >>> model.similarity_fn_name
                "dot"
                >>> model.similarity_fn_name = "cosine"
                >>> model.similarity(embeddings, embeddings)
                tensor([[    1.000,     0.441,     0.000,     0.000],
                        [    0.441,     1.000,     0.018,     0.018],
                        [    0.000,     0.018,     1.000,     0.406],
                        [    0.000,     0.018,     0.406,     1.000]])
        """
        # Access the property to trigger lazy initialization if needed
        self.similarity_fn_name  # noqa: B018
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

                >>> model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                >>> sentences = [
                ...     "The weather is so nice!",
                ...     "It's so sunny outside.",
                ...     "He's driving to the movie theater.",
                ...     "She's going to the cinema.",
                ... ]
                >>> embeddings = model.encode(sentences, convert_to_sparse_tensor=False)
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([12.871, 15.301])
                >>> model.similarity_fn_name
                "dot"
                >>> model.similarity_fn_name = "cosine"
                >>> model.similarity_pairwise(embeddings[::2], embeddings[1::2])
                tensor([0.441, 0.406])
        """
        # Access the property to trigger lazy initialization if needed
        self.similarity_fn_name  # noqa: B018
        return self._similarity_pairwise

    def _multi_process(
        self,
        inputs: list[TextInput],
        show_progress_bar: bool | None = True,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | torch.device | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **encode_kwargs,
    ) -> list[Tensor] | Tensor:
        """Internal method for multi-process encoding.

        Distributes encoding across multiple processes using the provided pool or list of devices.
        If a pool is not provided but ``device`` is a list, a pool is created and cleaned up automatically.
        """
        convert_to_tensor = encode_kwargs.get("convert_to_tensor", False)
        encode_kwargs["show_progress_bar"] = False

        # Create a pool if not provided, but a list of devices is
        created_pool = False
        if pool is None and isinstance(device, list):
            pool = self.start_multi_process_pool(device)
            created_pool = True

        try:
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            num_chunks = math.ceil(len(inputs) / chunk_size) if inputs else 0
            for chunk_id in range(num_chunks):
                chunk_start = chunk_id * chunk_size
                chunk = inputs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, encode_kwargs])

            output_list = sorted(
                [output_queue.get() for _ in trange(num_chunks, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],
            )

            # Check for errors from worker processes
            for output in output_list:
                if isinstance(output[1], Exception):
                    raise output[1]

            # Handle the various output formats
            embeddings = [output[1] for output in output_list]
            if embeddings:
                if isinstance(embeddings[0], list):
                    embeddings = list(itertools.chain.from_iterable(embeddings))
                elif isinstance(embeddings[0], torch.Tensor):
                    embeddings = torch.cat(embeddings)
                elif isinstance(embeddings[0], np.ndarray):
                    embeddings = np.concatenate(embeddings, axis=0)
            elif convert_to_tensor:
                embeddings = torch.Tensor()
            return embeddings

        finally:
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _multi_process_worker(
        target_device: str, model: SparseEncoder, input_queue: Queue, results_queue: Queue
    ) -> None:
        """Internal working process to encode sentences in multi-process setup.

        Workers are terminated externally via ``stop_multi_process_pool``.
        """
        while True:
            try:
                chunk_id, inputs, kwargs = input_queue.get()
                embeddings = model.encode(inputs, device=target_device, **kwargs)
                if isinstance(embeddings, torch.Tensor) and embeddings.device.type != "cpu":
                    embeddings = embeddings.cpu()
                results_queue.put([chunk_id, embeddings])

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error in worker process on {target_device}: {e}")
                try:
                    results_queue.put([chunk_id, e])
                except Exception:
                    pass
                break

    def get_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SparseEncoder.encode`.

        Unlike :class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`, sparse encoders do not support ``truncate_dim``,
        so this returns the raw output dimension from the last module in the pipeline.

        Returns:
            int or None: The number of dimensions in the output of ``encode``. If it's not known, it's ``None``.
        """
        output_dim = None
        for module in reversed(self._modules.values()):
            for name in (
                "get_embedding_dimension",
                "get_sentence_embedding_dimension",
                "get_word_embedding_dimension",
            ):
                method = getattr(module, name, None)
                if callable(method):
                    output_dim = method()
                    break
            if output_dim is not None:
                break
        return output_dim

    @deprecated(
        "The `get_sentence_embedding_dimension` method has been renamed to `get_embedding_dimension`.",
        category=FutureWarning,
    )
    def get_sentence_embedding_dimension(self) -> int | None:
        return self.get_embedding_dimension()

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
    ) -> tuple[list[nn.Module], dict[str, Any]]:
        """
        Creates a simple transformer-based model and returns the modules.
        For models with a ForMaskedLM architecture, uses SpladePooling with 'max' strategy.
        For regular Transformers, uses a CSR implementation (Pooling + SparseAutoEncoder) by default.

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
            tuple[list[nn.Module], dict[str, Any]]: The modules and an empty kwargs dict.
        """
        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = {**shared_kwargs, **(model_kwargs or {})}
        processor_kwargs = {**shared_kwargs, **(processor_kwargs or {})}
        config_kwargs = {**shared_kwargs, **(config_kwargs or {})}

        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_folder, **config_kwargs
        )

        is_mlm_model = any(arch.endswith("ForMaskedLM") for arch in getattr(config, "architectures", None) or [])

        if is_mlm_model:
            # For MLM models like BERT, RoBERTa, etc., use Transformer w. fill-mask with SpladePooling
            transformer_model = Transformer(
                model_name_or_path,
                transformer_task="fill-mask",
                cache_dir=cache_folder,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                config_kwargs=config_kwargs,
                backend=self.backend,
            )
            logger.info("Detected MLM architecture, using SpladePooling")
            pooling_model = SpladePooling(pooling_strategy="max")
            modules = [transformer_model, pooling_model]

        else:
            logger.info(
                "No MLM architecture detected, using default Transformer + mean Pooling + SparseAutoEncoder (CSR)"
            )
            transformer_model = Transformer(
                model_name_or_path,
                transformer_task="feature-extraction",
                cache_dir=cache_folder,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                config_kwargs=config_kwargs,
                backend=self.backend,
            )
            pooling = Pooling(transformer_model.get_embedding_dimension(), pooling_mode="mean")
            sae = SparseAutoEncoder(
                input_dim=pooling.get_embedding_dimension(),
                hidden_dim=4 * pooling.get_embedding_dimension(),
                k=256,  # Number of top values to keep
                k_aux=512,  # Number of top values for auxiliary loss
            )
            modules = [transformer_model, pooling, sae]

        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, {}

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
    ) -> tuple[list[nn.Module], dict[str, Any]]:
        """Converts a non-SparseEncoder model into a SparseEncoder by appending a SparseAutoEncoder.

        If ``model_type`` is ``"SentenceTransformer"``, loads the SentenceTransformer modules and appends a
        SparseAutoEncoder on top. Otherwise, falls back to :meth:`_load_default_modules`.
        """
        if model_type != "SentenceTransformer":
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

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = {**shared_kwargs, **(model_kwargs or {})}
        processor_kwargs = {**shared_kwargs, **(processor_kwargs or {})}
        config_kwargs = {**shared_kwargs, **(config_kwargs or {})}

        logger.info("SentenceTransformer model found, appending SparseAutoEncoder on top to form a CSR model")
        modules, self.module_kwargs = self._load_config_modules(
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
        modules = list(modules.values())
        # Use the output dimension of the last module as the SAE input dimension
        output_dim = None
        for module in reversed(modules):
            if hasattr(module, "get_embedding_dimension"):
                output_dim = module.get_embedding_dimension()
                break
        if output_dim is None:
            raise ValueError(
                "Cannot determine the embedding dimension from the loaded modules. "
                "At least one module must have a `get_embedding_dimension` method."
            )
        sae = SparseAutoEncoder(
            input_dim=output_dim,
            hidden_dim=4 * output_dim,
            k=output_dim // 4,  # Number of top values to keep
            k_aux=output_dim // 2,  # Number of top values for auxiliary loss
        )
        modules.append(sae)
        # The original README is not useful for this different architecture
        self._model_card_text = None
        return modules, self.module_kwargs

    @staticmethod
    def sparsity(embeddings: torch.Tensor) -> dict[str, float]:
        """
        Calculate sparsity statistics for the given embeddings, including the mean number of active
        (non-zero) dimensions and the mean sparsity ratio.

        For a single embedding (1D), the values are for that embedding directly. For a batch of embeddings
        (2D), they are averaged across the batch.

        Args:
            embeddings (torch.Tensor): The embeddings to analyze. Must be a 1D or 2D tensor.

        Returns:
            dict[str, float]: Dictionary with ``"active_dims"`` (mean active dimensions) and
                ``"sparsity_ratio"`` (mean sparsity ratio).

        Example:
            ::

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                embeddings = model.encode(["The weather is so nice!", "It's so sunny outside."])
                stats = model.sparsity(embeddings)
                print(stats)
                # => {'active_dims': 44.0, 'sparsity_ratio': 0.9985584020614624}
        """
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("Embeddings must be a torch.Tensor")

        if embeddings.ndim not in (1, 2):
            raise ValueError(f"Expected a 1D or 2D tensor, got {embeddings.ndim}D.")

        # Normalize 1D to 2D so we can use a single code path
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        num_rows, num_cols = embeddings.shape

        if num_rows == 0 or num_cols == 0:
            return {
                "active_dims": 0.0,
                "sparsity_ratio": 1.0,
            }

        # CSR gives O(1) per-row non-zero counts via crow_indices
        embeddings = embeddings.to_sparse_csr()
        crow_indices = embeddings.crow_indices()
        non_zero_per_row = crow_indices[1:] - crow_indices[:-1]

        mean_active_dims = torch.mean(non_zero_per_row.float()).item()
        mean_sparsity_ratio = 1.0 - (mean_active_dims / num_cols)

        return {
            "active_dims": mean_active_dims,
            "sparsity_ratio": mean_sparsity_ratio,
        }

    @property
    def max_seq_length(self) -> int:
        """
        Returns the maximal input sequence length for the model. Longer inputs will be truncated.

        Returns:
            int: The maximal input sequence length.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                print(model.max_seq_length)
                # => 512
        """
        return super().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        Property to set the maximal input sequence length for the model. Longer inputs will be truncated.
        """
        # Setter must be re-declared because the getter is overridden (Python property limitation)
        self[0].max_seq_length = value

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

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-v3")

                # You can now access the underlying transformers model
                transformers_model = model.transformers_model
                print(type(transformers_model))
                # => <class 'transformers.models.bert.modeling_bert.BertForMaskedLM'>
        """
        return super().transformers_model

    def _get_splade_pooling(self) -> SpladePooling | None:
        """Returns the SpladePooling module if present, or None. Only searches top-level modules."""
        for module in self._modules.values():
            if isinstance(module, SpladePooling):
                return module
        return None

    @property
    def splade_pooling_chunk_size(self) -> int | None:
        """
        Returns the chunk size of the SpladePooling module, if present.

        This chunk size is along the sequence length dimension (i.e., number of tokens per chunk).
        If None, processes the entire sequence at once. Using smaller chunks reduces memory usage but may
        lower training and inference speed. Default is None.

        This property is only meaningful for SPLADE-architecture models. For CSR-architecture models
        (Transformer + Pooling + SparseAutoEncoder), it returns None.

        Returns:
            int or None: The chunk size, or None if SpladePooling is not found or chunk_size is not set.
        """
        splade_pooling = self._get_splade_pooling()
        if splade_pooling is not None:
            return splade_pooling.chunk_size
        logger.warning("SpladePooling module not found. Cannot get chunk_size.")
        return None

    @splade_pooling_chunk_size.setter
    def splade_pooling_chunk_size(self, value: int | None) -> None:
        """
        Sets the chunk size of the SpladePooling module, if present.
        """
        splade_pooling = self._get_splade_pooling()
        if splade_pooling is not None:
            splade_pooling.chunk_size = value
        else:
            logger.warning("SpladePooling module not found. Cannot set chunk_size.")

    @staticmethod
    def intersection(
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
    ) -> Tensor:
        """
        Compute the intersection of two sparse embeddings via element-wise multiplication.

        For each dimension, the result retains the minimum contribution from both embeddings, keeping only
        dimensions where both inputs are positive (i.e., shared active dimensions). This is useful for
        token-level matching and interpretability when combined with :meth:`decode`.

        Args:
            embeddings_1 (torch.Tensor): First embedding tensor of shape ``(vocab_size,)``.
            embeddings_2 (torch.Tensor): Second embedding tensor of shape ``(vocab_size,)`` or
                ``(batch_size, vocab_size)``.

        Returns:
            torch.Tensor: Sparse intersection tensor with the same shape as ``embeddings_2``.

        Example:
            ::

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                query_emb = model.encode_query("What is AI?")
                doc_emb = model.encode_document("Artificial intelligence is a branch of computer science.")
                shared = model.intersection(query_emb, doc_emb)
                print(model.decode(shared, top_k=5))
        """
        if not embeddings_1.is_sparse:
            embeddings_1 = embeddings_1.to_sparse()
        if not embeddings_2.is_sparse:
            embeddings_2 = embeddings_2.to_sparse()

        if embeddings_1.ndim != 1:
            raise ValueError(f"Expected 1D tensor for embeddings_1, but got {embeddings_1.shape} shape.")

        if embeddings_1.shape[-1] != embeddings_2.shape[-1]:
            raise ValueError(
                f"Vocab dimension mismatch: embeddings_1 has {embeddings_1.shape[-1]}, "
                f"embeddings_2 has {embeddings_2.shape[-1]}."
            )

        if embeddings_2.ndim == 1:
            intersection = embeddings_1 * embeddings_2
        elif embeddings_2.ndim == 2:
            # Element-wise multiplication per row; Python loop is required as sparse broadcasting is limited
            intersection = torch.stack([embeddings_1 * embedding for embedding in embeddings_2])
        else:
            raise ValueError(f"Expected 1D or 2D tensor for embeddings_2, but got {embeddings_2.shape} shape.")

        # Coalesce to sum duplicate indices, then keep only positive values (shared active dimensions)
        intersection = intersection.coalesce()
        active_dims = intersection.values() > 0
        intersection = torch.sparse_coo_tensor(
            intersection.indices()[:, active_dims],
            intersection.values()[active_dims],
            size=intersection.size(),
            device=intersection.device,
        )

        return intersection

    def decode(
        self, embeddings: torch.Tensor, top_k: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """
        Decode a sparse embedding into (token, weight) pairs sorted by descending weight.

        Args:
            embeddings (torch.Tensor): Sparse embedding tensor of shape ``(vocab_size,)``
                for a single embedding or ``(batch_size, vocab_size)`` for a batch.
            top_k (int, optional): Maximum number of top-weighted tokens to return per sample.
                If ``None``, all non-zero tokens are returned. Must be positive. Defaults to ``None``.

        Returns:
            list[tuple[str, float]]: If the input is 1D, a list of ``(token, weight)`` tuples.
            list[list[tuple[str, float]]]: If the input is 2D, a list (one per sample)
                of lists of ``(token, weight)`` tuples.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(embeddings)}")

        # Track whether input was 1D so we can unwrap at the end
        was_1d = embeddings.ndim == 1
        if was_1d:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.ndim != 2:
            raise ValueError(f"Input tensor must be 1D or 2D, got {embeddings.ndim}D.")

        # Ensure COO sparse format for uniform .indices()/.values() access
        if not embeddings.is_sparse:
            embeddings = embeddings.to_sparse()

        embeddings = embeddings.coalesce()
        indices = embeddings.indices()
        values = embeddings.values()

        if values.numel() == 0:
            results: list[list[tuple[str, float]]] = [[] for _ in range(embeddings.size(0))]
            return results[0] if was_1d else results

        sample_indices, token_indices = indices[0], indices[1]
        sample_counts = torch.bincount(sample_indices, minlength=embeddings.size(0)).tolist()

        results = []
        start_idx = 0
        for count in sample_counts:
            if count == 0:
                results.append([])
                continue

            sample_values = values[start_idx : start_idx + count]
            sample_tokens = token_indices[start_idx : start_idx + count]

            effective_k = min(top_k, count) if top_k is not None else count
            if effective_k < count:
                top_values, top_idx = torch.topk(sample_values, effective_k)
                sample_tokens = sample_tokens[top_idx]
                sample_values = top_values
            else:
                sorted_idx = torch.argsort(sample_values, descending=True)
                sample_values = sample_values[sorted_idx]
                sample_tokens = sample_tokens[sorted_idx]

            token_strs = self.tokenizer.convert_ids_to_tokens(sample_tokens.tolist())
            results.append(list(zip(token_strs, sample_values.tolist())))

            start_idx += count

        return results[0] if was_1d else results

    def _push_to_hub_usage_tip(self, repo_id: str) -> str:
        class_name = self.__class__.__name__
        backend = self.get_backend()
        return f"""\
## Testing this pull request
You can test this pull request before merging by loading the model from this PR with the `revision` argument:
```python
from sentence_transformers import {class_name}

# NOTE: Update this to the number of your pull request
pr_number = 2
model = {class_name}(
    "{repo_id}",
    revision=f"refs/pr/{{pr_number}}",
    backend="{backend}",
)

# Verify that everything works as expected
embeddings = model.encode(["The weather is lovely today.", "It's so sunny outside!", "He drove to the stadium."])
print(embeddings.shape)

similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

---
*This PR was auto-generated with \
[`push_to_hub`](https://sbert.net/docs/package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub).*
"""
