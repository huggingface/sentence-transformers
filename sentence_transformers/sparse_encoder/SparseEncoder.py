from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from tqdm import trange
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from typing_extensions import deprecated

from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.model_card import SparseEncoderModelCardData
from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseAutoEncoder, SpladePooling
from sentence_transformers.util import batch_to_device, select_max_active_dims

logger = logging.getLogger(__name__)


class SparseEncoder(SentenceTransformer):
    """
    Loads or creates a SparseEncoder model that can be used to map sentences / text to sparse embeddings.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disk, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SparseEncoder model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SparseEncoder models from scratch.
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
        max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. Defaults to None. This means there will be no
            limit on the number of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.
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
        model_card_data (:class:`~sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
            on the different backends.

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
        max_active_dims: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SparseEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            similarity_fn_name=similarity_fn_name,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )
        if max_active_dims is not None:
            self.max_active_dims = max_active_dims
        else:
            for module in self._modules.values():
                if isinstance(module, SparseAutoEncoder):
                    self.max_active_dims = module.k
                    break
            else:
                self.max_active_dims = max_active_dims

    def encode_query(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
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
            convert_to_tensor (bool, optional): Whether the output should be a single stacked tensor (True) or a list
                of individual tensors (False). Sparse tensors may be challenging to slice, so this allows you to
                output lists of tensors instead. Defaults to True.
            convert_to_sparse_tensor (bool, optional): Whether the output should be in the format of a sparse (COO) tensor.
                Defaults to True.
            save_to_cpu (bool, optional):  Whether the output should be moved to cpu or stay on the device it has been computed on.
                Defaults to False
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d torch sparse tensor with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If save_to_cpu is True,
            the embeddings are moved to the CPU.

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
        if prompt_name is None and "query" in self.prompts and prompt is None:
            prompt_name = "query"

        return self.encode(
            sentences=sentences,
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

    def encode_document(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
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
            convert_to_tensor (bool, optional): Whether the output should be a single stacked tensor (True) or a list
                of individual tensors (False). Sparse tensors may be challenging to slice, so this allows you to
                output lists of tensors instead. Defaults to True.
            convert_to_sparse_tensor (bool, optional): Whether the output should be in the format of a sparse (COO) tensor.
                Defaults to True.
            save_to_cpu (bool, optional):  Whether the output should be moved to cpu or stay on the device it has been computed on.
                Defaults to False
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d torch sparse tensor with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If save_to_cpu is True,
            the embeddings are moved to the CPU.

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
                embeddings = model.encode(sentences)
                print(embeddings.shape)
                # (3, 30522)
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

    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = True,
        convert_to_sparse_tensor: bool = True,
        save_to_cpu: bool = False,
        device: str | list[str | torch.device] | None = None,
        max_active_dims: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
        """
        Computes sparse sentence embeddings.

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
            convert_to_tensor (bool, optional): Whether the output should be a single stacked tensor (True) or a list
                of individual tensors (False). Sparse tensors may be challenging to slice, so this allows you to
                output lists of tensors instead. Defaults to True.
            convert_to_sparse_tensor (bool, optional): Whether the output should be in the format of a sparse (COO) tensor.
                Defaults to True.
            save_to_cpu (bool, optional):  Whether the output should be moved to cpu or stay on the device it has been computed on.
                Defaults to False
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d torch sparse tensor with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If save_to_cpu is True,
            the embeddings are moved to the CPU.

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
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (
                logging.INFO,
                logging.DEBUG,
            )

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
                convert_to_tensor=convert_to_tensor,
                convert_to_sparse_tensor=convert_to_sparse_tensor,
                save_to_cpu=True,  # Move all embeddings to CPU to allow for concatenation
                max_active_dims=max_active_dims,
                **kwargs,
            )

        # Original encoding logic when not using multi-process
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
            tokenized_prompt = self.tokenize([prompt], **kwargs)
            if "input_ids" in tokenized_prompt:
                extra_features["prompt_length"] = tokenized_prompt["input_ids"].shape[-1] - 1

        # Here, device is either a single device string (e.g., "cuda:0", "cpu") for single-process encoding or None
        if device is None:
            device = self.device

        self.to(device)

        max_active_dims = max_active_dims if max_active_dims is not None else self.max_active_dims
        if max_active_dims is not None:
            kwargs["max_active_dims"] = max_active_dims

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, **kwargs)
            features = batch_to_device(features, self.device)
            features.update(extra_features)

            with torch.inference_mode():
                embeddings = self.forward(features, **kwargs)["sentence_embedding"].detach()

                if max_active_dims:
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

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function used by :meth:`SparseEncoder.similarity` and :meth:`SparseEncoder.similarity_pairwise`.

        Returns:
            Optional[str]: The name of the similarity function. Can be None if not set, in which case it will
                default to "cosine" when first called.

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
        value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction,
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

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
                >>> embeddings = model.encode(sentences, normalize_embeddings=True)
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
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
        return self._similarity

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
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.DOT
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
        max_active_dims: int | None = None,
    ) -> Tensor:
        """
        .. warning::
            This method is deprecated. You can now call :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
            with the same parameters instead, which will automatically handle multi-process encoding using the provided ``pool``.

        Encodes a list of sentences using multiple processes and GPUs via
        :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`.
        The sentences are chunked into smaller packages and sent to individual processes, which encode them on different
        GPUs or CPUs. This method is only suitable for encoding large sets of sentences.

        Args:
            sentences (List[str]): List of sentences to encode.
            pool (Dict[Literal["input", "output", "processes"], Any]): A pool of workers started with
                :meth:`SparseEncoder.start_multi_process_pool <sentence_transformers.sparse_encoder.SparseEncoder.start_multi_process_pool>`.
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
            max_active_dims (int, optional): The maximum number of active (non-zero) dimensions in the output of the model. `None` means we will
                used the value of the model's config. Defaults to None. If None in model's config it means there will be no limit on the number
                of active dimensions and can be slow or memory-intensive if your model wasn't (yet) finetuned to high sparsity.

        Returns:
            Tensor: A 2D tensor with shape [num_inputs, output_dimension].

        Example:
            ::

                from sentence_transformers import SparseEncoder

                def main():
                    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                    sentences = [
                        "The weather is so nice!",
                        "It's so sunny outside.",
                        "He's driving to the movie theater.",
                        "She's going to the cinema.",
                    ] * 1000

                    pool = model.start_multi_process_pool()
                    embeddings = model.encode_multi_process(sentences, pool)
                    model.stop_multi_process_pool(pool)

                    print(embeddings.shape)
                    # => (4000, 30522)

                if __name__ == "__main__":
                    main()
        """
        return self.encode(
            sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            save_to_cpu=False,
            max_active_dims=max_active_dims,
            pool=pool,
            chunk_size=chunk_size,
        )

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns the number of dimensions in the output of :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`.
        We override the function without updating regarding the truncate dim as for sparse model the dimension of the output
        is the same, only the active dimensions number changes.

        Returns:
            Optional[int]: The number of dimensions in the output of `encode`. If it's not known, it's `None`.
        """
        output_dim = None
        for mod in reversed(self._modules.values()):
            sent_embedding_dim_method = getattr(mod, "get_sentence_embedding_dimension", None)
            if callable(sent_embedding_dim_method):
                output_dim = sent_embedding_dim_method()
                break
        return output_dim

    @contextmanager
    def truncate_sentence_embeddings(self, truncate_dim: int | None) -> Iterator[None]:
        raise NotImplementedError(
            "SparseEncoder does not support truncating sentence embeddings. "
            "Use the `max_active_dims` parameter in the encode method instead if you want to limit the embedding memory usage."
        )

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
        with ``SparseEncoder(path)`` again.

        Args:
            path (str): Path on disk where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        return super().save(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

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
        with ``SparseEncoder(path)`` again.

        Args:
            path (str): Path on disk where the model will be saved.
            model_name (str, optional): Optional model name.
            create_model_card (bool, optional): If True, create a README.md with basic information about this model.
            train_datasets (List[str], optional): Optional list with the names of the datasets used to train the model.
            safe_serialization (bool, optional): If True, save the model using safetensors. If False, save the model
                the traditional (but unsafe) PyTorch way.
        """
        return super().save_pretrained(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    def _update_default_model_id(self, model_card):
        if self.model_card_data.model_id:
            model_card = model_card.replace(
                'model = SparseEncoder("sparse_encoder_model_id"',
                f'model = SparseEncoder("{self.model_card_data.model_id}"',
            )
        return model_card

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
        Uploads all elements of this Sparse Encoder to a new HuggingFace Hub repository.

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
        return super().push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            safe_serialization=safe_serialization,
            commit_message=commit_message,
            local_model_path=local_model_path,
            exist_ok=exist_ok,
            replace_model_card=replace_model_card,
            train_datasets=train_datasets,
            revision=revision,
            create_pr=create_pr,
        )

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
        Creates a simple transformer-based model and returns the modules.
        For MLMTransformer (models ending with ForMaskedLM), uses SpladePooling with 'max' strategy.
        For regular Transformer, uses CSR implementation by default.

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
            f"No sparse-encoder model found with name {model_name_or_path}. Creating a new one with defaults settings compatible to the base model."
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

        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_folder, **config_kwargs)

        # Check if the architecture ends with "ForMaskedLM"
        is_mlm_model = False
        if hasattr(config, "architectures") and config.architectures:
            for architecture in config.architectures:
                if architecture.endswith("ForMaskedLM"):
                    is_mlm_model = True
                    break
        if has_modules:
            logger.info(
                "A SentenceTransformer model found, using Sentence Transformer modules with SparseAutoEncoder modules on top to form a CSR model"
            )
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
            modules = [modules[str(i)] for i in range(len(modules.keys()))]
            input_dim = modules[0].get_word_embedding_dimension()
            hidden_dim = 4 * input_dim
            k = input_dim // 4  # Number of top values to keep
            k_aux = input_dim // 2  # Number of top values for auxiliary loss
            sae = SparseAutoEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                k=k,
                k_aux=k_aux,
            )
            modules.append(sae)
            self._model_card_text = None  # If we're loading a SentenceTransformer model, but adding a SparseAutoEncoder, then the original README isn't useful anymore as it's a different architecture

        elif is_mlm_model:
            # For MLM models like BERT, RoBERTa, etc., use MLMTransformer with SpladePooling
            logger.info(f"Detected MLM architecture: {config.architectures}, using SpladePooling")
            transformer_model = MLMTransformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args=model_kwargs,
                tokenizer_args=tokenizer_kwargs,
                config_args=config_kwargs,
                backend=self.backend,
            )
            pooling_model = SpladePooling(pooling_strategy="max")

            modules = [transformer_model, pooling_model]

        else:
            logger.info(
                "No MLM model found and no SentenceTransformer model found, using default transformer modules and mean pooling with SparseAutoEncoder modules on top to form a CSR model"
            )
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args=model_kwargs,
                tokenizer_args=tokenizer_kwargs,
                config_args=config_kwargs,
                backend=self.backend,
            )
            pooling = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode="mean")
            sae = SparseAutoEncoder(
                input_dim=pooling.get_sentence_embedding_dimension(),
                hidden_dim=4 * pooling.get_sentence_embedding_dimension(),
                k=256,  # Number of top values to keep
                k_aux=512,  # Number of top values for auxiliary loss
            )
            modules = [transformer_model, pooling, sae]

        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules

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
        Loads a full SparseEncoder model using the modules.json file.

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
        return super()._load_sbert_model(
            model_name_or_path=model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
        )

    @staticmethod
    @deprecated("SparseEncoder.load(...) is deprecated, use SparseEncoder(...) instead.")
    def load(input_path) -> SparseEncoder:
        return SparseEncoder(input_path)

    @staticmethod
    def sparsity(embeddings: torch.Tensor) -> dict[str, float]:
        """
        Calculate sparsity statistics for the given embeddings, including the mean number of active dimensions
        and the mean sparsity ratio.

        Args:
            embeddings (torch.Tensor): The embeddings to analyze.

        Returns:
            dict[str, float]: Dictionary with the mean active dimensions and mean sparsity ratio.

        Example
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

        # Handle 1D tensor case
        if embeddings.ndim == 1:
            num_cols = embeddings.shape[0]
            if not embeddings.is_sparse:
                embeddings = embeddings.to_sparse()
            num_active_dims = embeddings.coalesce().indices().shape[1]
            sparsity_ratio = 1.0 - (num_active_dims / num_cols)
            return {
                "active_dims": float(num_active_dims),
                "sparsity_ratio": float(sparsity_ratio),
            }

        # Handle 2D tensor case
        num_rows, num_cols = embeddings.shape

        if num_rows == 0 or num_cols == 0:
            return {
                "active_dims": 0.0,
                "sparsity_ratio": 1.0,
            }

        # Convert to the CSR format for convenience
        embeddings = embeddings.to_sparse_csr()

        # Calculate non-zero elements per row
        crow_indices = embeddings.crow_indices()
        non_zero_per_row = crow_indices[1:] - crow_indices[:-1]

        # Calculate mean values
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

                from sentence_transformers import SparseEncoder

                model = SparseEncoder("naver/splade-v3")

                # You can now access the underlying transformers model
                transformers_model = model.transformers_model
                print(type(transformers_model))
                # => <class 'transformers.models.bert.modeling_bert.BertForMaskedLM'>
        """
        return super().transformers_model

    @property
    def splade_pooling_chunk_size(self) -> int | None:
        """
        Returns the chunk size of the SpladePooling module, if present.
        This Chunk size is along the sequence length dimension (i.e., number of tokens per chunk).
        If None, processes entire sequence at once. Using smaller chunks the reduces memory usage but may
        lower the training and inference speed. Default is None.

        Returns:
            Optional[int]: The chunk size, or None if SpladePooling is not found or chunk_size is not set.
        """
        for mod in self._modules.values():
            if isinstance(mod, SpladePooling):
                return mod.chunk_size
        logger.warning("SpladePooling module not found. Cannot get chunk_size.")
        return None

    @splade_pooling_chunk_size.setter
    def splade_pooling_chunk_size(self, value: int) -> None:
        """
        Sets the chunk size of the SpladePooling module, if present.
        """
        for mod in self._modules.values():
            if isinstance(mod, SpladePooling):
                mod.chunk_size = value
                break
        else:
            logger.warning("SpladePooling module not found. Cannot set chunk_size.")

    def intersection(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
    ) -> Tensor:
        """
        Compute the intersection of two sparse embeddings.

        Args:
            embeddings_1 (torch.Tensor): First embedding tensor, (vocab).
            embeddings_2 (torch.Tensor): Second embedding tensor, (vocab) or (batch_size, vocab).

        Returns:
            torch.Tensor: Intersection of the two embeddings.
        """
        if not embeddings_1.is_sparse:
            embeddings_1 = embeddings_1.to_sparse()
        if not embeddings_2.is_sparse:
            embeddings_2 = embeddings_2.to_sparse()

        if embeddings_1.ndim != 1:
            raise ValueError(f"Expected 1D tensor for embeddings_1, but got {embeddings_1.shape} shape.")

        if embeddings_2.ndim == 1:
            intersection = embeddings_1 * embeddings_2
        elif embeddings_2.ndim == 2:
            intersection = torch.stack([embeddings_1 * embedding for embedding in embeddings_2])
        else:
            raise ValueError(f"Expected 1D tensor or 2D tensor for embeddings_2, but got {embeddings_2.shape} shape.")

        # Cheaply remove zero values
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
        Decode top K tokens and weights from a sparse embedding.
        If none will just return the all tokens and weights

        Args:
            embeddings (torch.Tensor): Sparse embedding tensor (batch, vocab) or (vocab).
            top_k (int, optional): Number of top tokens to return per sample. If None, returns all non-zero tokens.

        Returns:
            list[tuple[str, float]] | list[list[tuple[str, float]]]: List of tuples (token, weight) for each embedding.
            If batch input, returns a list of lists of tuples.
        """
        # Ensure we have a sparse tensor for efficient processing
        if not embeddings.is_sparse and not getattr(embeddings, "is_sparse_csr", False):
            embeddings = embeddings.to_sparse()

        # For a single embedding vector
        if embeddings.dim() == 1:
            embeddings = embeddings.coalesce() if embeddings.is_sparse else embeddings
            values = embeddings.values()
            indices = embeddings.indices().squeeze()
            if values.numel() == 0:
                return []

            # Apply top-k if specified
            if top_k is not None:
                top_values, top_idx = torch.topk(values, min(top_k, values.numel()))
                indices = indices[top_idx]
                values = top_values
            else:
                # Sort values and indices
                sorted_indices = torch.argsort(values, descending=True)
                indices = indices[sorted_indices]
                values = values[sorted_indices]

            # Convert token IDs to strings
            tokens = self.tokenizer.convert_ids_to_tokens(indices.tolist())

            # Return a dictionary mapping tokens to weights
            return list(zip(tokens, values.tolist()))

        # For a batch of embeddings
        elif embeddings.dim() == 2:
            embeddings = embeddings.coalesce() if embeddings.is_sparse else embeddings

            # Extract indices and values
            indices = embeddings.indices()
            values = embeddings.values()

            if values.numel() == 0:
                return [{}] * embeddings.size(0)

            # Sample indices (first dimension) and token indices (second dimension)
            sample_indices, token_indices = indices[0], indices[1]

            # Count tokens per sample
            sample_counts = torch.bincount(sample_indices, minlength=embeddings.size(0)).tolist()

            # Apply top-k if specified
            if top_k is not None:
                results = []
                start_idx = 0
                for i, count in enumerate(sample_counts):
                    if count == 0:
                        results.append([])
                        continue

                    sample_values = values[start_idx : start_idx + count]
                    sample_tokens = token_indices[start_idx : start_idx + count]

                    if count > top_k:
                        top_values, top_idx = torch.topk(sample_values, top_k)
                        top_tokens = sample_tokens[top_idx]
                        token_strs = self.tokenizer.convert_ids_to_tokens(top_tokens.tolist())
                        results.append(list(zip(token_strs, top_values.tolist())))
                    else:
                        # Sort values and indices
                        sorted_indices = torch.argsort(sample_values, descending=True)
                        sample_values, sample_tokens = sample_values[sorted_indices], sample_tokens[sorted_indices]
                        token_strs = self.tokenizer.convert_ids_to_tokens(sample_tokens.tolist())
                        results.append(list(zip(token_strs, sample_values.tolist())))

                    start_idx += count

                return results
            else:
                # Process all tokens for each sample
                results = []
                start_idx = 0
                for i, count in enumerate(sample_counts):
                    if count == 0:
                        results.append([])
                        continue

                    sample_values = values[start_idx : start_idx + count]
                    sample_tokens = token_indices[start_idx : start_idx + count]
                    # Sort values and indices
                    sorted_indices = torch.argsort(sample_values, descending=True)
                    sample_values, sample_tokens = sample_values[sorted_indices], sample_tokens[sorted_indices]
                    token_strs = self.tokenizer.convert_ids_to_tokens(sample_tokens.tolist())
                    results.append(list(zip(token_strs, sample_values.tolist())))

                    start_idx += count

                # Fill in empty results for samples with no tokens
                if len(results) < embeddings.size(0):
                    results.extend([[]] * (embeddings.size(0) - len(results)))

                return results

        else:
            raise ValueError("Input tensor must be 1D or 2D.")
