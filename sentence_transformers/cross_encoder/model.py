from __future__ import annotations

import logging
import math
import queue
from collections import OrderedDict
from collections.abc import Callable
from multiprocessing import Queue
from typing import Any, Literal, overload

import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import trange
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, is_datasets_available
from transformers.utils import logging as transformers_logging
from typing_extensions import deprecated

from sentence_transformers.base.modality_types import PairableInput, PairInput
from sentence_transformers.base.model import BaseModel
from sentence_transformers.base.modules import Transformer
from sentence_transformers.cross_encoder.fit_mixin import FitMixin
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData
from sentence_transformers.cross_encoder.modules.logit_score import LogitScore
from sentence_transformers.util import batch_to_device, fullname, import_from_string
from sentence_transformers.util.decorators import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)

# NOTE: transformers wraps the regular logging module for e.g. warning_once
logger = transformers_logging.get_logger(__name__)


class CrossEncoder(BaseModel, FitMixin):
    """
    Loads or creates a CrossEncoder model that takes a sentence pair as input and outputs a score or label.

    A CrossEncoder does not produce sentence embeddings. Instead, it processes both sentences jointly through the
    transformer and outputs a score (regression) or class probabilities (classification). This makes it more
    accurate for pairwise tasks like reranking or semantic textual similarity, but it cannot pre-compute embeddings
    for individual sentences.

    Args:
        model_name_or_path (str, optional): If a filepath on disk, loads the model from that path. Otherwise, tries
            to download a pre-trained CrossEncoder model. If that fails, tries to construct a model from the
            Hugging Face Hub with that name. Defaults to None.
        modules (list[nn.Module], optional): A list of torch modules that are called sequentially. Can be used to
            create custom CrossEncoder models from scratch. Defaults to None.
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
            documentation for more details. For example, you can set ``classifier_dropout`` or ``num_labels``
            via this parameter. Defaults to None.
        model_card_data (:class:`~sentence_transformers.cross_encoder.model_card.CrossEncoderModelCardData`, optional):
            A model card data object that contains information about the model. Used to generate a model card
            when saving the model. If not set, a default model card data object is created. Defaults to None.
        backend (str, optional): The backend to use for inference. Can be ``"torch"`` (default), ``"onnx"``,
            or ``"openvino"``. Defaults to ``"torch"``.
        num_labels (int, optional): Number of labels of the classifier. If 1, the CrossEncoder is a regression
            model that outputs a continuous score. If > 1, it outputs several scores that can be soft-maxed to
            get probability scores for the different classes. Defaults to None.
        max_length (int, optional): Max length for input sequences. Longer sequences will be truncated. If None,
            the max length of the model will be used. Defaults to None.
        activation_fn (Callable, optional): Activation function applied on top of the model's logits during
            :meth:`predict`. If None, ``nn.Sigmoid()`` is used when ``num_labels=1``, else ``nn.Identity()``.
            Defaults to None.

    Example:
        ::

            from sentence_transformers import CrossEncoder

            # Load a pre-trained CrossEncoder model
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

            # Predict scores for sentence pairs
            pairs = [
                ("How many people live in Berlin?", "Berlin had a population of 3,520,031 in 2019."),
                ("How many people live in Berlin?", "Berlin is well known for its museums."),
            ]
            scores = model.predict(pairs)
            print(scores)
            # [8.607  1.133]

            # Rank documents by relevance to a query
            results = model.rank(
                "How many people live in Berlin?",
                ["Berlin had a population of 3,520,031 in 2019.", "Berlin is well known for its museums."],
            )
            print(results)
            # [{'corpus_id': 0, 'score': 8.607317}, {'corpus_id': 1, 'score': 1.1329174}]
    """

    model_card_data_class = CrossEncoderModelCardData
    default_huggingface_organization: str | None = "cross-encoder"
    _model_card_model_id_placeholder = "cross_encoder_model_id"

    @cross_encoder_init_args_decorator
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
        model_kwargs: dict | None = None,
        processor_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
        model_card_data: CrossEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        # CrossEncoder-specific args
        num_labels: int | None = None,
        max_length: int | None = None,
        activation_fn: Callable | None = None,
    ) -> None:
        # Set before super().__init__() so _parse_model_config can check these
        self.activation_fn = None

        if num_labels is not None:
            if config_kwargs is None:
                config_kwargs = {}
            config_kwargs["num_labels"] = num_labels

        if max_length is not None:
            if processor_kwargs is None:
                processor_kwargs = {}
            processor_kwargs["model_max_length"] = max_length

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
        self.model_card_data: CrossEncoderModelCardData

        # If an activation function is provided, use it. Otherwise, load the default one/from backwards compatibility
        # if it wasn't set during super().__init__()
        if activation_fn is not None:
            self.activation_fn = activation_fn
        elif self.activation_fn is None:
            self.activation_fn = self.get_default_activation_fn()

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
        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = {**shared_kwargs} if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        processor_kwargs = {**shared_kwargs} if processor_kwargs is None else {**shared_kwargs, **processor_kwargs}
        config_kwargs = {**shared_kwargs} if config_kwargs is None else {**shared_kwargs, **config_kwargs}

        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)

        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            **config_kwargs,
        )
        if (
            hasattr(config, "architectures")
            and config.architectures is not None
            and config.architectures[0].endswith("ForCausalLM")
        ):
            transformer_model = Transformer(
                model_name_or_path,
                transformer_task="text-generation",
                cache_dir=cache_folder,
                model_kwargs=model_kwargs,
                processor_kwargs=processor_kwargs,
                config_kwargs=config_kwargs,
                backend=self.backend,
            )
            true_token_id = transformer_model.tokenizer.convert_tokens_to_ids("yes")
            false_token_id = transformer_model.tokenizer.convert_tokens_to_ids("no")
            if true_token_id is None or false_token_id is None:
                raise ValueError(
                    "The tokenizer does not have 'yes' and/or 'no' tokens, which are used as the "
                    "default true/false tokens for the LogitScore post-processing module. Please "
                    "provide custom modules with your desired LogitScore configuration, or use a "
                    "model with a tokenizer that supports these tokens."
                )
            post_processing = LogitScore(
                true_token_id=true_token_id,
                false_token_id=false_token_id,
            )
            return [transformer_model, post_processing], {}

        # Otherwise, assume sequence-classification
        transformer_model = Transformer(
            model_name_or_path,
            transformer_task="sequence-classification",
            cache_dir=cache_folder,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            backend=self.backend,
        )
        return [transformer_model], {}

    def _multi_process(
        self,
        inputs: list[PairInput],
        show_progress_bar: bool | None = True,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **predict_kwargs,
    ):
        convert_to_tensor = predict_kwargs.get("convert_to_tensor", False)
        convert_to_numpy = predict_kwargs.get("convert_to_numpy", True)
        predict_kwargs["show_progress_bar"] = False

        created_pool = False
        if pool is None and isinstance(device, list) and len(device) > 0:
            pool = self.start_multi_process_pool(device)
            created_pool = True

        # Create a pool if is not provided, but a list of devices is
        try:
            # Determine chunk size if not provided. As a default, aim for 10 chunks per process, with a maximum of 5000 sentences per chunk.
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)  # Ensure at least 1

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            # Send inputs to the input queue in chunks
            chunk_id = -1  # We default to -1 to handle empty input gracefully
            for chunk_id, chunk_start in enumerate(range(0, len(inputs), chunk_size)):
                chunk = inputs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, predict_kwargs])

            # Collect results from output queue
            output_list = sorted(
                [output_queue.get() for _ in trange(chunk_id + 1, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],  # Sort by chunk_id
            )

            # Handle the various output formats: torch tensors, numpy arrays, or
            # list of dictionaries, also when empty.
            scores = [output[1] for output in output_list]

            # Check for errors in results
            if any(len(output) > 2 and output[2] is not None for output in output_list):
                # Error occurred in worker
                error_output = next(output for output in output_list if len(output) > 2 and output[2])
                raise RuntimeError(f"Error in worker process: {error_output[2]}")

            if scores:
                if isinstance(scores[0], torch.Tensor):
                    scores = torch.cat(scores)
                elif isinstance(scores[0], np.ndarray):
                    scores = np.concatenate(scores, axis=0)
                elif isinstance(scores[0], list):
                    scores = sum(scores, [])
                else:
                    scores = sum(scores, [])

            elif convert_to_tensor:
                scores = torch.tensor([], device=self.model.device)
            elif convert_to_numpy:
                scores = np.array([])
            else:
                scores = []
            return scores

        finally:
            # Clean up the pool if we created it
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _multi_process_worker(
        target_device: str,
        model: CrossEncoder,
        input_queue: Queue,
        results_queue: Queue,
    ) -> None:
        """
        Internal working process to predict input pairs in a multi-process setup.

        """
        while True:
            chunk_id = None
            try:
                chunk_id, sentence_pairs, kwargs = input_queue.get()
                scores = model.predict(sentence_pairs, device=target_device, **kwargs)

                # If multi-process scores are not on CPUs, move them to CPU, so they can all be concatenated later
                if isinstance(scores, torch.Tensor) and scores.device.type != "cpu":
                    scores = scores.cpu()
                elif isinstance(scores, np.ndarray):
                    scores = np.asarray(scores)
                elif isinstance(scores, list):
                    scores = [
                        score.cpu() if isinstance(score, torch.Tensor) and score.device.type != "cpu" else score
                        for score in scores
                    ]
                results_queue.put([chunk_id, scores])

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error in worker process on {target_device}: {e}")
                try:
                    results_queue.put([chunk_id, None, str(e)])
                except Exception:
                    pass
                break

    def _resolve_activation_fn(self, activation_fn_path: str) -> Callable | None:
        """Instantiate an activation function from a dotted path string, respecting trust_remote_code."""
        if self.trust_remote_code or activation_fn_path.startswith("torch."):
            return import_from_string(activation_fn_path)()
        logger.warning(
            f"Activation function path '{activation_fn_path}' is not trusted, using default activation function instead. "
            "Please load the CrossEncoder with `trust_remote_code=True` to allow loading custom activation "
            "functions via the configuration."
        )
        return None

    def get_default_activation_fn(self) -> Callable:
        activation_fn_path = None
        if hasattr(self.config, "sentence_transformers") and "activation_fn" in self.config.sentence_transformers:
            activation_fn_path = self.config.sentence_transformers["activation_fn"]

        # Backwards compatibility with <v4.0: we stored the activation_fn under 'sbert_ce_default_activation_function'
        elif (
            hasattr(self.config, "sbert_ce_default_activation_function")
            and self.config.sbert_ce_default_activation_function is not None
        ):
            activation_fn_path = self.config.sbert_ce_default_activation_function
            del self.config.sbert_ce_default_activation_function

        if activation_fn_path is not None:
            resolved = self._resolve_activation_fn(activation_fn_path)
            if resolved is not None:
                return resolved

        if self.config.num_labels == 1:
            return nn.Sigmoid()
        return nn.Identity()

    @property
    def config(self) -> PretrainedConfig:
        return self[0].model.config

    @property
    def model(self) -> PreTrainedModel:
        return self[0].model

    @property
    def num_labels(self) -> int:
        for module in reversed(self):
            if isinstance(module, Transformer):
                return module.model.config.num_labels
            if isinstance(module, LogitScore):
                return 1
        # Default to 1, not commonly reached
        return 1

    def __setattr__(self, name: str, value: Any) -> None:
        # We don't want activation_fn to be registered as a module, instead we want it as a normal attribute
        # This avoids issues with saving/loading the model
        if name == "activation_fn":
            return super(torch.nn.Module, self).__setattr__(name, value)
        return super().__setattr__(name, value)

    @property
    @deprecated("The `max_length` property was renamed and is now deprecated. Please use `max_seq_length` instead.")
    def max_length(self) -> int:
        return self.max_seq_length

    @max_length.setter
    @deprecated("The `max_length` property was renamed and is now deprecated. Please use `max_seq_length` instead.")
    def max_length(self, value: int) -> None:
        self.max_seq_length = value

    @property
    @deprecated(
        "The `default_activation_function` property was renamed and is now deprecated. "
        "Please use `activation_fn` instead."
    )
    def default_activation_function(self) -> Callable:
        return self.activation_fn

    @overload
    def predict(
        self,
        inputs: PairInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        inputs: list[PairInput] | PairInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[True] = True,
        convert_to_tensor: Literal[False] = False,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        inputs: list[PairInput] | PairInput,
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: Literal[True] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        inputs: list[PairInput],
        prompt_name: str | None = ...,
        prompt: str | None = ...,
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[torch.Tensor]: ...

    @torch.inference_mode()
    @cross_encoder_predict_rank_args_decorator
    def predict(
        self,
        inputs: list[PairInput] | PairInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Performs predictions with the CrossEncoder on the given input pairs.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        Args:
            inputs (Union[List[PairInput], PairInput]): A list of input pairs or one input pair, where each element
                can be a string, image, or multimodal dict.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding.
            prompt (Optional[str], optional): The prompt to use for encoding.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn (callable, optional): Activation function applied on the logits output of the CrossEncoder.
                If None, the ``model.activation_fn`` will be used, which defaults to :class:`torch.nn.Sigmoid` if num_labels=1, else
                :class:`torch.nn.Identity`. Defaults to None.
            apply_softmax (bool, optional): If set to True and `model.num_labels > 1`, applies softmax on the logits
                output such that for each sample, the scores of each class sum to 1. Defaults to False.
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, output
                a list of PyTorch tensors. Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (Union[str, List[str]], optional): Device(s) to use for computation. Can be a single device string
                (e.g., "cuda:0", "cpu") or a list of devices (e.g., ["cuda:0", "cuda:1"]). If a list is provided,
                multiprocessing will be used automatically. Defaults to None.
            pool (Dict[str, Any], optional): A pool of workers created with :meth:`start_multi_process_pool`. If provided,
                multiprocessing will be used. If None and ``device`` is a list, a pool will be created automatically.
                Defaults to None.
            chunk_size (int, optional): Size of chunks for multiprocessing. If None, a sensible default is calculated.
                Only used when ``pool`` is not None or ``device`` is a list. Defaults to None.

        Returns:
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Predictions for the passed input pairs.
            The return type depends on the ``convert_to_numpy`` and ``convert_to_tensor`` parameters.
            If ``convert_to_tensor`` is True, the output will be a :class:`torch.Tensor`.
            If ``convert_to_numpy`` is True, the output will be a :class:`numpy.ndarray`.
            Otherwise, the output will be a list of :class:`torch.Tensor` values.

        Examples:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/stsb-roberta-base")
                sentences = [["I love cats", "Cats are amazing"], ["I prefer dogs", "Dogs are loyal"]]
                model.predict(sentences)
                # => array([0.6912767, 0.4303499], dtype=float32)

                # Using multiprocessing with automatic pool
                scores = model.predict(sentences, device=["cuda:0", "cuda:1"])

                # Using multiprocessing with manual pool
                pool = model.start_multi_process_pool()
                scores = model.predict(sentences, pool=pool)
                model.stop_multi_process_pool(pool)
        """
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        # Cast an individual pair to a list with length 1
        is_singular_input = self.is_singular_input(inputs)
        if is_singular_input:
            inputs = [inputs]
        elif not isinstance(inputs, list):
            # Materialize e.g. datasets.Column to avoid slow Arrow deserialization on each index
            inputs = list(inputs)

        # If pool or a list of devices is provided, use multi-process prediction
        if pool is not None or (isinstance(device, list) and len(device) > 0):
            pred_scores = self._multi_process(
                inputs=inputs,
                # Utility and post-processing parameters
                show_progress_bar=show_progress_bar,
                # Multi-process encoding parameters
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                # Prediction parameters
                prompt=prompt,
                prompt_name=prompt_name,
                batch_size=batch_size,
                activation_fn=activation_fn,
                apply_softmax=apply_softmax,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
                **kwargs,
            )
            if is_singular_input:
                pred_scores = pred_scores[0]
            return pred_scores

        prompt = self._resolve_prompt(prompt, prompt_name)

        # Here, device is either a single device string (e.g., "cuda:0", "cpu") for single-process encoding or None
        if device is None:
            device = self.model.device

        self.to(device)

        self.eval()
        activation_fn = activation_fn or self.activation_fn
        num_labels = self.num_labels

        pred_scores = []
        length_sorted_idx = np.argsort([-self._input_length(pair) for pair in inputs])
        if self._can_flatten_inputs():
            length_sorted_idx = self._interleave_sorted_indices(length_sorted_idx)
        inputs_sorted = [inputs[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(inputs_sorted), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = inputs_sorted[start_index : start_index + batch_size]
            features = self.preprocess(batch, prompt=prompt, **kwargs)
            features = batch_to_device(features, device)
            out_features = self.forward(features, **kwargs)
            scores = out_features["scores"]

            if activation_fn is not None:
                scores = activation_fn(scores)

            if apply_softmax and scores.ndim > 1:
                scores = torch.nn.functional.softmax(scores, dim=1)

            # Squeeze [batch_size, 1] -> [batch_size] for single-label models
            if num_labels == 1 and scores.ndim > 1:
                scores = scores.squeeze(-1)

            pred_scores.extend(scores)

        pred_scores = [pred_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(pred_scores):
                pred_scores = torch.stack(pred_scores)
            else:
                pred_scores = torch.tensor([], device=device)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if is_singular_input:
            pred_scores = pred_scores[0]

        return pred_scores

    @cross_encoder_predict_rank_args_decorator
    def rank(
        self,
        query: PairableInput,
        documents: list[PairableInput],
        top_k: int | None = None,
        return_documents: bool = False,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> list[dict[Literal["corpus_id", "score", "text"], int | float | str]]:
        """
        Performs ranking with the CrossEncoder on the given query and documents. Returns a sorted list with the document indices and scores.

        .. tip::

            Adjusting ``batch_size`` can significantly improve processing speed. The optimal value depends on your
            hardware, model size, precision, and input length. Benchmark a few batch sizes on a small subset of your
            data to find the best value.

        Args:
            query (PairableInput): A single query, e.g. a string, image, or multimodal dict.
            documents (List[PairableInput]): A list of documents, e.g. strings, images, or multimodal dicts.
            top_k (Optional[int], optional): Return the top-k documents. If None, all documents are returned. Defaults to None.
            return_documents (bool, optional): If True, also returns the documents. If False, only returns the indices and scores. Defaults to False.
            prompt_name (Optional[str], optional): The name of the prompt to use for encoding.
            prompt (Optional[str], optional): The prompt to use for encoding.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn ([type], optional): Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
            apply_softmax (bool, optional): If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output. Defaults to False.
            convert_to_tensor (bool, optional): Convert the output to a tensor. Defaults to False.
            device (Union[str, List[str]], optional): Device(s) to use for computation. Can be a single device string
                (e.g., "cuda:0", "cpu") or a list of devices (e.g., ["cuda:0", "cuda:1"]). If a list is provided,
                multiprocessing will be used automatically. Defaults to None.
            pool (Dict[str, Any], optional): A pool of workers created with :meth:`start_multi_process_pool`. If provided,
                multiprocessing will be used. If None and ``device`` is a list, a pool will be created automatically.
                Defaults to None.
            chunk_size (int, optional): Size of chunks for multiprocessing. If None, a sensible default is calculated.
                Only used when ``pool`` is not None or ``device`` is a list. Defaults to None.

        Returns:
            List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]: A sorted list with the "corpus_id", "score", and optionally "text" of the documents.

        Example:
            ::

                from sentence_transformers import CrossEncoder
                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

                query = "Who wrote 'To Kill a Mockingbird'?"
                documents = [
                    "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
                    "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
                    "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
                    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
                    "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
                    "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
                ]

                model.rank(query, documents, return_documents=True)

            ::

                [{'corpus_id': 0,
                'score': 10.67858,
                'text': "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature."},
                {'corpus_id': 2,
                'score': 9.761677,
                'text': "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961."},
                {'corpus_id': 1,
                'score': -3.3099542,
                'text': "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil."},
                {'corpus_id': 5,
                'score': -4.8989105,
                'text': "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."},
                {'corpus_id': 4,
                'score': -5.082967,
                'text': "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era."}]
        """
        if self.num_labels != 1:
            raise ValueError(
                "CrossEncoder.rank() only works for models with num_labels=1. "
                "Consider using CrossEncoder.predict() with input pairs instead."
            )
        query_doc_pairs: list[PairInput] = [[query, doc] for doc in documents]
        scores = self.predict(
            inputs=query_doc_pairs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            activation_fn=activation_fn,
            apply_softmax=apply_softmax,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            pool=pool,
            chunk_size=chunk_size,
        )

        results = []
        for i, score in enumerate(scores):
            # TODO v6: convert score to float(score) for cleaner output
            results.append({"corpus_id": i, "score": score})
            if return_documents:
                results[-1].update({"text": documents[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def is_singular_input(self, inputs: PairInput | list[PairInput]) -> bool:
        """
        Check if the input represents a single example or a batch of examples.

        Args:
            inputs: The input to check.
        Returns:
            bool: True if the input is a single example, False if it is a batch.
        """
        list_types = (list, tuple)
        if is_datasets_available():
            try:
                from datasets import Column

                list_types += (Column,)
            except ImportError:
                pass
        return (not isinstance(inputs, list_types)) or (len(inputs) > 0 and not isinstance(inputs[0], list_types))

    def _get_model_config(self) -> dict[str, Any]:
        return super()._get_model_config() | {
            "activation_fn": fullname(self.activation_fn),
        }

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        super()._parse_model_config(model_config)
        if "activation_fn" in model_config:
            activation_fn_path = model_config["activation_fn"]
            if activation_fn_path is not None:
                resolved = self._resolve_activation_fn(activation_fn_path)
                if resolved is not None:
                    self.activation_fn = resolved

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
scores = model.predict([("The weather is lovely today.", "It's so sunny outside!")])
print(scores)

rankings = model.rank("The weather is lovely today.", ["It's so sunny outside!", "He drove to the stadium."])
print(rankings)
```

---
*This PR was auto-generated with \
[`push_to_hub`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.push_to_hub).*
"""
