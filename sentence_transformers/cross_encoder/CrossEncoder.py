from __future__ import annotations

import logging
import math
import os
import queue
import tempfile
import traceback
from collections.abc import Callable
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import torch
import torch.multiprocessing as mp
from huggingface_hub import HfApi
from packaging import version
from torch import nn
from tqdm.autonotebook import trange
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    is_torch_npu_available,
)
from transformers.utils import PushToHubMixin
from typing_extensions import deprecated

from sentence_transformers import __version__
from sentence_transformers.backend import load_onnx_model, load_openvino_model
from sentence_transformers.cross_encoder.fit_mixin import FitMixin
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData, generate_model_card
from sentence_transformers.cross_encoder.util import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from sentence_transformers.util import fullname, get_device_name, import_from_string, load_file_path

logger = logging.getLogger(__name__)


def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper


class CrossEncoder(nn.Module, PushToHubMixin, FitMixin):
    """
    A CrossEncoder takes exactly two sentences / texts as input and either predicts
    a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
    on a scale of 0 ... 1.

    It does not yield a sentence embedding and does not work for individual sentences.

    Args:
        model_name_or_path (str): A model name from Hugging Face Hub that can be loaded with AutoModel, or a path to a local
            model. We provide several pre-trained CrossEncoder models that can be used for common tasks.
        num_labels (int, optional): Number of labels of the classifier. If 1, the CrossEncoder is a regression model that
            outputs a continuous score 0...1. If > 1, it output several scores that can be soft-maxed to get
            probability scores for the different classes. Defaults to None.
        max_length (int, optional): Max length for input sequences. Longer sequences will be truncated. If None, max
            length of the model will be used. Defaults to None.
        activation_fn (Callable, optional): Callable (like nn.Sigmoid) about the default activation function that
            should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1,
            else nn.Identity(). Defaults to None.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        cache_folder (`str`, `Path`, optional): Path to the folder where cached files are stored.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine. Defaults to False.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face. Defaults to None.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
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

            See the `AutoModelForSequenceClassification.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details. For example, you can set ``classifier_dropout`` via this parameter.
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/cross_encoder/usage/efficiency.html for benchmarking information
            on the different backends.
    """

    @cross_encoder_init_args_decorator
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int | None = None,
        max_length: int | None = None,
        activation_fn: Callable | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        config_kwargs: dict | None = None,
        model_card_data: CrossEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        super().__init__()
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}
        self.model_card_data = model_card_data or CrossEncoderModelCardData(local_files_only=local_files_only)
        self.trust_remote_code = trust_remote_code
        self._model_card_text = None
        self.backend = backend

        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **config_kwargs,
        )
        if hasattr(config, "sentence_transformers") and "version" in config.sentence_transformers:
            model_version = config.sentence_transformers["version"]
            if version.parse(model_version) > version.parse(__version__):
                logger.warning(
                    f"You are trying to use a model that was created with Sentence Transformers version {model_version}, "
                    f"but you're currently using version {__version__}. This might cause unexpected behavior or errors. "
                    "In that case, try to update to the latest version."
                )

        classifier_trained = False
        if config.architectures is not None:
            classifier_trained = any([arch.endswith("ForSequenceClassification") for arch in config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            config.num_labels = num_labels
        self._load_model(
            model_name_or_path,
            config=config,
            backend=backend,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **model_kwargs,
        )

        if "model_max_length" not in tokenizer_kwargs and max_length is not None:
            tokenizer_kwargs["model_max_length"] = max_length
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            **tokenizer_kwargs,
        )
        if "model_max_length" not in tokenizer_kwargs and hasattr(self.config, "max_position_embeddings"):
            self.tokenizer.model_max_length = min(self.tokenizer.model_max_length, self.config.max_position_embeddings)

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path,
            "README.md",
            token=model_kwargs.get("token", None),
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

        self.set_activation_fn(activation_fn)

        if device is None:
            device = get_device_name()
            logger.info(f"Use pytorch device: {device}")
        self.to(device)

        # Pass the model to the model card data for later use in generating a model card upon saving this model
        self.model_card_data.register_model(self)
        self.model_card_data.set_base_model(model_name_or_path, revision=revision)

    def _load_model(
        self,
        model_name_or_path: str,
        config: PretrainedConfig,
        backend: str,
        **model_kwargs,
    ) -> None:
        if backend == "torch":
            self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=config,
                **model_kwargs,
            )
        elif backend == "onnx":
            self.model = load_onnx_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name="sequence-classification",
                **model_kwargs,
            )
        elif backend == "openvino":
            self.model = load_openvino_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name="sequence-classification",
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def get_backend(self) -> Literal["torch", "onnx", "openvino"]:
        """Return the backend used for inference, which can be one of "torch", "onnx", or "openvino".

        Returns:
            str: The backend used for inference.
        """
        return self.backend

    def start_multi_process_pool(
        self, target_devices: list[str] | None = None
    ) -> dict[Literal["input", "output", "processes"], Any]:
        """
        Starts a multi-process pool to process the prediction with several independent processes
        via :meth:`CrossEncoder.predict <sentence_transformers.cross_encoder.CrossEncoder.predict>`.

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

    def _multi_process(
        self,
        sentence_pairs: list[tuple[str, str]] | list[list[str]],
        show_progress_bar: bool | None = True,
        input_was_singular: bool = False,
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
                chunk_size = min(math.ceil(len(sentence_pairs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)  # Ensure at least 1

            input_queue: torch.multiprocessing.Queue = pool["input"]
            output_queue: torch.multiprocessing.Queue = pool["output"]

            # Send inputs to the input queue in chunks
            chunk_id = -1  # We default to -1 to handle empty input gracefully
            for chunk_id, chunk_start in enumerate(range(0, len(sentence_pairs), chunk_size)):
                chunk = sentence_pairs[chunk_start : chunk_start + chunk_size]
                input_queue.put([chunk_id, chunk, predict_kwargs])

            # Collect results from output queue
            output_list = sorted(
                [output_queue.get() for _ in trange(chunk_id + 1, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],  # Sort by chunk_id
            )

            # Handle singular input case -> return the first (only) result directly
            if input_was_singular and output_list:
                return output_list[0][1][0] if len(output_list[0][1]) > 0 else output_list[0][1]

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
        Internal working process to predict sentence pairs in a multi-process setup.

        """
        while True:
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

    def set_activation_fn(self, activation_fn: Callable | None, set_default: bool = True) -> None:
        if activation_fn is not None:
            self.activation_fn = activation_fn
        else:
            self.activation_fn = self.get_default_activation_fn()

        if set_default:
            self.set_config_value("activation_fn", fullname(self.activation_fn))

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
            if self.trust_remote_code or activation_fn_path.startswith("torch."):
                return import_from_string(activation_fn_path)()
            logger.warning(
                f"Activation function path '{activation_fn_path}' is not trusted, using default activation function instead. "
                "Please load the CrossEncoder with `trust_remote_code=True` to allow loading custom activation "
                "functions via the configuration."
            )

        if self.config.num_labels == 1:
            return nn.Sigmoid()
        return nn.Identity()

    def set_config_value(self, key: str, value) -> None:
        """
        Set a value in the underlying model's config.

        Args:
            key (str): The key to set.
            value: The value to set.
        """
        try:
            if not hasattr(self.config, "sentence_transformers"):
                self.config.sentence_transformers = {}
            self.config.sentence_transformers[key] = value
        except Exception as e:
            logger.warning(f"Was not able to add '{key}' to the config: {str(e)}")

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def num_labels(self) -> int:
        return self.config.num_labels

    @property
    def max_length(self) -> int:
        return self.tokenizer.model_max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        self.tokenizer.model_max_length = value

    @property
    @deprecated(
        "The `default_activation_function` property was renamed and is now deprecated. "
        "Please use `activation_fn` instead."
    )
    def default_activation_function(self) -> Callable:
        return self.activation_fn

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @overload
    def predict(
        self,
        sentences: tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[True] = True,
        convert_to_tensor: Literal[False] = False,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: bool = ...,
        convert_to_tensor: Literal[True] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor: ...

    @overload
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        activation_fn: Callable | None = ...,
        apply_softmax: bool | None = ...,
        convert_to_numpy: Literal[False] = ...,
        convert_to_tensor: Literal[False] = ...,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> list[torch.Tensor]: ...

    @torch.inference_mode()
    @cross_encoder_predict_rank_args_decorator
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
    ) -> list[torch.Tensor] | np.ndarray | torch.Tensor:
        """
        Performs predictions with the CrossEncoder on the given sentence pairs.

        Args:
            sentences (Union[List[Tuple[str, str]], Tuple[str, str]]): A list of sentence pairs [(Sent1, Sent2), (Sent3, Sent4)]
                or one sentence pair (Sent1, Sent2).
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
            show_progress_bar (bool, optional): Output progress bar. Defaults to None.
            activation_fn (callable, optional): Activation function applied on the logits output of the CrossEncoder.
                If None, the ``model.activation_fn`` will be used, which defaults to :class:`torch.nn.Sigmoid` if num_labels=1, else
                :class:`torch.nn.Identity`. Defaults to None.
            convert_to_numpy (bool, optional): Convert the output to a numpy matrix. Defaults to True.
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
            Union[List[torch.Tensor], np.ndarray, torch.Tensor]: Predictions for the passed sentence pairs.
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
        # Cast an individual pair to a list with length 1
        input_was_singular = False
        if sentences and isinstance(sentences, (list, tuple)) and isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_singular = True

        # If pool or a list of devices is provided, use multi-process prediction
        if pool is not None or (isinstance(device, list) and len(device) > 0):
            return self._multi_process(
                sentence_pairs=sentences,
                # Utility and post-processing parameters
                show_progress_bar=show_progress_bar,
                input_was_singular=input_was_singular,
                # Multi-process encoding parameters
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                # Prediction parameters
                batch_size=batch_size,
                activation_fn=activation_fn,
                apply_softmax=apply_softmax,
                convert_to_numpy=convert_to_numpy,
                convert_to_tensor=convert_to_tensor,
            )

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if activation_fn is not None:
            self.set_activation_fn(activation_fn, set_default=False)

        # Here, device is either a single device string (e.g., "cuda:0", "cpu") for single-process encoding or None
        if device is None:
            device = self.model.device

        self.to(device)

        pred_scores = []
        self.eval()
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = sentences[start_index : start_index + batch_size]
            features = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            features.to(device)
            model_predictions = self.model(**features, return_dict=True)
            logits = self.activation_fn(model_predictions.logits)

            if apply_softmax and logits.ndim > 1:
                logits = torch.nn.functional.softmax(logits, dim=1)
            pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            if len(pred_scores):
                pred_scores = torch.stack(pred_scores)
            else:
                pred_scores = torch.tensor([], device=device)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().float().numpy() for score in pred_scores])

        if input_was_singular:
            pred_scores = pred_scores[0]

        return pred_scores

    @cross_encoder_predict_rank_args_decorator
    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        return_documents: bool = False,
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

        Args:
            query (str): A single query.
            documents (List[str]): A list of documents.
            top_k (Optional[int], optional): Return the top-k documents. If None, all documents are returned. Defaults to None.
            return_documents (bool, optional): If True, also returns the documents. If False, only returns the indices and scores. Defaults to False.
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
        query_doc_pairs = [[query, doc] for doc in documents]
        scores = self.predict(
            sentences=query_doc_pairs,
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
            results.append({"corpus_id": i, "score": score})
            if return_documents:
                results[-1].update({"text": documents[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Saves the model and tokenizer to path; identical to `save_pretrained`
        """
        if path is None:
            return

        logger.info(f"Save model to {path}")
        self.set_config_value("version", __version__)
        self.model.save_pretrained(path, safe_serialization=safe_serialization, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)
        self._create_model_card(path)

    def save_pretrained(self, path: str, *, safe_serialization: bool = True, **kwargs) -> None:
        """
        Save the model and tokenizer to the specified path.

        Args:
            path (str): Directory where the model should be saved
            safe_serialization (bool, optional): Whether to save using `safetensors` or the traditional
                PyTorch way. Defaults to True.
            **kwargs: Additional arguments passed to the underlying save methods of the model and tokenizer.

        Returns:
            None
        """
        return self.save(path, safe_serialization=safe_serialization, **kwargs)

    def _create_model_card(self, path: str) -> None:
        """
        Create an automatic model and stores it in the specified path. If no training was done and the loaded model
        was a CrossEncoder model already, then its model card is reused.

        Args:
            path (str): The path where the model card will be stored.

        Returns:
            None
        """
        # If we loaded a model from the Hub, and no training was done, then
        # we don't generate a new model card, but reuse the old one instead.
        if self._model_card_text and "generated_from_trainer" not in self.model_card_data.tags:
            model_card = self._model_card_text
            if self.model_card_data.model_id:
                # If the original model card was saved without a model_id, we replace the model_id with the new model_id
                model_card = model_card.replace(
                    'model = CrossEncoder("cross_encoder_model_id"',
                    f'model = CrossEncoder("{self.model_card_data.model_id}"',
                )
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

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        safe_serialization: bool = True,
        commit_message: str | None = None,
        exist_ok: bool = False,
        revision: str | None = None,
        create_pr: bool = False,
        tags: list[str] | None = None,
    ) -> str:
        """
        Upload the CrossEncoder model to the Hugging Face Hub.

        Example:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
                model.push_to_hub("username/my-crossencoder-model")
                # => "https://huggingface.co/username/my-crossencoder-model"

        Args:
            repo_id (str): The name of the repository on the Hugging Face Hub, e.g. "username/repo_name",
                "organization/repo_name" or just "repo_name".
            token (str, optional): The authentication token to use for the Hugging Face Hub API.
                If not provided, will use the token stored via the Hugging Face CLI.
            private (bool, optional): Whether to create a private repository. If not specified,
                the repository will be public.
            safe_serialization (bool, optional): Whether or not to convert the model weights in safetensors
                format for safer serialization. Defaults to True.
            commit_message (str, optional): The commit message to use for the push. Defaults to "Add new CrossEncoder model".
            exist_ok (bool, optional): If True, do not raise an error if the repository already exists.
                Ignored if ``create_pr=True``. Defaults to False.
            revision (str, optional): The git branch to commit to. Defaults to the head of the 'main' branch.
            create_pr (bool, optional): Whether to create a Pull Request with the upload or directly commit. Defaults to False.
            tags (list[str], optional): A list of tags to add to the model card. Defaults to None.

        Returns:
            str: URL of the commit or pull request (if create_pr=True)
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
        if tags is not None:
            self.model_card_data.add_tags(tags)

        if revision is not None:
            api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

        if commit_message is None:
            commit_message = "Add new CrossEncoder model"
        commit_description = ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(
                tmp_dir,
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

    @property
    def transformers_model(self) -> PreTrainedModel | None:
        """
        Property to get the underlying transformers PreTrainedModel instance.

        Returns:
            PreTrainedModel or None: The underlying transformers model or None if not found.

        Example:
            ::

                from sentence_transformers import CrossEncoder

                model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

                # You can now access the underlying transformers model
                transformers_model = model.transformers_model
                print(type(transformers_model))
                # => <class 'transformers.models.bert.modeling_bert.BertForSequenceClassification'>
        """
        # This property simply points to self.model, it exists primarily to have the same interface
        # as SentenceTransformer and SparseEncoder models.
        return self.model

    @property
    def _target_device(self) -> torch.device:
        logger.warning(
            "`CrossEncoder._target_device` has been removed, please use `CrossEncoder.device` instead.",
        )
        return self.device

    @_target_device.setter
    def _target_device(self, device: int | str | torch.device | None = None) -> None:
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        # Propagate the gradient checkpointing to the transformer model
        return self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
