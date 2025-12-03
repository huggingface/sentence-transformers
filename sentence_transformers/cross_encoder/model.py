from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path

# TODO: OrderedDict from typing or typing_dict? Or collections?
from typing import Any, Literal, overload

import numpy as np
import torch
from torch import nn
from tqdm.autonotebook import trange
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from typing_extensions import deprecated

from sentence_transformers.base.model import BaseModel
from sentence_transformers.base.models import Transformer
from sentence_transformers.cross_encoder.fit_mixin import FitMixin
from sentence_transformers.cross_encoder.model_card import CrossEncoderModelCardData
from sentence_transformers.cross_encoder.models.CausalScoreHead import CausalScoreHead
from sentence_transformers.cross_encoder.util import (
    cross_encoder_init_args_decorator,
    cross_encoder_predict_rank_args_decorator,
)
from sentence_transformers.util import fullname, import_from_string

logger = logging.getLogger(__name__)


def _save_pretrained_wrapper(_save_pretrained_fn: Callable, subfolder: str) -> Callable[..., None]:
    def wrapper(save_directory: str | Path, **kwargs) -> None:
        os.makedirs(Path(save_directory) / subfolder, exist_ok=True)
        return _save_pretrained_fn(Path(save_directory) / subfolder, **kwargs)

    return wrapper


class CrossEncoder(BaseModel, FitMixin):
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

    # TODO: Check backwards incompatibilities with the methods that are now handled by the superclass

    model_card_data_class = CrossEncoderModelCardData
    default_huggingface_organization: str | None = "cross-encoder"

    @cross_encoder_init_args_decorator
    # TODO: Should we remove 'prompts'? They're a tad duplicate if we also have templates
    # But it's possible to have a template with a spot for instructions (prompts), so maybe keep both
    def __init__(
        self,
        model_name_or_path: str,
        *,
        modules: Iterable[nn.Module] | None = None,
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
        # CrossEncoder-specific args
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        num_labels: int | None = None,
        max_length: int | None = None,
        activation_fn: Callable | None = None,
    ) -> None:
        # CrossEncoder-specific attributes
        self.prompts = prompts or {}
        self.default_prompt_name = default_prompt_name
        self.activation_fn = None

        if num_labels is not None:
            if config_kwargs is None:
                config_kwargs = {}
            config_kwargs["num_labels"] = num_labels

        if max_length is not None:
            if tokenizer_kwargs is None:
                tokenizer_kwargs = {}
            tokenizer_kwargs["model_max_length"] = max_length

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
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )

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
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> tuple[OrderedDict[str, nn.Module], OrderedDict[str, Any]]:
        # logger.warning(
        #     f"No CrossEncoder model found with name {model_name_or_path}. Creating a new one with mean pooling."
        # )

        shared_kwargs = {
            "token": token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        model_kwargs = shared_kwargs if model_kwargs is None else {**shared_kwargs, **model_kwargs}
        tokenizer_kwargs = shared_kwargs if tokenizer_kwargs is None else {**shared_kwargs, **tokenizer_kwargs}
        config_kwargs = shared_kwargs if config_kwargs is None else {**shared_kwargs, **config_kwargs}

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
                model_args=model_kwargs,
                tokenizer_args=tokenizer_kwargs,
                config_args=config_kwargs,
                backend=self.backend,
            )
            post_processing = CausalScoreHead(
                true_token_id=transformer_model.tokenizer.convert_tokens_to_ids("yes"),
                false_token_id=transformer_model.tokenizer.convert_tokens_to_ids("no"),
            )
            return [transformer_model, post_processing], {}

        # Otherwise, assume sequence-classification
        transformer_model = Transformer(
            model_name_or_path,
            transformer_task="sequence-classification",
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
            backend=self.backend,
        )
        return [transformer_model], {}

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

    '''
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a value in the underlying model's config under `sentence_transformers`."""
        try:
            if not hasattr(self.config, "sentence_transformers"):
                self.config.sentence_transformers = {}
            self.config.sentence_transformers[key] = value
        except Exception as e:
            logger.warning(f"Was not able to add '{key}' to the config: {str(e)}")
    '''

    @property
    def config(self) -> PretrainedConfig:
        # return self.model.config
        # TODO: This won't work nicely
        return self[0].model.config

    @property
    def model(self) -> PreTrainedModel:
        return self[0].model

    @property
    def num_labels(self) -> int:
        for module in reversed(self):
            if isinstance(module, Transformer):
                return module.model.config.num_labels
            if isinstance(module, CausalScoreHead):
                return module.num_labels
        # Default to 1, not commonly reached
        return 1

    def __setattr__(self, name: str, value: Any) -> None:
        # We don't want activation_fn to be registered as a module, instead we want it as a normal attribute
        # This avoids issues with saving/loading the model
        if name == "activation_fn":
            return super(torch.nn.Module, self).__setattr__(name, value)
        return super().__setattr__(name, value)

    '''
    def forward(self, input: dict[str, torch.Tensor] = None, **kwargs) -> dict[str, torch.Tensor]:
        """Run a forward pass through all modules in the model.

        The preferred calling pattern is ``model(input)`` where ``input`` is a dictionary of tensors.
        For backwards compatibility, calling ``model(**inputs)`` is still supported. In that case,
        the keyword arguments are collected into a single input dictionary, akin to ``model(inputs)``.
        """
        if input is None and kwargs:
            input = kwargs
        return super().forward(input)
    '''

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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> list[torch.Tensor]: ...

    @torch.inference_mode()
    @cross_encoder_predict_rank_args_decorator
    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]] | tuple[str, str] | list[str],
        batch_size: int = 32,
        prompt: str | None = None,
        prompt_name: str | None = None,
        show_progress_bar: bool | None = None,
        activation_fn: Callable | None = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs,
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
        """
        # Cast an individual pair to a list with length 1
        input_was_singular = False
        if sentences and isinstance(sentences, (list, tuple)) and isinstance(sentences[0], str):
            sentences = [sentences]
            input_was_singular = True

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

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

        pred_scores = []
        self.eval()
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            batch = sentences[start_index : start_index + batch_size]
            # TODO: Let's also allow a prompt
            # TODO: This should probably call self.tokenize() instead
            # '''
            if hasattr(self, "template") and self.template is not None:
                template_parts = self.template.split("{document}")  # TODO: What if the prompt is after the document?
                main_template = template_parts[0] + "{document}"
                suffix = template_parts[1]

                tokenized_suffix = self._get_tokenized_suffix(suffix)
                num_suffix_tokens = len(tokenized_suffix["input_ids"])
                tokenized = self.tokenizer(
                    [main_template.format(query=query, document=document, prompt=prompt) for query, document in batch],
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length - num_suffix_tokens,
                )

                # for key in tokenized.keys():
                #     tokenized[key] = [items + tokenized_suffix[key] for items in tokenized[key]]

                batch = [body + suffix for body in self.tokenizer.batch_decode(tokenized["input_ids"])]

                # self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
                # features = self.tokenizer.pad(
                #     tokenized,
                #     padding=True,
                #     return_tensors="pt",
                # )
            # else:
            # TODO: Add prompt on this level as well
            # features = self.tokenizer(
            #     batch,
            #     padding=True,
            #     truncation=True,
            #     return_tensors="pt",
            # )
            # features = self.tokenize(batch, **kwargs)
            # '''
            features = self.preprocess(batch, **kwargs)
            features.to(self.device)
            out_features = self.forward(features, **kwargs)
            scores = out_features["scores"]

            activation_fn = activation_fn or self.activation_fn
            if activation_fn is not None:
                scores = activation_fn(scores)

            # NOTE: This is just backwards compatibility with the code below, we can optimize this
            if scores.ndim == 1:
                scores = scores.unsqueeze(1)

            if apply_softmax and scores.ndim > 1:
                scores = torch.nn.functional.softmax(scores, dim=1)
            pred_scores.extend(scores)

        if self.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            if len(pred_scores):
                pred_scores = torch.stack(pred_scores)
            else:
                pred_scores = torch.tensor([], device=self.device)
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
        )

        results = []
        for i, score in enumerate(scores):
            results.append({"corpus_id": i, "score": score})
            if return_documents:
                results[-1].update({"text": documents[i]})

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _get_tokenized_suffix(self, suffix):
        # Cache tokenized suffix to avoid redundant tokenization
        if not hasattr(self, "_cached_tokenized_suffix"):
            self._cached_tokenized_suffix = {}
        cache = self._cached_tokenized_suffix
        if suffix not in cache:
            cache[suffix] = self.tokenizer(
                suffix,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=True,
            )
        return cache[suffix]

    def _get_model_config(self) -> dict[str, Any]:
        return super()._get_model_config() | {
            "activation_fn": fullname(self.activation_fn),
        }

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        if "activation_fn" in model_config:
            activation_fn_path = model_config["activation_fn"]
            # TODO: This is duplicate with get_default_activation_fn, definitely refactor
            if activation_fn_path is not None:
                if self.trust_remote_code or activation_fn_path.startswith("torch."):
                    self.activation_fn = import_from_string(activation_fn_path)()
                    return
                logger.warning(
                    f"Activation function path '{activation_fn_path}' is not trusted, using default activation function instead. "
                    "Please load the CrossEncoder with `trust_remote_code=True` to allow loading custom activation "
                    "functions via the configuration."
                )
