from __future__ import annotations

import json
import logging
import math
import queue
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Any, ClassVar, Literal, overload

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import trange
from transformers.utils import logging as transformers_logging

from sentence_transformers.base import BaseModel
from sentence_transformers.base.modality_types import SingleInput
from sentence_transformers.base.modules import Transformer
from sentence_transformers.base.modules.dense import Dense
from sentence_transformers.multi_vector_encoder.model_card import MultiVectorEncoderModelCardData
from sentence_transformers.multi_vector_encoder.modules import HierarchicalPooling, MultiVectorMask
from sentence_transformers.multi_vector_encoder.modules.hierarchical_pooling import pool_document_embeddings
from sentence_transformers.sentence_transformer.modules import Normalize
from sentence_transformers.util import batch_to_device, load_file_path
from sentence_transformers.util.misc import import_from_string
from sentence_transformers.util.quantization import quantize_embeddings
from sentence_transformers.util.similarity import SimilarityFunction

logger = transformers_logging.get_logger(__name__)


# Rewrite PyLate's `pylate.*` refs to ST equivalents so we never import `pylate` at load (a no-op for native
# ST saves). The backbone Transformer holds the multi-vector knobs (query_length, do_query_expansion, ...)
# directly, so no class remapping is needed.
_CLASS_REF_ALIASES: dict[str, str] = {
    "pylate.models.Dense.Dense": "sentence_transformers.base.modules.dense.Dense",
}


@dataclass
class _LegacyStash:
    """Per-checkpoint values recovered from legacy save formats (PyLate v3 top-level config,
    Stanford-NLP ColBERT ``artifact.metadata``) that downstream load steps consume: prefix tokens
    to register on the tokenizer, multi-vector knobs to forward into ``Transformer.__init__`` via
    :meth:`MultiVectorEncoder._get_module_init_defaults`, and a skiplist word list to seed the
    default :class:`MultiVectorMask`. Empty for native MVE saves.
    """

    transformer_config: dict[str, Any] = field(default_factory=dict)
    prefixes: dict[str, str] = field(default_factory=dict)
    skiplist_words: list[str] | None = None
    is_pylate_v3: bool = False

    # Top-level PyLate keys that flow into ``Transformer.__init__`` via ``_get_module_init_defaults``.
    _PYLATE_TRANSFORMER_KEYS: ClassVar[tuple[str, ...]] = (
        "query_length",
        "document_length",
        "do_query_expansion",
        "attend_to_expansion_tokens",
    )


class MultiVectorEncoder(BaseModel):
    """
    Loads or creates a multi-vector / late-interaction (ColBERT-style) embedding model.

    Unlike :class:`~sentence_transformers.SentenceTransformer` which produces a single vector per input,
    :class:`MultiVectorEncoder` produces a *sequence* of vectors per input, one per token. Scoring between
    queries and documents is done with the MaxSim late-interaction operator: for each query token, take the
    max similarity to any document token, then sum across query tokens.

    Args:
        model_name_or_path (str, optional): If a filepath on disk, loads the model from that path. Otherwise,
            tries to download a pre-trained MultiVectorEncoder model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name. Defaults to None.
        modules (list[nn.Module], optional): A list of torch modules that are called sequentially. Can be used
            to create custom MultiVectorEncoder models from scratch. Defaults to None.
        device (str, optional): Device (like ``"cuda"``, ``"cpu"``, ``"mps"``, ``"npu"``) that should be used
            for computation. If None, checks if a GPU can be used. Defaults to None.
        prompts (dict[str, str], optional): Standard ST prompts dict, prepended to inputs by the encode methods.
            For ColBERT-style models supply ``{"query": "[Q] ", "document": "[D] "}`` (or whatever the model's
            prefix tokens are). Legacy PyLate / Stanford-NLP checkpoints stored these as separate
            ``query_prefix`` / ``document_prefix`` fields and are auto-promoted on load.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied. Defaults to None.
        cache_folder (str, optional): Path to store models. Can also be set by the
            ``SENTENCE_TRANSFORMERS_HOME`` environment variable. Defaults to None.
        trust_remote_code (bool, optional): Whether to allow for custom models defined on the Hub in their own
            modeling files. Defaults to False.
        revision (str, optional): The specific model version to use. Defaults to None.
        local_files_only (bool, optional): Whether to only look at local files. Defaults to False.
        token (bool or str, optional): Hugging Face authentication token. Defaults to None.
        model_kwargs (dict[str, Any], optional): Keyword arguments passed to the underlying Hugging Face
            Transformers model. Defaults to None.
        processor_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            processor / tokenizer. Defaults to None.
        config_kwargs (dict[str, Any], optional): Keyword arguments passed to the Hugging Face Transformers
            config. Defaults to None.
        model_card_data (MultiVectorEncoderModelCardData, optional): A model card data object. Defaults to None.
        backend (str, optional): The backend to use for inference. Only ``"torch"`` is supported.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function. Defaults
            to ``"maxsim"``.

    Note:
        Length / expansion / masking knobs (``query_length``, ``document_length``, ``do_query_expansion``,
        ``attend_to_expansion_tokens``, ``skiplist_words``, …) live on the underlying modules
        (:class:`~sentence_transformers.base.modules.Transformer` and
        :class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask`); set them after
        construction with e.g. ``model[0].query_length = 64``.

    Example:
        ::

            from sentence_transformers import MultiVectorEncoder

            model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

            queries = ["What is the capital of France?"]
            documents = [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ]

            query_embeddings = model.encode_query(queries)
            document_embeddings = model.encode_document(documents)

            scores = model.similarity(query_embeddings, document_embeddings)
            print(scores)
    """

    model_card_data_class = MultiVectorEncoderModelCardData
    default_huggingface_organization: str | None = None
    _default_prompts: dict[str, str | None] = {"query": None, "document": None}
    _model_card_model_id_placeholder = "multi_vector_encoder_model_id"
    model_type: str = "MultiVectorEncoder"

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
        model_card_data: MultiVectorEncoderModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        similarity_fn_name: str | SimilarityFunction | None = None,
    ) -> None:
        if backend != "torch":
            raise NotImplementedError(
                f"MultiVectorEncoder currently only supports backend='torch', got backend={backend!r}. "
                "ONNX/OpenVINO export is future work: the per-row masking and variable-length output do "
                "not map cleanly to fixed-shape graphs."
            )

        # Stash before super().__init__ so _parse_model_config only falls back to saved config when unset.
        self.similarity_fn_name = similarity_fn_name
        # Legacy-checkpoint state populated by ``_parse_model_config`` (PyLate v3) and
        # ``_maybe_load_stanford_metadata`` (Stanford-NLP); empty for native MVE saves.
        self._legacy = _LegacyStash()
        # User-supplied ``modules=...`` skips legacy module rewrites so an intentional sentence-level Dense isn't clobbered.
        self._user_supplied_modules = modules is not None

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
        self.model_card_data: MultiVectorEncoderModelCardData

        self._apply_legacy_fixups()

        # Resolve any MultiVectorMask's skiplist words against the tokenizer.
        for module in self._modules.values():
            if isinstance(module, MultiVectorMask):
                module.resolve_with_tokenizer(self.tokenizer)

    def encode_query(
        self,
        inputs: list[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        convert_to_padded_tensor: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        device: str | torch.device | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        pool_factor: int = 1,
        **kwargs: Any,
    ) -> list[Tensor] | list[np.ndarray] | Tensor | np.ndarray:
        """Compute query embeddings. Uses the "query" prompt if available and routes through the query side.

        See :meth:`encode` for the full parameter documentation. This method differs only by:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses the predefined ``"query"`` prompt when one
           exists in the model's ``prompts`` dictionary.
        2. It sets the ``task`` to ``"query"``: the query prefix token is inserted, the max sequence length is
           ``query_length``, and (when ``do_query_expansion=True``) the input is padded with mask tokens.
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
            convert_to_numpy=convert_to_numpy,
            convert_to_padded_tensor=convert_to_padded_tensor,
            precision=precision,
            device=device,
            normalize_embeddings=normalize_embeddings,
            pool=pool,
            chunk_size=chunk_size,
            pool_factor=pool_factor,
            task="query",
            **kwargs,
        )

    def encode_document(
        self,
        inputs: list[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        convert_to_padded_tensor: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        device: str | torch.device | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        pool_factor: int = 1,
        **kwargs: Any,
    ) -> list[Tensor] | list[np.ndarray] | Tensor | np.ndarray:
        """Compute document embeddings. Uses the first available of ``"document"`` / ``"passage"`` / ``"corpus"``
        prompts and routes through the document side.

        See :meth:`encode` for the full parameter documentation. This method differs only by:

        1. If no ``prompt_name`` or ``prompt`` is provided, it uses the first available of ``"document"`` /
           ``"passage"`` / ``"corpus"`` from the model's ``prompts`` dictionary.
        2. It sets the ``task`` to ``"document"``: the document prefix token is inserted, the max sequence
           length is ``document_length``, and skiplist tokens (e.g. punctuation) are excluded from the output.
        """
        if prompt_name is None and prompt is None:
            for candidate in ("document", "passage", "corpus"):
                if candidate in self.prompts:
                    prompt_name = candidate
                    break

        return self.encode(
            inputs=inputs,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            convert_to_padded_tensor=convert_to_padded_tensor,
            precision=precision,
            device=device,
            normalize_embeddings=normalize_embeddings,
            pool=pool,
            chunk_size=chunk_size,
            pool_factor=pool_factor,
            task="document",
            **kwargs,
        )

    # TODO: Consider replacing convert_to_* with return_as / output_format & incorporate "features" as well
    def encode(
        self,
        inputs: list[SingleInput] | SingleInput,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        convert_to_padded_tensor: bool = False,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        device: str | torch.device | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        # TODO: Should we remove pool_factor in favor of e.g. pool_kwargs for more flexibility in the future?
        # We can always update that in the future with a small decorator though
        pool_factor: int = 1,  # TODO: Should we reintroduce protected_tokens?
        task: str | None = None,
        **kwargs: Any,
    ) -> list[Tensor] | list[np.ndarray] | Tensor | np.ndarray:
        """Compute multi-vector token-level embeddings.

        .. tip::

            Prefer :meth:`encode_query` and :meth:`encode_document` for retrieval tasks. They set the
            ``task`` for you and route through the correct prefix / length / masking. Use :meth:`encode`
            directly only when you want to override the ``task`` explicitly.

        Args:
            inputs: The inputs to embed. Can be a string, a list of strings, or multimodal inputs
                (dicts, images, arrays).
            prompt_name (str, optional): The name of the prompt to use for encoding.
            prompt (str, optional): A prompt string to prepend to each input. Overrides ``prompt_name``.
            batch_size (int, optional): Batch size for the forward pass. Defaults to 32.
            show_progress_bar (bool, optional): Whether to show a progress bar. Defaults to None (auto).
            convert_to_tensor (bool, optional): If True, returns a list of :class:`torch.Tensor`. Overrides
                ``convert_to_numpy``. Defaults to False.
            convert_to_numpy (bool, optional): If True (default), returns a list of :class:`numpy.ndarray`.
            convert_to_padded_tensor (bool, optional): If True, pad each input's per-token embedding to the
                same length and return a single 3D :class:`torch.Tensor` of shape
                ``(num_inputs, max_tokens, embedding_dim)`` instead of a variable-length list. The
                padding-mask is reconstructable via ``(emb != 0).any(-1)``. Overrides ``convert_to_numpy``
                and ``convert_to_tensor`` (always returns a Tensor). Defaults to False.
            precision (str, optional): The output precision. One of ``"float32"``, ``"int8"``, ``"uint8"``,
                ``"binary"``, ``"ubinary"``. Defaults to ``"float32"``.
            device (str, torch.device, list, or None): Device(s) for computation. Defaults to None.
            normalize_embeddings (bool, optional): If True, L2-normalize each per-token embedding before
                returning. Use this when the loaded pipeline does not include a :class:`Normalize` module
                but you still want unit-norm vectors. No-op when a token-level ``Normalize`` already ran.
                Defaults to False.
            pool (dict, optional): A multi-process pool created via :meth:`start_multi_process_pool`.
            chunk_size (int, optional): Chunk size for multi-process encoding.
            pool_factor (int, optional): Hierarchical token-pooling factor for documents (1/pool_factor of
                the tokens are retained). 1 (default) disables pooling. Only applies to documents.
            task (str, optional): One of ``"query"``, ``"document"``, ``"passage"``, ``"corpus"``. Sets
                the prefix / length / masking strategy.

        Returns:
            list[Tensor] | list[ndarray] | Tensor | ndarray: By default, a list of per-input 2D arrays of shape
            ``(num_tokens_i, embedding_dim)`` (variable-length). With ``convert_to_padded_tensor=True``, a
            single 3D :class:`torch.Tensor` of shape ``(num_inputs, max_tokens, embedding_dim)``. If a single
            string is passed, the outer list is unwrapped (e.g. a bare 2D array for the default).
        """
        is_query = task == "query"

        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        # convert_to_tensor / convert_to_padded_tensor both produce Tensor output; suppress numpy conversion.
        if convert_to_tensor or convert_to_padded_tensor:
            convert_to_numpy = False

        is_singular_input = self.is_singular_input(inputs)
        if is_singular_input:
            inputs = [inputs]
        elif not isinstance(inputs, list):
            inputs = inputs.tolist() if isinstance(inputs, np.ndarray) else list(inputs)

        # Validate kwargs (matching SparseEncoder.encode behaviour).
        model_kwargs = self.get_model_kwargs()
        if unused_kwargs := set(kwargs) - set(model_kwargs) - {"task", "processing_kwargs"}:
            raise ValueError(
                f"{self.__class__.__name__}.encode() has been called with additional keyword arguments that "
                f"this model does not use: {list(unused_kwargs)}."
            )

        if pool is not None or (isinstance(device, list) and len(device) > 0):
            embeddings = self._multi_process(
                inputs=inputs,
                show_progress_bar=show_progress_bar,
                pool=pool,
                device=device,
                chunk_size=chunk_size,
                prompt_name=prompt_name,
                prompt=prompt,
                batch_size=batch_size,
                convert_to_tensor=convert_to_tensor,
                convert_to_numpy=convert_to_numpy,
                # Pad once after merging chunks, not per-worker: per-chunk max lengths would mismatch.
                convert_to_padded_tensor=False,
                precision=precision,
                normalize_embeddings=normalize_embeddings,
                pool_factor=pool_factor,
                task=task,
                **kwargs,
            )
            if convert_to_padded_tensor:
                embeddings = self._stack_padded(embeddings)
            if is_singular_input:
                embeddings = embeddings[0]
            return embeddings

        prompt = self._resolve_prompt(prompt, prompt_name)

        if device is None:
            device = self.device
        self.to(device)
        self.eval()

        all_embeddings: list[Tensor | np.ndarray] = []
        length_sorted_idx = np.argsort([-self._input_length(sen) for sen in inputs])
        inputs_sorted = [inputs[idx] for idx in length_sorted_idx]

        desc = f"Encoding {'queries' if is_query else 'documents'}"
        for start_index in trange(0, len(inputs), batch_size, desc=desc, disable=not show_progress_bar):
            inputs_batch = inputs_sorted[start_index : start_index + batch_size]
            features = self.preprocess(inputs_batch, prompt=prompt, task=task, **kwargs)
            features = batch_to_device(features, device)

            with torch.inference_mode():
                features = self.forward(features, task=task)
                token_embeddings = features["token_embeddings"]
                masks = features["attention_mask"].bool()
                batch_embeddings: list[Tensor] = [
                    token_embedding[mask] for token_embedding, mask in zip(token_embeddings, masks)
                ]
                if normalize_embeddings:
                    batch_embeddings = [nn.functional.normalize(emb, p=2, dim=-1) for emb in batch_embeddings]

            if pool_factor > 1 and not is_query:
                if any(isinstance(module, HierarchicalPooling) for module in self):
                    logger.warning_once(
                        f"Ignoring encode(pool_factor={pool_factor}): this model already includes a "
                        "`HierarchicalPooling` module, so per-call pooling would pool tokens twice. "
                        "Drop `pool_factor` from the encode call to silence this warning."
                    )
                else:
                    batch_embeddings = self.pool_embeddings_hierarchical(batch_embeddings, pool_factor=pool_factor)

            if convert_to_numpy:
                batch_embeddings = [emb.cpu() for emb in batch_embeddings]

            all_embeddings.extend(batch_embeddings)

        # Restore original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        # Quantize on the unpadded matrices
        if precision and precision != "float32":
            all_embeddings = quantize_embeddings(embeddings=all_embeddings, precision=precision)

        if convert_to_numpy:
            all_embeddings = [
                emb.float().cpu().numpy()
                if isinstance(emb, Tensor) and emb.dtype == torch.bfloat16
                else (emb.cpu().numpy() if isinstance(emb, Tensor) else emb)
                for emb in all_embeddings
            ]

        if convert_to_padded_tensor:
            result = self._stack_padded(all_embeddings)
        else:
            result = all_embeddings

        if is_singular_input:
            result = result[0]

        return result

    @staticmethod
    def _stack_padded(embeddings: Sequence[Tensor | np.ndarray]) -> Tensor:
        """Pad a variable-length list of per-input 2D embeddings into one
        ``(num_inputs, max_tokens, embedding_dim)`` tensor, padding with 0. The padding mask is
        recoverable via ``(emb != 0).any(-1)``.
        """
        if not embeddings:
            return torch.empty(0)
        tensors = [torch.from_numpy(emb) if isinstance(emb, np.ndarray) else emb for emb in embeddings]
        return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)

    def pool_embeddings_hierarchical(
        self,
        documents_embeddings: list[Tensor],
        pool_factor: int = 1,
        protected_tokens: int = 1,
    ) -> list[Tensor]:
        """Reduce the token count of each document via hierarchical (Ward) clustering.

        Keeps the first ``protected_tokens`` tokens unchanged (typically the [CLS] token), then clusters the
        remaining tokens into ``num_tokens // pool_factor`` clusters and replaces each cluster with its mean.
        Helpful for indexing long documents at lower storage cost.

        This is the per-call counterpart to the
        :class:`~sentence_transformers.multi_vector_encoder.modules.HierarchicalPooling` module: ship the
        module to bake always-on pooling into a model, or use this method / ``encode(pool_factor=...)``
        for ad-hoc pooling. Don't combine both, or documents get pooled twice.
        """
        return [
            pool_document_embeddings(doc.cpu(), pool_factor=pool_factor, protected_tokens=protected_tokens)
            for doc in documents_embeddings
        ]

    @property
    def similarity_fn_name(self) -> Literal["maxsim"]:
        """The similarity function used by :meth:`similarity` and :meth:`similarity_pairwise`. Defaults to
        ``"maxsim"`` on first access if not explicitly set."""
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.MAXSIM
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self,
        value: Literal["maxsim"] | SimilarityFunction | None,
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        if value is not None and value != SimilarityFunction.MAXSIM.value:
            raise ValueError(
                f"MultiVectorEncoder only supports the MaxSim similarity function, got {value!r}. "
                "Cosine / dot / euclidean / manhattan are defined on single vectors and don't compose "
                "with ragged per-token embeddings."
            )
        self._similarity_fn_name = value
        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    @overload
    def similarity(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor: ...

    @overload
    def similarity(self, embeddings1: list[Tensor], embeddings2: list[Tensor]) -> Tensor: ...

    @property
    def similarity(self) -> Callable[[Tensor | list[Tensor], Tensor | list[Tensor]], Tensor]:
        """Compute the all-pairs MaxSim score matrix between two collections of multi-vector embeddings.

        Returns a matrix of shape ``(num_embeddings_1, num_embeddings_2)``.

        Example::

            >>> model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
            >>> query_embeddings = model.encode_query(["What is the capital of France?"])
            >>> document_embeddings = model.encode_document(["Paris is the capital of France.", "Berlin is the capital of Germany."])
            >>> model.similarity(query_embeddings, document_embeddings)
            tensor([[..., ...]])
        """
        self.similarity_fn_name  # noqa: B018 (trigger lazy init)
        return self._similarity

    @overload
    def similarity_pairwise(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor: ...

    @overload
    def similarity_pairwise(self, embeddings1: list[Tensor], embeddings2: list[Tensor]) -> Tensor: ...

    @property
    def similarity_pairwise(
        self,
    ) -> Callable[[Tensor | list[Tensor], Tensor | list[Tensor]], Tensor]:
        """Compute the pairwise MaxSim score vector between matched query / document pairs."""
        self.similarity_fn_name  # noqa: B018 (trigger lazy init)
        return self._similarity_pairwise

    def _get_model_config(self) -> dict[str, Any]:
        config = super()._get_model_config()
        config["similarity_fn_name"] = self._similarity_fn_name
        return config

    def _apply_legacy_fixups(self) -> None:
        """Patch up modules loaded from save formats that predate :class:`MultiVectorEncoder`
        (PyLate v3, Stanford-NLP ColBERT, pre-v5.4 ST ``Dense``). Each step is a no-op for modern
        saves and for user-supplied ``modules=...``.
        """
        # Backwards-compat only: register a legacy in-vocab marker (e.g. [unused0]) as a special token so
        # text-prepending reproduces the trained tokenization. ``_legacy.prefixes`` is set only for those.
        if self._legacy.prefixes:
            self._register_prefix_tokens(self._legacy.prefixes)

        # PyLate v3 / pre-v5.4 ST Dense saved no IO names; redirect them to token level. Skip when the user
        # passed modules=... so an intentional sentence-level Dense in a hybrid pipeline isn't clobbered.
        if not self._user_supplied_modules:
            for module in self._modules.values():
                if isinstance(module, Dense) and module.module_input_name == "sentence_embedding":
                    module.module_input_name = "token_embeddings"
                    module.module_output_name = "token_embeddings"

        # PyLate v3 listed only [Transformer, Dense] (masking/normalize were inline). Append the missing
        # modules. Other load paths build the full sequence themselves.
        if self._legacy.is_pylate_v3:
            # PyLate <=3 applied a punctuation skiplist by default. Preserve that for v3 saves whose
            # config doesn't pin an explicit ``skiplist_words``.
            skiplist = (
                self._legacy.skiplist_words if self._legacy.skiplist_words is not None else list(string.punctuation)
            )
            self.append(MultiVectorMask(skiplist_words=skiplist))
            self.append(Normalize(module_input_name="token_embeddings"))

    def _register_prefix_tokens(self, prompts: dict[str, str]) -> None:
        """Mark a prompt-prefix token as special so the tokenizer emits it as a single piece.

        Call only with the prefixes of an existing token-prepended checkpoint (the caller guards on
        ``self._legacy.prefixes``). Needed for checkpoints (Stanford ColBERTv2, answerai-colbert, ...) whose
        prefix is an in-vocab marker like ``[unused0]`` applied via token insertion at training time, so
        their saved tokenizer never marked it special. Prepending it as text would shatter it
        (``[unused0]`` -> ``['[','unused','##0',']']``) and diverge from training; registering it
        restores single-piece tokenization, making text-prepending byte-identical to token insertion.

        Three gates keep this a no-op when no fix is required:

        1. Skip tokens already special / added (e.g. modern ``[Q] `` checkpoints). Nothing to do.
        2. Skip tokens not in the vocab: a non-vocab prefix (``[Q]`` on a plain BERT, or a text prompt
           like ``query: ``) is left as ordinary text rather than growing the embedding table.
        3. Skip tokens the tokenizer already emits as a single piece, no fix needed.
        """
        tokenizer = self.tokenizer
        if tokenizer is None:
            return
        added = set(getattr(tokenizer, "added_tokens_encoder", None) or {}) | set(tokenizer.all_special_tokens)
        vocab = tokenizer.get_vocab()
        to_register: list[str] = []
        for value in prompts.values():
            if not value or not value.split():
                continue
            prefix = value.split(None, 1)[0]
            if prefix in added or prefix not in vocab:
                continue
            if tokenizer.tokenize(prefix) == [prefix]:
                continue
            to_register.append(prefix)
        if to_register:
            tokenizer.add_special_tokens({"additional_special_tokens": to_register})

    def _parse_model_config(self, model_config: dict[str, Any]) -> None:
        super()._parse_model_config(model_config)
        # ``similarity_fn_name`` is not inherited from the saved config, as we're currently only supporting "maxsim".
        # PyLate v3 (model_type == "ColBERT") saved a plain Transformer and only [Transformer, Dense]. Flag it
        # so _apply_legacy_fixups appends the missing MultiVectorMask + token-level Normalize.
        self._legacy.is_pylate_v3 = model_config.get("model_type") == "ColBERT"
        # PyLate <=3 saved [Q]/[D] as top-level query_prefix/document_prefix (inserted as tokens). We route
        # them through `prompts` as text instead, recording them on the stash for special-token registration.
        for prefix_key, prompt_key in (("query_prefix", "query"), ("document_prefix", "document")):
            if prefix_key in model_config:
                self._legacy.prefixes[prompt_key] = model_config[prefix_key]
                if not self.prompts.get(prompt_key):
                    self.prompts[prompt_key] = model_config[prefix_key]
        # Filter ``None`` values so missing/null PyLate knobs fall through to the Transformer's own defaults.
        pylate_knobs = {
            key: model_config[key]
            for key in _LegacyStash._PYLATE_TRANSFORMER_KEYS
            if key in model_config and model_config[key] is not None
        }
        # PyLate defaults ``do_query_expansion`` to True. Apply that for any PyLate-style save that
        # didn't pin it, otherwise the [MASK] query expansion silently turns off.
        pylate_marker_keys = _LegacyStash._PYLATE_TRANSFORMER_KEYS + (
            "query_prefix",
            "document_prefix",
            "skiplist_words",
        )
        if "do_query_expansion" not in pylate_knobs and any(key in model_config for key in pylate_marker_keys):
            pylate_knobs["do_query_expansion"] = True
        self._legacy.transformer_config.update(pylate_knobs)
        self._legacy.skiplist_words = model_config.get("skiplist_words")

    def _load_module_class_from_ref(self, class_ref: str, *args: Any, **kwargs: Any) -> nn.Module:
        # Rewrite PyLate refs to ST equivalents (avoid importing pylate), then defer to the base resolver.
        # The backbone Transformer is promoted to MV by the loaders, not remapped here.
        class_ref = _CLASS_REF_ALIASES.get(class_ref, class_ref)
        return super()._load_module_class_from_ref(class_ref, *args, **kwargs)

    def _get_module_init_defaults(self, class_ref: str) -> dict[str, Any]:
        """Forward legacy top-level multi-vector knobs into the backbone Transformer's ``__init__``."""
        if not self._legacy.transformer_config:
            return {}
        class_ref = _CLASS_REF_ALIASES.get(class_ref, class_ref)
        try:
            cls = import_from_string(class_ref)
        except ImportError:
            return {}
        if not (isinstance(cls, type) and issubclass(cls, Transformer)):
            return {}
        return dict(self._legacy.transformer_config)

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
        """Build the default module sequence for a fresh MultiVectorEncoder.

        Two paths:
        1. Stanford-NLP ColBERT (``architectures == ["HF_ColBERT"]``): load the backbone, read
           ``artifact.metadata`` to recover special tokens / lengths, and append a token-level
           :class:`~sentence_transformers.base.modules.dense.Dense` loaded from the inline
           ``linear.weight`` stored at the repo root.
        2. Bare transformer: load the backbone and append a freshly-initialised projection layer
           operating on ``token_embeddings`` (output dim 128). To customise, pass ``modules=...``.
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

        config_json_path = load_file_path(
            model_name_or_path,
            "config.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        architectures: list[str] = []
        if config_json_path is not None:
            with open(config_json_path, encoding="utf-8") as fIn:
                architectures = json.load(fIn).get("architectures") or []
        is_stanford_colbert = "HF_ColBERT" in architectures
        if is_stanford_colbert:
            self._maybe_load_stanford_metadata(
                model_name_or_path,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
                token=token,
            )

        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            backend=self.backend,
            # Stanford-NLP artifact.metadata values (query_maxlen / doc_maxlen / attend_to_mask_tokens)
            # flow in here; empty stash leaves the Transformer at its base defaults.
            **self._legacy.transformer_config,
        )
        modules: list[nn.Module] = [transformer_model]

        if is_stanford_colbert:
            modules.append(
                self._build_stanford_projection(
                    model_name_or_path,
                    cache_folder=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                    token=token,
                )
            )
            logger.info(
                "Detected a Stanford-NLP ColBERT checkpoint; loaded the inline projection weights and metadata."
            )
        else:
            hidden_size = transformer_model.get_embedding_dimension()
            modules.append(
                Dense(
                    in_features=hidden_size,
                    out_features=128,
                    bias=False,
                    activation_function=nn.Identity(),
                    module_input_name="token_embeddings",
                )
            )
            logger.info(
                f"No ColBERT checkpoint detected; added a randomly-initialised projection of "
                f"({hidden_size}, 128). Training is required before this model is useful. "
                "To customise the projection (e.g. a different output dim), pass `modules=...` instead."
            )
        # Stanford-NLP loads pre-seed ``_legacy.skiplist_words`` via ``mask_punctuation``. Bare HF stays ``None`` (empty).
        modules.append(MultiVectorMask(skiplist_words=self._legacy.skiplist_words))
        modules.append(Normalize(module_input_name="token_embeddings"))

        if not local_files_only:
            self.model_card_data.set_base_model(model_name_or_path, revision=revision)
        return modules, {}

    @staticmethod
    def _build_stanford_projection(
        model_name_or_path: str,
        cache_folder: str | None,
        revision: str | None,
        local_files_only: bool,
        token: bool | str | None,
    ) -> Dense:
        """Build a token-level :class:`~sentence_transformers.base.modules.dense.Dense` from a
        Stanford-NLP ColBERT checkpoint.

        Stanford-NLP checkpoints (``colbert-ir/colbertv2.0`` and friends) store the projection weight
        at the repo root under the key ``linear.weight`` (alongside the encoder weights), rather than
        in a ``2_Dense/`` subfolder. We read that weight, infer the in/out dimensions, and return a
        freshly-initialised Dense module with the weight loaded.
        """
        weights = Dense.load_torch_weights(
            model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        linear_weight = weights["linear.weight"]
        out_features, in_features = linear_weight.shape
        return Dense(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            activation_function=nn.Identity(),
            init_weight=linear_weight,
            module_input_name="token_embeddings",
        )

    def _maybe_load_stanford_metadata(
        self,
        model_name_or_path: str,
        cache_folder: str | None,
        revision: str | None,
        local_files_only: bool,
        token: bool | str | None,
    ) -> None:
        """Read Stanford-NLP ColBERT settings from ``artifact.metadata`` and stash them on ``self._legacy``
        for the Transformer constructor + prefix-token registration to consume. Falls back to the
        standard ``[unused0]`` / ``[unused1]`` markers when the file is absent.
        """
        metadata_path = Dense.load_file_path(
            model_name_or_path,
            filename="artifact.metadata",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if metadata_path is None:
            logger.warning(
                "No artifact.metadata file found for the Stanford-NLP ColBERT checkpoint; using default values."
            )
            metadata = {}
        else:
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info("Loaded configuration from the Stanford-NLP ColBERT artifact.metadata file.")

        # Stanford-NLP ColBERT inserts these markers as token ids; record them for special-token registration.
        self._legacy.prefixes = {
            "query": (metadata.get("query_token_id") or "[unused0]") + " ",
            "document": (metadata.get("doc_token_id") or "[unused1]") + " ",
        }
        for role, marker in self._legacy.prefixes.items():
            if not self.prompts.get(role):
                self.prompts[role] = marker
        for meta_key, attr in (
            ("query_maxlen", "query_length"),
            ("doc_maxlen", "document_length"),
            ("attend_to_mask_tokens", "attend_to_expansion_tokens"),
        ):
            if metadata.get(meta_key) is not None:
                self._legacy.transformer_config.setdefault(attr, metadata[meta_key])
        # Stanford-NLP ColBERT always [MASK]-expands queries (core scoring trick, not in ``artifact.metadata``).
        self._legacy.transformer_config.setdefault("do_query_expansion", True)
        # Stanford-NLP's ``--mask-punctuation`` CLI flag defaults to ``False`` (``store_true``). Follow that for missing keys.
        self._legacy.skiplist_words = list(string.punctuation) if metadata.get("mask_punctuation", False) else []

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
        """Convert a SentenceTransformer (and similar) checkpoint into a MultiVectorEncoder.

        If a final :class:`~sentence_transformers.base.modules.dense.Dense` head is present, it is
        redirected to operate on ``token_embeddings`` (preserving the learned projection weights);
        otherwise a fresh randomly-initialised token-level projection is appended.
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

        modules, module_kwargs = self._load_config_modules(
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
        modules_list = list(modules.values())

        # Drop pooling / sentence-level Normalize (we want token-level); a token-level Normalize is re-appended below.
        from sentence_transformers.sentence_transformer.modules import Pooling

        filtered: list[nn.Module] = []
        for module in modules_list:
            if isinstance(module, Pooling):
                continue
            if isinstance(module, Normalize) and module.module_input_name == "sentence_embedding":
                continue
            filtered.append(module)

        transformer = next((m for m in filtered if isinstance(m, Transformer)), None)
        if transformer is None:
            raise ValueError(
                "Cannot convert this SentenceTransformer checkpoint into a MultiVectorEncoder: "
                "no Transformer module was found among the loaded modules."
            )

        if not any(isinstance(m, Dense) for m in filtered):
            hidden_size = transformer.get_embedding_dimension()
            filtered.append(
                Dense(
                    in_features=hidden_size,
                    out_features=128,
                    bias=False,
                    activation_function=nn.Identity(),
                    module_input_name="token_embeddings",
                )
            )
            logger.info(
                f"Appended a randomly-initialised projection ({hidden_size}, 128) to a SentenceTransformer "
                "checkpoint. Training is required before this model is useful. To customise the projection "
                "(e.g. a different output dim), pass `modules=...` instead."
            )
        # PyLate saves stash an explicit ``skiplist_words`` here. Bare ST checkpoints stay ``None`` (empty default).
        filtered.append(MultiVectorMask(skiplist_words=self._legacy.skiplist_words))
        filtered.append(Normalize(module_input_name="token_embeddings"))

        # Source is single-vector: its inherited "cosine"/"dot" can't score ragged per-token embeddings, so
        # this now-multi-vector model uses MaxSim.
        self.similarity_fn_name = SimilarityFunction.MAXSIM

        # The original README is for a different architecture; clear it so we don't accidentally serve it.
        self._model_card_text = None
        return filtered, module_kwargs

    def _get_model_type(
        self,
        model_name_or_path: str,
        token: bool | str | None,
        cache_folder: str | None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        """Detect the model type. Adds Stanford-NLP and PyLate-v3 detection on top of the base behaviour.

        - If ``config.json`` has ``architectures == ["HF_ColBERT"]`` (Stanford ColBERT), return
          ``"MultiVectorEncoder"`` so we route through ``_load_default_modules`` to read the inline weights.
        - If ``config_sentence_transformers.json`` has ``model_type == "ColBERT"`` (PyLate v3), normalise it
          to ``"MultiVectorEncoder"`` so the standard config-modules loader runs.
        """
        # Check the config_sentence_transformers.json first.
        config_st_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_st_json_path is not None:
            with open(config_st_json_path, encoding="utf8") as fIn:
                cfg = json.load(fIn)
            model_type = cfg.get("model_type")
            if model_type == "ColBERT":
                return "MultiVectorEncoder"
            if model_type is not None:
                return model_type

        # Check the HF config.json for the HF_ColBERT architecture marker.
        config_json_path = load_file_path(
            model_name_or_path,
            "config.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        if config_json_path is not None:
            with open(config_json_path, encoding="utf-8") as fIn:
                if "HF_ColBERT" in (json.load(fIn).get("architectures") or []):
                    return "MultiVectorEncoder"

        # Fall back to the base behaviour.
        return super()._get_model_type(
            model_name_or_path,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

    def get_embedding_dimension(self) -> int | None:
        """The dimensionality of each token vector returned by :meth:`encode`."""
        for module in reversed(self._modules.values()):
            method = getattr(module, "get_embedding_dimension", None)
            if callable(method):
                return method()
        return None

    def _multi_process(
        self,
        inputs: list[SingleInput],
        show_progress_bar: bool | None = True,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        device: str | torch.device | list[str | torch.device] | None = None,
        chunk_size: int | None = None,
        **encode_kwargs,
    ) -> list[Tensor] | list[np.ndarray]:
        encode_kwargs["show_progress_bar"] = False
        created_pool = False
        if pool is None and isinstance(device, list):
            pool = self.start_multi_process_pool(device)
            created_pool = True
        try:
            if chunk_size is None:
                chunk_size = min(math.ceil(len(inputs) / len(pool["processes"]) / 10), 5000)
                chunk_size = max(chunk_size, 1)

            input_queue: Queue = pool["input"]
            output_queue: Queue = pool["output"]

            num_chunks = math.ceil(len(inputs) / chunk_size) if inputs else 0
            for chunk_id in range(num_chunks):
                start = chunk_id * chunk_size
                input_queue.put([chunk_id, inputs[start : start + chunk_size], encode_kwargs])

            output_list = sorted(
                [output_queue.get() for _ in trange(num_chunks, desc="Chunks", disable=not show_progress_bar)],
                key=lambda x: x[0],
            )

            for output in output_list:
                if isinstance(output[1], Exception):
                    raise output[1]

            embeddings: list[Tensor | np.ndarray] = []
            for _, chunk_result in output_list:
                if isinstance(chunk_result, list):
                    embeddings.extend(chunk_result)
                else:
                    embeddings.append(chunk_result)
            return embeddings
        finally:
            if created_pool:
                self.stop_multi_process_pool(pool)

    @staticmethod
    def _multi_process_worker(
        target_device: str, model: MultiVectorEncoder, input_queue: Queue, results_queue: Queue
    ) -> None:
        while True:
            try:
                chunk_id, inputs, kwargs = input_queue.get()
                embeddings = model.encode(inputs, device=target_device, **kwargs)
                if isinstance(embeddings, list):
                    embeddings = [
                        emb.cpu() if isinstance(emb, Tensor) and emb.device.type != "cpu" else emb
                        for emb in embeddings
                    ]
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
queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
```

---
*This PR was auto-generated with \
[`push_to_hub`](https://sbert.net/docs/package_reference/multi_vector_encoder/MultiVectorEncoder.html#sentence_transformers.multi_vector_encoder.MultiVectorEncoder.push_to_hub).*
"""
