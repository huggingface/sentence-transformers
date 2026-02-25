from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from sentence_transformers.models import Transformer
from sentence_transformers.multi_vec_encoder import LateInteractionPooling
from sentence_transformers.multi_vec_encoder.similarity import maxsim, maxsim_pairwise
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class MultiVectorEncoder(SentenceTransformer):
    """
    Multi-Vector Encoder for multi-vector encoding.

    Unlike standard SentenceTransformer which produces a single embedding per text,
    MultiVectorEncoder produces multiple embeddings (one per token) and computes
    similarity via MaxSim (maximum similarity) between token embeddings.

    Args:
        model_name_or_path: If it is a filepath on disk, it loads the model from that path.
            If it is not a path, it first tries to download a pre-trained MultiVectorEncoder.
            If that fails, tries to construct a model from the Hugging Face Hub with that name.
        modules: A list of torch Modules that should be called sequentially.
            Can be used to create custom MultiVectorEncoder models from scratch.
        device: Device (like "cuda", "cpu", "mps") that should be used for computation.
            If None, checks if a GPU can be used.
        prompts: A dictionary with prompts for the model. The key is the prompt name,
            the value is the prompt text.
        default_prompt_name: The name of the prompt that should be used by default.
        cache_folder: Path to store models.
        trust_remote_code: Whether or not to allow for custom models defined on the Hub.
        revision: The specific model version to use.
        local_files_only: Whether or not to only look at local files.
        token: Hugging Face authentication token to download private models.
        model_kwargs: Additional model configuration parameters.
        tokenizer_kwargs: Additional tokenizer configuration parameters.
        config_kwargs: Additional model configuration parameters.
        backend: The backend to use for inference ("torch", "onnx", or "openvino").
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        modules: Iterable[nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def encode_query(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[np.ndarray] | list[Tensor]:
        """
        Encode queries into multi-vector token embeddings.

        This method is a specialized version of :meth:`encode` that:
        1. Uses a predefined "query" prompt if available
        2. Sets the task to "query" for Router-based models

        Args:
            sentences: The sentences to embed.
            prompt_name: The name of the prompt to use. Defaults to "query" if available.
            prompt: The prompt text to prepend.
            batch_size: The batch size for encoding.
            show_progress_bar: Whether to show a progress bar.
            convert_to_numpy: Whether to convert outputs to numpy arrays.
            convert_to_tensor: Whether to convert outputs to tensors.
            device: Device to use for computation.
            normalize_embeddings: Whether to L2-normalize each token embedding.
            **kwargs: Additional arguments passed to encode.

        Returns:
            List of embeddings, each with shape [num_tokens, dim].
        """
        if prompt_name is None and "query" in self.prompts and prompt is None:
            prompt_name = "query"

        return self.encode(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            task="query",
            **kwargs,
        )

    def encode_document(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[np.ndarray] | list[Tensor]:
        """
        Encode documents into multi-vector token embeddings.

        This method is a specialized version of :meth:`encode` that:
        1. Uses a predefined "document" prompt if available
        2. Sets the task to "document" for Router-based models

        Args:
            sentences: The sentences to embed.
            prompt_name: The name of the prompt to use. Defaults to "document"/"passage"/"corpus" if available.
            prompt: The prompt text to prepend.
            batch_size: The batch size for encoding.
            show_progress_bar: Whether to show a progress bar.
            convert_to_numpy: Whether to convert outputs to numpy arrays.
            convert_to_tensor: Whether to convert outputs to tensors.
            device: Device to use for computation.
            normalize_embeddings: Whether to L2-normalize each token embedding.
            **kwargs: Additional arguments passed to encode.

        Returns:
            List of embeddings, each with shape [num_tokens, dim].
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
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            task="document",
            **kwargs,
        )

    def encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | None = None,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> list[np.ndarray] | list[Tensor]:
        """
        Encode sentences into multi-vector token embeddings.

        Unlike standard SentenceTransformer.encode() which returns a single embedding per sentence,
        this returns a list of embeddings (one per sentence), where each embedding has shape
        [num_tokens, dim] representing the token-level embeddings.

        Note:
            Token normalization is handled by LateInteractionPooling (normalize=True by default).
            The normalize_embeddings parameter is passed to the parent but does not affect
            token embeddings output - use the pooling layer's normalize setting instead.

        Args:
            sentences: The sentences to embed.
            prompt_name: The name of the prompt to use.
            prompt: The prompt text to prepend.
            batch_size: The batch size for encoding.
            show_progress_bar: Whether to show a progress bar.
            convert_to_numpy: Whether to convert outputs to numpy arrays.
            convert_to_tensor: Whether to convert outputs to tensors.
            device: Device to use for computation.
            normalize_embeddings: Passed to parent (normalization handled by pooling layer).
            **kwargs: Additional arguments.

        Returns:
            List of embeddings, each with shape [num_tokens, dim].
            The number of tokens varies per sentence based on tokenization.
        """
        return super().encode(
            sentences=sentences,
            prompt_name=prompt_name,
            prompt=prompt,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value="token_embeddings",
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

    def similarity(
        self,
        query_embeddings: list[np.ndarray] | list[Tensor] | Tensor,
        document_embeddings: list[np.ndarray] | list[Tensor] | Tensor,
    ) -> Tensor:
        """
        Compute MaxSim similarity between query and document embeddings.

        Args:
            query_embeddings: Query token embeddings. Either:
                - List of arrays/tensors with shape [num_query_tokens, dim] each
                - Padded tensor with shape [batch_q, max_query_tokens, dim]
            document_embeddings: Document token embeddings. Either:
                - List of arrays/tensors with shape [num_doc_tokens, dim] each
                - Padded tensor with shape [batch_d, max_doc_tokens, dim]

        Returns:
            Similarity scores with shape [batch_q, batch_d].
        """
        query_embs, query_mask = self._prepare_embeddings_for_similarity(query_embeddings)
        doc_embs, doc_mask = self._prepare_embeddings_for_similarity(document_embeddings)

        return maxsim(query_embs, doc_embs, query_mask, doc_mask)

    def similarity_pairwise(
        self,
        query_embeddings: list[np.ndarray] | list[Tensor] | Tensor,
        document_embeddings: list[np.ndarray] | list[Tensor] | Tensor,
    ) -> Tensor:
        """
        Compute pairwise MaxSim similarity between corresponding query-document pairs.

        Args:
            query_embeddings: Query token embeddings with batch size N.
            document_embeddings: Document token embeddings with batch size N.

        Returns:
            Similarity scores with shape [N].
        """
        query_embs, query_mask = self._prepare_embeddings_for_similarity(query_embeddings)
        doc_embs, doc_mask = self._prepare_embeddings_for_similarity(document_embeddings)

        return maxsim_pairwise(query_embs, doc_embs, query_mask, doc_mask)

    def _prepare_embeddings_for_similarity(
        self,
        embeddings: list[np.ndarray] | list[Tensor] | Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Convert embeddings to padded tensor format with attention mask.

        Args:
            embeddings: Either a list of variable-length embeddings with shape [num_tokens, dim] each,
                or a pre-padded tensor with shape [batch, max_tokens, dim].

        Returns:
            Tuple of (padded_embeddings, attention_mask).

        Raises:
            ValueError: If embeddings is empty or has inconsistent dimensions.
        """
        # Handle pre-padded tensor input
        if isinstance(embeddings, Tensor):
            mask = torch.ones(embeddings.shape[:-1], device=embeddings.device, dtype=torch.long)
            return embeddings, mask

        # Validate non-empty list
        if len(embeddings) == 0:
            raise ValueError("embeddings list cannot be empty")

        # Convert numpy arrays to tensors
        if isinstance(embeddings[0], np.ndarray):
            embeddings = [torch.from_numpy(e) for e in embeddings]

        # Validate all elements are tensors and have consistent dimensions, collect lengths
        dim = embeddings[0].shape[-1]
        lengths = []
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, Tensor):
                raise ValueError(f"Expected Tensor at index {i}, got {type(emb).__name__}")
            if emb.ndim != 2:
                raise ValueError(f"Expected 2D tensor [num_tokens, dim] at index {i}, got shape {emb.shape}")
            if emb.shape[-1] != dim:
                raise ValueError(f"Inconsistent embedding dimension at index {i}: expected {dim}, got {emb.shape[-1]}")
            lengths.append(emb.shape[0])

        # Padding
        padded = pad_sequence(embeddings, batch_first=True, padding_value=0.0)

        # Mask creation
        device = embeddings[0].device
        lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1)  # [batch_size, 1]
        positions = torch.arange(padded.shape[1], device=device).unsqueeze(0)  # [1, max_len]
        mask = (positions < lengths_tensor).long()

        return padded, mask

    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        return_documents: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Rank documents by relevance to a query using MaxSim.

        Args:
            query: The query string.
            documents: List of document strings to rank.
            top_k: Number of top documents to return. If None, returns all.
            return_documents: Whether to include document text in results.
            batch_size: Batch size for encoding documents.
            show_progress_bar: Whether to show a progress bar during encoding.
            **kwargs: Additional arguments passed to encode methods.

        Returns:
            List of dicts with keys:
                - "corpus_id": Index of the document
                - "score": MaxSim similarity score
                - "text": Document text (if return_documents=True)
        """
        # Encode query and documents
        query_embedding = self.encode_query(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=True,
            **kwargs,
        )
        document_embeddings = self.encode_document(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            **kwargs,
        )

        # Compute similarities
        scores = self.similarity(query_embedding, document_embeddings)
        scores = scores[0]  # Remove query batch dimension

        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)

        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        # Build results
        results = []
        for idx in sorted_indices:
            idx = int(idx)
            result = {
                "corpus_id": idx,
                "score": float(scores[idx]),
            }
            if return_documents:
                result["text"] = documents[idx]
            results.append(result)

        return results

    def get_token_embedding_dimension(self) -> int | None:
        """
        Returns the dimension of each token embedding.

        Returns:
            The token embedding dimension, or None if unknown.
        """
        # Check for LateInteractionPooling module
        for module in self._modules.values():
            if isinstance(module, LateInteractionPooling):
                return module.get_output_dimension()

        # Fall back to transformer dimension
        for module in self._modules.values():
            if hasattr(module, "get_word_embedding_dimension"):
                return module.get_word_embedding_dimension()

        return None

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns None since multi-vector encoder models produce multi-vector embeddings,
        not single sentence embeddings.
        """
        return None

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
        Creates a multi-vector encoder model from a transformer and returns the modules.

        For models without existing multi-vector encoder configuration, creates:
        - Transformer module
        - LateInteractionPooling module (with projection to 128 dimensions)
        """
        logger.warning(
            f"No multi-vector encoder model found with name {model_name_or_path}. "
            "Creating a new model with default multi-vector encoder configuration."
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

        # Create transformer module
        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args=model_kwargs,
            tokenizer_args=tokenizer_kwargs,
            config_args=config_kwargs,
            backend=self.backend,
        )

        # Create multi-vector pooling (project to 128 dimensions, normalize)
        word_embedding_dimension = transformer_model.get_word_embedding_dimension()
        pooling_model = LateInteractionPooling(
            word_embedding_dimension=word_embedding_dimension,
            output_dimension=128,
            normalize=True,
        )

        modules = [transformer_model, pooling_model]

        return modules

    def save(
        self,
        path: str,
        model_name: str | None = None,
        create_model_card: bool = True,
        train_datasets: list[str] | None = None,
        safe_serialization: bool = True,
    ) -> None:
        """
        Save the model and its configuration files to a directory.

        Args:
            path: Path on disk where the model will be saved.
            model_name: Optional model name.
            create_model_card: If True, create a README.md with model information.
            train_datasets: Optional list of dataset names used to train the model.
            safe_serialization: If True, save using safetensors format.
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
        Save the model and its configuration files to a directory.
        Alias for :meth:`save`.
        """
        return super().save_pretrained(
            path=path,
            model_name=model_name,
            create_model_card=create_model_card,
            train_datasets=train_datasets,
            safe_serialization=safe_serialization,
        )

    @property
    def max_seq_length(self) -> int:
        """Returns the maximum input sequence length for the model."""
        return super().max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value) -> None:
        """Sets the maximum input sequence length for the model."""
        self._first_module().max_seq_length = value
