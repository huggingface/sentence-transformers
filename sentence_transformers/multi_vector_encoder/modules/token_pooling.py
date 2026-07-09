"""Token pooling for multi-vector encoders.

Reduce each document's per-token embedding count to lower storage cost. Three ways to apply:

1. As a module in a :class:`~sentence_transformers.MultiVectorEncoder` pipeline: model authors
   put a :class:`HierarchicalTokenPooling` (or any :class:`BaseTokenPooling` subclass) at the end of
   the modules list to bake pooling into the checkpoint. Every consumer of the saved model gets
   the pooled output.
2. Per-call at encode time: users pass ``pooling=`` to
   :meth:`~sentence_transformers.MultiVectorEncoder.encode` / ``encode_query`` / ``encode_document``
   to opt in to pooling for a specific call.
3. Standalone: users call :meth:`BaseTokenPooling.pool` on already-encoded embeddings to compress
   before storage.

Pooling is not idempotent. Combining more than one of the above will pool multiple times.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from sentence_transformers.base.modules.module import Module


def _unbind_padded(embeddings: Tensor, attention_mask: Tensor | None, padding_side: str) -> list[Tensor]:
    """Split a padded ``(B, T, D)`` tensor into a list of ``(t_i, D)`` per-sample tensors.

    Pass ``attention_mask`` for the general case. Without a mask, falls back to zero-row detection
    (matches colpali_engine's convention): trims to the range from the first/last row with a
    non-zero embedding on the given ``padding_side``. Rows of all zeros in the middle of a
    sequence are preserved by the fallback path.
    """
    if attention_mask is not None:
        mask = attention_mask.bool()
        return [emb[m] for emb, m in zip(embeddings, mask)]
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}.")
    result: list[Tensor] = []
    for emb in embeddings:
        real = (emb.abs().sum(dim=-1) > 0).nonzero(as_tuple=True)[0]
        if real.numel() == 0:
            result.append(emb[:0])
        elif padding_side == "right":
            result.append(emb[: int(real[-1]) + 1])
        else:  # "left"
            result.append(emb[int(real[0]) :])
    return result


def _pad_to_3d(embeddings: list[Tensor], padding_side: str) -> Tensor:
    """Pad a list of ``(t_i, D)`` tensors back into a ``(B, max_t, D)`` tensor.

    Callers guarantee a non-empty list.
    """
    if padding_side == "right":
        return torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    # Left padding: reverse each row, right-pad, reverse the sequence axis back.
    reversed_embs = [emb.flip(dims=(0,)) for emb in embeddings]
    padded = torch.nn.utils.rnn.pad_sequence(reversed_embs, batch_first=True, padding_value=0.0)
    return padded.flip(dims=(1,))


class BaseTokenPooling(Module, ABC):
    """Abstract base for token pooling strategies. Subclasses implement :meth:`_pool_one` (per-sample
    pooling) and set ``config_keys`` for save/load. See the module docstring for the three ways to
    apply a pooling (pipeline module, per-call ``pooling=`` kwarg, standalone :meth:`pool`).
    """

    forward_kwargs: set[str] = {"task"}

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # Queries pass through (ColBERT-style: only compress the document index).
        if task == "query":
            return features
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"].bool()
        device = token_embeddings.device
        if token_embeddings.shape[0] == 0:
            # Empty batch: keep the (0, T, D) shape and the (0, T) mask consistent with the input.
            return features
        pooled_list = [self._pool_one(emb[m]) for emb, m in zip(token_embeddings, attention_mask)]
        # Always right-pad back to (B, max_t, D). The output padding side is an internal detail:
        # downstream consumers read the returned mask, not the side.
        padded = _pad_to_3d(pooled_list, padding_side="right").to(device)
        lengths = torch.as_tensor([row.shape[0] for row in pooled_list], device=device, dtype=torch.long)
        new_mask = torch.arange(int(lengths.max()), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        features["token_embeddings"] = padded
        features["attention_mask"] = new_mask
        return features

    def pool(
        self,
        embeddings: list[Tensor] | Tensor,
        *,
        attention_mask: Tensor | None = None,
        padding_side: str = "right",
    ) -> list[Tensor] | Tensor:
        """Apply the pool strategy to a list of 2D or a 3D padded tensor.

        Args:
            embeddings: A list of ``(t_i, D)`` tensors, or a 3D ``(B, T, D)`` padded tensor.
            attention_mask: Optional ``(B, T)`` boolean mask for the 3D case. Required unless
                the padding is zero-valued (in which case the input boundary is detected by
                ``padding_side``, but a real token whose embedding is exactly zero at the
                boundary of the content region will be clipped).
            padding_side: ``"left"`` or ``"right"``. Used to detect the input boundary when no
                mask is passed, and always used to re-pad the output when the input was 3D.

        Returns:
            List of ``(num_out, D)`` tensors when the input was a list. A padded 3D tensor
            when the input was 3D (padded on ``padding_side``).
        """
        if isinstance(embeddings, Tensor):
            if embeddings.ndim != 3:
                raise ValueError(f"Tensor input must be 3D (B, T, D); got shape {tuple(embeddings.shape)}.")
            if padding_side not in ("left", "right"):
                raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}.")
            if embeddings.shape[0] == 0:
                # Batch=0 short-circuit: preserve the input's (0, 0, D) shape.
                return embeddings.new_zeros((0, 0, embeddings.shape[-1]))
            embeddings_list = _unbind_padded(embeddings, attention_mask, padding_side)
            return _pad_to_3d([self._pool_one(e) for e in embeddings_list], padding_side)
        if isinstance(embeddings, list):
            if not embeddings:
                return []
            if embeddings[0].ndim != 2:
                raise ValueError(f"list entries must be 2D tensors; got shape {tuple(embeddings[0].shape)}.")
            return [self._pool_one(e) for e in embeddings]
        raise ValueError("embeddings must be a list of 2D tensors or a 3D padded tensor.")

    @abstractmethod
    def _pool_one(self, embedding: Tensor) -> Tensor:
        """Pool a single ``(num_tokens, D)`` tensor into a ``(num_out, D)`` tensor."""

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)


def _hierarchical_pool_one(embedding: Tensor, pool_factor: int, protected_tokens: int) -> Tensor:
    """Ward hierarchical clustering on cosine distance for a single 2D embedding."""
    from scipy.cluster import hierarchy

    device = embedding.device
    embedding = embedding.cpu()
    protected = embedding[:protected_tokens]
    to_pool = embedding[protected_tokens:]
    num_to_pool = len(to_pool)
    num_clusters = max(num_to_pool // pool_factor, 1)

    if num_clusters >= num_to_pool:
        return embedding.to(device)

    to_pool_fp32 = to_pool.float()
    cos_sim = torch.mm(to_pool_fp32, to_pool_fp32.t()).numpy()
    condensed = np.clip(1 - cos_sim, 0, 2)[np.triu_indices(num_to_pool, k=1)]
    linkage = hierarchy.linkage(condensed, method="ward")
    labels = hierarchy.fcluster(linkage, t=num_clusters, criterion="maxclust") - 1  # 0-indexed
    # Dense-index the labels so pooled rows line up with valid cluster ids even if fcluster returns
    # fewer than num_clusters distinct labels (possible with ties).
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    unique_labels, inverse = torch.unique(labels_tensor, return_inverse=True)
    num_actual = unique_labels.numel()
    cluster_sums = torch.zeros(num_actual, to_pool.shape[1], dtype=to_pool.dtype)
    cluster_counts = torch.zeros(num_actual, dtype=torch.long)
    cluster_sums.scatter_add_(0, inverse.unsqueeze(1).expand_as(to_pool), to_pool)
    cluster_counts.scatter_add_(0, inverse, torch.ones(num_to_pool, dtype=torch.long))
    pooled = cluster_sums / cluster_counts.unsqueeze(1)
    return torch.cat([protected, pooled], dim=0).to(device)


class HierarchicalTokenPooling(BaseTokenPooling):
    """Ward-linkage hierarchical clustering on cosine similarity. Keeps the first
    ``protected_tokens`` untouched (typically the ``[CLS]``), clusters the rest into
    ``num_tokens // pool_factor`` groups, and replaces each cluster with its mean.

    Ported from PyLate / colpali-engine. Assumes L2-normalized embeddings (place after a
    :class:`~sentence_transformers.sentence_transformer.modules.Normalize` in the pipeline).

    Args:
        pool_factor: Keep roughly ``1 / pool_factor`` of each document's tokens. ``1`` (default)
            disables pooling (the module becomes a no-op).
        protected_tokens: Leading tokens excluded from pooling (typically ``[CLS]``). Default 1.
    """

    config_keys: list[str] = ["pool_factor", "protected_tokens"]

    def __init__(self, pool_factor: int = 1, protected_tokens: int = 1) -> None:
        super().__init__()
        if pool_factor < 1:
            raise ValueError(f"pool_factor must be >= 1, got {pool_factor}.")
        if protected_tokens < 0:
            raise ValueError(f"protected_tokens must be >= 0, got {protected_tokens}.")
        self.pool_factor = pool_factor
        self.protected_tokens = protected_tokens

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # No-op fast path for the default disabled setting so an included-but-off module has no cost.
        if self.pool_factor <= 1:
            return features
        return super().forward(features, task=task)

    def _pool_one(self, embedding: Tensor) -> Tensor:
        return _hierarchical_pool_one(embedding, self.pool_factor, self.protected_tokens)


class LambdaTokenPooling(BaseTokenPooling):
    """User-supplied pool function applied per-sample.

    Cannot be baked into a saved checkpoint (a Python callable isn't serializable), and will not
    round-trip through :meth:`MultiVectorEncoder.encode`'s multi-process path (``pool=`` arg) if
    ``pool_func`` is a lambda or nested function. Use in the pipeline for experimentation, or
    standalone / per-call for ad-hoc compression.

    Args:
        pool_func: A callable that takes a ``(num_tokens, dim)`` tensor and returns a
            ``(num_out, dim)`` tensor. Applied once per document in a batch.

    Example::

        def halve(emb: Tensor) -> Tensor:
            # Average consecutive pairs of tokens; drop the tail if odd.
            n = emb.size(0)
            return emb[: n - n % 2].view(n // 2, 2, -1).mean(dim=1)

        pooling = LambdaTokenPooling(pool_func=halve)
        pooled = pooling.pool(document_embeddings)
    """

    def __init__(self, pool_func: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.pool_func = pool_func

    def _pool_one(self, embedding: Tensor) -> Tensor:
        return self.pool_func(embedding)

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        raise NotImplementedError(
            "LambdaTokenPooling wraps a user-supplied Python callable and cannot be saved. Use "
            "it standalone or per-call to encode(pooling=...). For a saveable pipeline module, "
            "subclass BaseTokenPooling with your own class."
        )
