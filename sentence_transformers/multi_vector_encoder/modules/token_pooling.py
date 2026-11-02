from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import cache
from math import ceil, isfinite
from pathlib import Path
from typing import Any, overload

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
    """Abstract base for token pooling strategies. Subclasses implement :meth:`pool_one` (per-sample
    pooling) and set ``config_keys`` for save/load.

    Args:
        tasks: Task names this pooling applies to. A single string counts as a one-element list.
            Inputs encoded for any other task pass through unchanged. Defaults to ``["document"]``
            (ColBERT-style: only compress the document index). Use ``["query", "document"]`` for
            strategies that compress queries too, e.g. CRISP-style fixed-k clustering. Inputs
            encoded without a task count as ``"document"``.
    """

    config_keys: list[str] = ["tasks"]
    forward_kwargs: set[str] = {"task"}

    def __init__(self, *, tasks: str | list[str] | None = None) -> None:
        super().__init__()
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks: list[str] = list(tasks) if tasks is not None else ["document"]

    def applies_to(self, task: str | None) -> bool:
        """Whether this pooling applies to inputs encoded for ``task`` (``None`` counts as ``"document"``)."""
        return (task or "document") in self.tasks

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        if not self.applies_to(task):
            return features
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"].bool()
        device = token_embeddings.device
        if token_embeddings.shape[0] == 0:
            # Empty batch: keep the (0, T, D) shape and the (0, T) mask consistent with the input.
            return features
        pooled_list = [self.pool_one(emb[m]) for emb, m in zip(token_embeddings, attention_mask)]
        # Always right-pad back to (B, max_t, D). The output padding side is an internal detail:
        # downstream consumers read the returned mask, not the side.
        padded = _pad_to_3d(pooled_list, padding_side="right").to(device)
        lengths = torch.as_tensor([row.shape[0] for row in pooled_list], device=device, dtype=torch.long)
        new_mask = torch.arange(int(lengths.max()), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        features["token_embeddings"] = padded
        features["attention_mask"] = new_mask
        return features

    @overload
    def pool(
        self,
        embeddings: list[Tensor],
        *,
        task: str | None = ...,
        attention_mask: Tensor | None = ...,
        padding_side: str = ...,
    ) -> list[Tensor]: ...

    @overload
    def pool(
        self,
        embeddings: Tensor,
        *,
        task: str | None = ...,
        attention_mask: Tensor | None = ...,
        padding_side: str = ...,
    ) -> Tensor: ...

    def pool(
        self,
        embeddings: list[Tensor] | Tensor,
        *,
        task: str | None = None,
        attention_mask: Tensor | None = None,
        padding_side: str = "right",
    ) -> list[Tensor] | Tensor:
        """Apply the pool strategy to a list of 2D or a 3D padded tensor.

        Args:
            embeddings: A list of ``(t_i, D)`` tensors, or a 3D ``(B, T, D)`` padded tensor.
            task: Task the embeddings were encoded for. If it is not in :attr:`tasks`, the input
                is returned unchanged. ``None`` (default) counts as ``"document"``, so standalone
                document compression needs no extra argument.
                :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode` forwards its ``task``
                here for per-call pooling.
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
        if not self.applies_to(task):
            return embeddings
        if isinstance(embeddings, Tensor):
            if embeddings.ndim != 3:
                raise ValueError(f"Tensor input must be 3D (B, T, D); got shape {tuple(embeddings.shape)}.")
            if padding_side not in ("left", "right"):
                raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}.")
            if embeddings.shape[0] == 0:
                # Batch=0 short-circuit: preserve the input's (0, 0, D) shape.
                return embeddings.new_zeros((0, 0, embeddings.shape[-1]))
            embeddings_list = _unbind_padded(embeddings, attention_mask, padding_side)
            return _pad_to_3d([self.pool_one(e) for e in embeddings_list], padding_side)
        if isinstance(embeddings, list):
            if not embeddings:
                return []
            if embeddings[0].ndim != 2:
                raise ValueError(f"list entries must be 2D tensors; got shape {tuple(embeddings[0].shape)}.")
            return [self.pool_one(e) for e in embeddings]
        raise ValueError("embeddings must be a list of 2D tensors or a 3D padded tensor.")

    @abstractmethod
    def pool_one(self, embedding: Tensor, **kwargs) -> Tensor:
        """Pool a single ``(num_tokens, D)`` tensor into a ``(num_out, D)`` tensor."""

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)


@cache
def _ward_linkage_fn() -> Callable[..., np.ndarray]:
    """Resolve the Ward linkage backend once (fastcluster when installed, scipy otherwise).

    Python does not cache failed imports, so an uncached lookup would walk sys.path for every
    pooled document.
    """
    try:
        from fastcluster import linkage
    except (ImportError, OSError, RuntimeError):
        from scipy.cluster.hierarchy import linkage
    return linkage


def _normalize_token_embeddings(embedding: Tensor) -> Tensor:
    dtype = embedding.dtype
    if dtype in (torch.float16, torch.bfloat16):
        embedding = embedding.float()
    return torch.nn.functional.normalize(embedding, dim=-1).to(dtype)


def _hierarchical_linkage(embedding: Tensor) -> np.ndarray:
    from scipy.spatial.distance import squareform

    embedding = embedding.detach().float().cpu()
    distances = (1 - torch.mm(embedding, embedding.t())).clamp_(0, 2).numpy()
    return _ward_linkage_fn()(squareform(distances, checks=False), method="ward")


def _hierarchical_pool_one(
    embedding: Tensor,
    pool_factor: float,
    num_protected_tokens: int,
    normalize_embeddings: bool = True,
    max_tokens: int | None = None,
    distance_threshold: float | None = None,
) -> Tensor:
    """Ward hierarchical clustering on cosine distance for a single 2D embedding.

    Uses fastcluster for the Ward linkage when installed (moderately faster, same clustering as
    scipy except on exact distance ties). Falls back to scipy silently.
    """
    from scipy.cluster import hierarchy

    device = embedding.device
    protected = embedding[:num_protected_tokens]
    to_pool = embedding[num_protected_tokens:]
    num_to_pool = len(to_pool)
    if max_tokens is not None:
        num_clusters = min(max_tokens - num_protected_tokens, num_to_pool)
    else:
        num_clusters = max(int(num_to_pool // pool_factor), 1)

    if num_to_pool > 1 and (distance_threshold is not None or num_clusters < num_to_pool):
        to_pool = to_pool.cpu()
        linkage = _hierarchical_linkage(to_pool)
        if distance_threshold is None:
            labels = hierarchy.fcluster(linkage, t=num_clusters, criterion="maxclust") - 1
        else:
            labels = hierarchy.fcluster(linkage, t=distance_threshold, criterion="distance") - 1
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
        if cluster_sums.requires_grad:
            pooled = pooled.detach() + (cluster_sums - cluster_sums.detach())
        embedding = torch.cat([protected, pooled.to(device)], dim=0)

    return _normalize_token_embeddings(embedding) if normalize_embeddings else embedding


class HierarchicalTokenPooling(BaseTokenPooling):
    """Ward-linkage hierarchical clustering on cosine similarity. Excludes the first
    ``num_protected_tokens`` from clustering (typically the ``[CLS]``), clusters the rest into
    ``num_tokens // pool_factor`` groups, and replaces each cluster with its mean. All output tokens
    are L2-normalized by default.

    During training, a straight-through estimator gives each cluster member the full gradient
    through its cluster mean before output normalization, without dividing by the cluster size.

    Assumes L2-normalized embeddings (place after a
    :class:`~sentence_transformers.sentence_transformer.modules.Normalize` in the pipeline).

    Installing the optional ``fastcluster`` package speeds up the clustering step. Its output is
    identical to scipy's except on exact distance ties, which the two libraries may break
    differently.

    Reference compatibility: with ``normalize_embeddings=False``, this implementation matches PyLate
    after its condensed-Ward fix (PyLate > 1.3.4). It intentionally does NOT reproduce released
    PyLate <= 1.3.4 (square-matrix ``linkage`` quirk, protected tokens re-appended at the end) nor colpali-engine's
    ``HierarchicalTokenPooler`` (different linkage input, cluster means re-normalized to unit norm,
    no protected-token concept), so indexes pooled with those tools will not be byte-reproducible.
    For the closest colpali-engine setup, pass ``num_protected_tokens=0`` and ``normalize_embeddings=True``.

    Args:
        pool_factor: Keep roughly ``1 / pool_factor`` of each document's tokens. Fractional values
            (e.g. ``1.5`` keeps two thirds) allow compression steps between the integer factors.
            ``1.0`` (default) disables factor-based clustering. Output normalization still applies
            when enabled. Non-default factors are mutually exclusive with ``max_tokens`` and
            ``distance_threshold``.
        max_tokens: Maximum output token count, including protected tokens. Shorter documents
            are not clustered. Defaults to None.
        distance_threshold: Cut each document's Ward tree at this distance. Token counts vary
            with the document's embedding geometry. Mutually exclusive with ``max_tokens`` and
            a non-default ``pool_factor``. Use :meth:`calibrate_distance_threshold` to choose a threshold for an
            average token budget on representative document embeddings. Defaults to None.
        num_protected_tokens: Leading tokens excluded from pooling (typically ``[CLS]``). Default 1.
            colpali-engine has no protected-token concept: use ``0`` when matching its setup.
        normalize_embeddings: L2-normalize all output tokens, including protected tokens and when no
            tokens are merged. For float16 and bfloat16, computes in float32 and casts back to the
            input dtype. Defaults to True. Saved configurations without this setting load with False
            to preserve compatibility with existing indexes.
        tasks: Task names this pooling applies to. Defaults to ``["document"]`` (only compress
            documents).
    """

    config_keys: list[str] = [
        "pool_factor",
        "max_tokens",
        "distance_threshold",
        "num_protected_tokens",
        "normalize_embeddings",
        "tasks",
    ]

    def __init__(
        self,
        pool_factor: float = 1.0,
        *,
        max_tokens: int | None = None,
        distance_threshold: float | None = None,
        num_protected_tokens: int = 1,
        normalize_embeddings: bool = True,
        tasks: str | list[str] | None = None,
    ) -> None:
        super().__init__(tasks=tasks)
        if pool_factor < 1:
            raise ValueError(f"pool_factor must be >= 1, got {pool_factor}.")
        if num_protected_tokens < 0:
            raise ValueError(f"num_protected_tokens must be >= 0, got {num_protected_tokens}.")
        if distance_threshold is not None:
            if not isfinite(distance_threshold) or distance_threshold < 0:
                raise ValueError("distance_threshold must be finite and >= 0.")
            if pool_factor != 1 or max_tokens is not None:
                raise ValueError(
                    "distance_threshold, max_tokens and a non-default pool_factor are mutually exclusive."
                )
        if max_tokens is not None:
            if pool_factor != 1:
                raise ValueError(
                    f"pool_factor and max_tokens are mutually exclusive, got pool_factor={pool_factor} "
                    f"and max_tokens={max_tokens}."
                )
            if max_tokens <= num_protected_tokens:
                raise ValueError(
                    f"max_tokens must exceed num_protected_tokens to leave room for at least one pooled "
                    f"token, got max_tokens={max_tokens} and num_protected_tokens={num_protected_tokens}."
                )
        self.pool_factor = pool_factor
        self.max_tokens = max_tokens
        self.distance_threshold = distance_threshold
        self.num_protected_tokens = num_protected_tokens
        self.normalize_embeddings = normalize_embeddings

    @classmethod
    def load_config(cls, *args, **kwargs) -> dict[str, Any]:
        config = super().load_config(*args, **kwargs)
        config.setdefault("normalize_embeddings", False)
        return config

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        if self.pool_factor <= 1 and self.max_tokens is None and self.distance_threshold is None:
            if self.normalize_embeddings and self.applies_to(task):
                features["token_embeddings"] = _normalize_token_embeddings(features["token_embeddings"])
            return features
        return super().forward(features, task=task)

    def pool_one(self, embedding: Tensor, **kwargs) -> Tensor:
        return _hierarchical_pool_one(
            embedding,
            self.pool_factor,
            self.num_protected_tokens,
            self.normalize_embeddings,
            self.max_tokens,
            self.distance_threshold,
        )

    @staticmethod
    @torch.no_grad()
    def calibrate_distance_threshold(
        embeddings: Iterable[Tensor],
        *,
        target_tokens: float,
        num_protected_tokens: int = 1,
    ) -> float | None:
        """Choose a Ward distance threshold for an average document token budget.

        The calibration average is at most ``target_tokens``. Individual documents and the
        average on a different corpus can exceed the target.

        Args:
            embeddings: Representative, unpadded ``(num_tokens, dim)`` document embeddings from
                the same encoder and preprocessing used for the index. Inputs must be L2-normalized.
            target_tokens: Desired average output token count, including protected tokens.
                A budget at least as large as the input average returns None.
            num_protected_tokens: Leading tokens excluded from clustering. Defaults to 1. Use
                the same value when initializing the pooling module with the returned threshold.

        Returns:
            A threshold for initializing ``HierarchicalTokenPooling(distance_threshold=...)``,
            or None when the requested budget does not require compression.

        Example::

            threshold = HierarchicalTokenPooling.calibrate_distance_threshold(document_embeddings, target_tokens=16)
            pooling = HierarchicalTokenPooling(distance_threshold=threshold)
        """
        from scipy.cluster.hierarchy import maxdists

        if not isfinite(target_tokens) or target_tokens <= 0:
            raise ValueError("target_tokens must be finite and > 0.")
        if num_protected_tokens < 0:
            raise ValueError(f"num_protected_tokens must be >= 0, got {num_protected_tokens}.")
        heights = []
        count = total_tokens = minimum_tokens = 0
        for embedding in embeddings:
            if not isinstance(embedding, Tensor) or embedding.ndim != 2:
                raise ValueError("Calibration embeddings must be 2D tensors.")
            count += 1
            total_tokens += len(embedding)
            protected = min(num_protected_tokens, len(embedding))
            to_pool = embedding[protected:]
            minimum_tokens += protected + min(len(to_pool), 1)
            if len(to_pool) > 1:
                heights.append(maxdists(_hierarchical_linkage(to_pool)))
        if count == 0:
            raise ValueError("Provide at least one calibration document.")
        if target_tokens * count < minimum_tokens:
            raise ValueError(f"target_tokens must be at least {minimum_tokens / count:g} for these protected tokens.")
        merges = ceil(total_tokens - target_tokens * count)
        if merges > 0:
            all_heights = np.concatenate(heights)
            return float(np.partition(all_heights, merges - 1)[merges - 1])
        return None


class LambdaTokenPooling(BaseTokenPooling):
    """User-supplied pool function applied per-sample.

    Cannot be baked into a saved checkpoint (a Python callable isn't serializable), and will not
    round-trip through :meth:`MultiVectorEncoder.encode`'s multi-process path (``pool=`` arg) if
    ``pool_fn`` is a lambda or nested function. Use in the pipeline for experimentation, or
    standalone / per-call for ad-hoc compression.

    Args:
        pool_fn: A callable that takes a ``(num_tokens, dim)`` tensor and returns a
            ``(num_out, dim)`` tensor. Applied once per document in a batch.

    Example::

        def halve(emb: Tensor) -> Tensor:
            # Average consecutive pairs of tokens, dropping the tail if odd.
            n = emb.size(0)
            return emb[: n - n % 2].view(n // 2, 2, -1).mean(dim=1)

        pooling = LambdaTokenPooling(pool_fn=halve)
        pooled = pooling.pool(document_embeddings)
    """

    def __init__(self, pool_fn: Callable[[Tensor], Tensor], *, tasks: str | list[str] | None = None) -> None:
        super().__init__(tasks=tasks)
        self.pool_fn = pool_fn

    def pool_one(self, embedding: Tensor, **kwargs) -> Tensor:
        return self.pool_fn(embedding)

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        raise NotImplementedError(
            "LambdaTokenPooling wraps a user-supplied Python callable and cannot be saved. Use "
            "it standalone or per-call to encode(token_pooling=...). For a saveable pipeline module, "
            "subclass BaseTokenPooling with your own class."
        )
