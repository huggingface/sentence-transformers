from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from sentence_transformers.base.modules.module import Module

logger = logging.getLogger(__name__)


def pool_document_embeddings(
    token_embeddings: Tensor,
    pool_factor: int,
    protected_tokens: int = 1,
) -> Tensor:
    """Reduce a single document's token count via hierarchical (Ward) clustering.

    Clusters the ``(num_tokens, dim)`` embeddings into ``num_tokens // pool_factor`` clusters (Ward
    linkage on cosine distance) and replaces each cluster with its mean, keeping the first
    ``protected_tokens`` tokens (typically ``[CLS]``) untouched. Assumes the embeddings are
    L2-normalized, since clustering uses cosine similarity. Returns a tensor on the same device as the
    input. Ported from PyLate (``pylate/models/colbert.py``).
    """
    from scipy.cluster import hierarchy

    device = token_embeddings.device
    token_embeddings = token_embeddings.cpu()
    protected = token_embeddings[:protected_tokens]
    to_pool = token_embeddings[protected_tokens:]
    num_embeddings = len(to_pool)
    num_clusters = max(num_embeddings // pool_factor, 1)
    if num_clusters >= num_embeddings:
        return token_embeddings.to(device)

    cos_sim = torch.mm(to_pool, to_pool.t()).numpy()
    condensed = (1 - cos_sim)[np.triu_indices(num_embeddings, k=1)]
    linkage = hierarchy.linkage(condensed, method="ward")
    labels = hierarchy.fcluster(linkage, t=num_clusters, criterion="maxclust")
    labels_tensor = torch.from_numpy(labels.astype(np.int64)) - 1
    cluster_sums = torch.zeros(num_clusters, to_pool.shape[1], dtype=to_pool.dtype)
    cluster_counts = torch.zeros(num_clusters, dtype=torch.long)
    cluster_sums.scatter_add_(0, labels_tensor.unsqueeze(1).expand_as(to_pool), to_pool)
    cluster_counts.scatter_add_(0, labels_tensor, torch.ones(num_embeddings, dtype=torch.long))
    keep = cluster_counts > 0
    pooled = cluster_sums[keep] / cluster_counts[keep].unsqueeze(1)
    return torch.cat([protected, pooled], dim=0).to(device)


class HierarchicalPooling(Module):
    """Module that reduces each document's token count via hierarchical clustering.

    An opt-in, index-time storage optimization for late-interaction models: it clusters each
    document's token embeddings into ``num_tokens // pool_factor`` groups (Ward linkage on cosine
    distance) and replaces each group with its mean, keeping the first ``protected_tokens`` tokens
    (typically ``[CLS]``) untouched. Queries pass through unchanged.

    Place it at the *end* of the pipeline, after a token-level
    :class:`~sentence_transformers.sentence_transformer.modules.Normalize`, since the clustering metric
    assumes L2-normalized embeddings. It reads ``task`` from the forward kwargs and only pools when the
    task is not ``"query"``.

    This is the always-on, model-author-controlled counterpart to the per-call ``pool_factor`` argument
    of :meth:`MultiVectorEncoder.encode <sentence_transformers.MultiVectorEncoder.encode>`. Use one or
    the other, not both, or documents would be pooled twice.

    Args:
        pool_factor: Keep roughly ``1 / pool_factor`` of each document's tokens. 1 (default) disables
            pooling (the module becomes a no-op).
        protected_tokens: Number of leading tokens excluded from pooling. Defaults to 1 (the ``[CLS]``).
    """

    config_keys: list[str] = ["pool_factor", "protected_tokens"]
    forward_kwargs: set[str] = {"task"}

    def __init__(self, pool_factor: int = 1, protected_tokens: int = 1) -> None:
        super().__init__()
        self.pool_factor = pool_factor
        self.protected_tokens = protected_tokens

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        if self.pool_factor <= 1 or task == "query":
            return features

        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"].bool()
        pooled_rows = [
            pool_document_embeddings(emb[mask], self.pool_factor, self.protected_tokens)
            for emb, mask in zip(token_embeddings, attention_mask)
        ]

        device = token_embeddings.device
        lengths = [len(row) for row in pooled_rows]
        padded = torch.nn.utils.rnn.pad_sequence(pooled_rows, batch_first=True, padding_value=0.0)
        new_mask = torch.zeros(len(pooled_rows), max(lengths), dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            new_mask[i, :length] = True

        features["token_embeddings"] = padded
        features["attention_mask"] = new_mask
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
