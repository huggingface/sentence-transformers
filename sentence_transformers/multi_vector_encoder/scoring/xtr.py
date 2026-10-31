from __future__ import annotations

import numpy as np
import torch
from transformers.utils import is_torchdynamo_compiling

from sentence_transformers.util.similarity import _chunk_ranges, _fill_empty_document_scores, _zero_row_mask
from sentence_transformers.util.tensor import _convert_to_tensor


def xtr_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    top_k: int = 256,
    chunk_elements: int | None = None,
) -> torch.Tensor:
    """XTR (eXtendable Token Retrieval) contrastive scoring with global top-k token retrieval.

    For each query token, the top-k matches are selected globally across all in-batch document tokens
    (simulating retrieval from an index). Returns the full ``(Q, Q*N)`` cross-product score matrix with
    query-major ordering: ``scores[i, j*N + n]`` is query ``i`` against query ``j``'s ``n``-th document.
    The positive for query ``i`` sits at column ``i*N``. As in :func:`~sentence_transformers.util.similarity.maxsim`,
    the matmul runs in the input dtype (integer embeddings are upcast to float32 first) and the
    per-query-token accumulation and returned scores are float32.

    Each score is the sum of the query's retrieved per-token maxima divided by ``Z``, the number of
    query tokens that retrieved at least one of the document's tokens (Lee et al. 2023, eq. 5). This
    deviates from PyLate / PrimeQA, which divide by the count of positive per-token maxima instead.

    Args:
        queries_embeddings: ``(Q, q_tokens, dim)`` query embeddings.
        documents_embeddings: ``(Q, N, d_tokens, dim)`` stacked per-query document groups.
        queries_mask: optional ``(Q, q_tokens)`` mask. If None, all-zero query rows are treated as padding.
        documents_mask: optional ``(Q, N, d_tokens)`` mask. If None, one is derived by treating
            all-zero document rows as padding (like :func:`~sentence_transformers.util.similarity.maxsim`).
        top_k: Positive number of top token matches to retain per query token across all Q*N documents.
        chunk_elements: Element budget for the matmul + ``masked_fill`` phase, matching
            :func:`~sentence_transformers.util.similarity.maxsim`'s parameter of the same name:
            CPU eager training scoring streams global top-k candidates and recomputes only winning dot products
            for autograd, avoiding the full token similarity grid when the estimated winner gathers
            are smaller. The budget covers a matmul chunk, not all live state: top-k candidates,
            per-document reductions and gathered embeddings also consume memory. Cutoff ties require
            a full token-score row to preserve top-k selection. This path requires enabled autograd and at
            least one input requiring gradients. Inference, non-CPU devices, compilation, nonfinite or
            sentinel scores, and shapes with expensive winner gathers retain the legacy dense reduction.
            Memory savings trade off against additional selection work; no latency improvement is
            guaranteed. Defaults to None (legacy single matmul).

    Notes:
        Adapted from PyLate / PrimeQA (Apache 2.0). Pass this (or :class:`XTRScores`, a configured reusable
        instance) as ``similarity_fct`` to any :mod:`~sentence_transformers.multi_vector_encoder.losses` loss
        to switch from ColBERT-style MaxSim scoring to XTR-style top-k scoring. To compile the hot path,
        wrap it: ``similarity_fct=torch.compile(xtr_scores)``.
    """
    if isinstance(top_k, bool) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

    # Integer (quantized) embeddings would carry through to an integer score grid, which the
    # dtype-min masked_fill below cannot express: torch.finfo rejects it.
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    if not queries_embeddings.is_floating_point():
        queries_embeddings = queries_embeddings.float()
    if not documents_embeddings.is_floating_point():
        documents_embeddings = documents_embeddings.float()
    Qb, Qt, H = queries_embeddings.shape
    D, N, Dt, _ = documents_embeddings.shape
    Db = D * N
    docs_flat = documents_embeddings.reshape(Db, Dt, H)
    Q_flat = queries_embeddings.reshape(Qb * Qt, H)
    if documents_mask is not None:
        docs_mask_flat = documents_mask.reshape(Db, Dt)
    else:
        # Mirror maxsim: treat all-zero rows as padding, or a pad token's 0.0 wins the per-document
        # max over genuinely negative real similarities.
        docs_mask_flat = _zero_row_mask(docs_flat)
    if queries_mask is None:
        # Match the document-side fallback: zero-padded query rows must not contribute to scores or Z.
        queries_mask = _zero_row_mask(queries_embeddings).bool()

    if (
        chunk_elements is not None
        and queries_embeddings.device.type == "cpu"
        and documents_embeddings.device.type == "cpu"
        and not is_torchdynamo_compiling()
        and torch.is_grad_enabled()
        and not torch.is_inference_mode_enabled()
        and (queries_embeddings.requires_grad or documents_embeddings.requires_grad)
    ):
        streamed = _streaming_xtr_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask,
            docs_mask_flat.reshape(D, N, Dt),
            top_k,
            chunk_elements,
        )
        if streamed is not None:
            return streamed

    if chunk_elements is None:
        D_flat = docs_flat.reshape(Db * Dt, H).T
        scores = (Q_flat @ D_flat).view(Qb, Qt, Db, Dt)
        scores.masked_fill_(
            ~docs_mask_flat.bool().unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
    else:
        # Same per-document-token cost accounting as maxsim: the (Qb, Qt) score column plus the
        # embedding row.
        score_chunks = []
        for d_start, d_end in _chunk_ranges([Dt] * Db, Qb * Qt + H, chunk_elements):
            db = d_end - d_start
            chunk_D_flat = docs_flat[d_start:d_end].reshape(db * Dt, H).T
            chunk_scores = (Q_flat @ chunk_D_flat).view(Qb, Qt, db, Dt)
            chunk_mask = docs_mask_flat[d_start:d_end]
            chunk_scores.masked_fill_(
                ~chunk_mask.bool().unsqueeze(0).unsqueeze(0),
                torch.finfo(chunk_scores.dtype).min,
            )
            score_chunks.append(chunk_scores)
        scores = torch.cat(score_chunks, dim=2)
        # Freed before the scatter below, or the per-chunk copies and the full grid are alive at once.
        del score_chunks, chunk_scores

    clubbed = scores.flatten(2, 3)
    top_values, indices = clubbed.topk(min(top_k, clubbed.shape[-1]), dim=-1, sorted=False)
    # Non-retrieved tokens sit at the dtype minimum, or a 0 placeholder wins the per-document max over
    # a genuinely retrieved negative. Filled in place: clubbed views scores, which is dead from here.
    masked = clubbed.fill_(torch.finfo(clubbed.dtype).min).scatter_(-1, indices, top_values)
    # Paper-exact Z (Lee et al. 2023, eq. 5): the count of query tokens that retrieved at least one
    # real token of the document. PyLate / PrimeQA count positive per-token maxima with a 1e-3 clamp
    # instead, which amplifies all-negative rows over 1000x and diverges once top_k spans the token pool.
    # Pad-slot retrievals (top_k spanning the pool) land in a throwaway Db-th bucket.
    doc_ids = torch.where(docs_mask_flat.bool().reshape(-1)[indices], indices // Dt, Db)
    alpha = torch.zeros(Qb, Qt, Db + 1, dtype=torch.bool, device=indices.device).scatter_(-1, doc_ids, True)[..., :Db]
    # fp32 accumulation like maxsim: summing per-token maxima in half precision collapses the
    # score grid (PyLate instead sums in the input dtype and casts the result).
    topk_scores_max = masked.view(Qb, Qt, Db, Dt).max(dim=-1).values.float()
    # XTR imputes a missing similarity as 0, which is what a document with no retrieved token gets.
    topk_scores_max = topk_scores_max.masked_fill(~alpha, 0.0)

    topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)
    alpha = alpha & queries_mask.bool().unsqueeze(-1)

    scores_sum = topk_scores_max.sum(dim=1)
    Z = alpha.float().sum(dim=1).clamp_(min=1)
    scores = scores_sum / Z
    # A fully masked document retrieves nothing, so the imputation above scores it 0 and it would rank
    # among the real documents. The sentinel puts it below every one of them.
    return _fill_empty_document_scores(scores, ~docs_mask_flat.bool().any(dim=-1))


def xtr_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    top_k: int = 256,
    chunk_elements: int | None = None,
) -> torch.Tensor:
    """XTR scoring for knowledge distillation.

    Same global top-k scoring as :func:`xtr_scores`, but returns each query's own N-way document scores
    ``(Q, N)`` instead of the full ``(Q, Q*N)`` cross-product, matching the interface expected by
    :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss`.
    """
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    Q, N = documents_embeddings.shape[:2]
    all_scores = xtr_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        top_k=top_k,
        chunk_elements=chunk_elements,
    )
    idx = torch.arange(Q, device=all_scores.device).unsqueeze(1) * N + torch.arange(N, device=all_scores.device)
    return all_scores.gather(1, idx)


def xtr_scores_pairwise(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    top_k: int = 256,
    chunk_elements: int | None = None,
) -> torch.Tensor:
    """Pairwise XTR scoring: compute the XTR score for matched ``(query_i, document_i)`` pairs.

    Returns a float32 1D tensor of length ``batch_size``. XTR's top-k runs globally over the batch's
    pooled document tokens, so unlike MaxSim a pair's score shifts with the batch composition
    whenever ``top_k`` is below the pooled token count.
    """
    # Reshape to (Q, N=1, Dt, H), score with xtr_kd_scores, take the diagonal.
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    if documents_embeddings.ndim == 3:
        documents_embeddings = documents_embeddings.unsqueeze(1)
    if documents_mask is not None and documents_mask.ndim == 2:
        documents_mask = documents_mask.unsqueeze(1)
    scores = xtr_kd_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        top_k=top_k,
        chunk_elements=chunk_elements,
    )
    return scores.squeeze(-1)


class XTRScores:
    """Configured, reusable :func:`xtr_scores` callable for use as a loss ``similarity_fct``.

    Stores ``top_k`` / ``chunk_elements`` so they don't have to be re-passed on every call (the bare
    function would otherwise need :func:`functools.partial`). See :func:`xtr_scores` for the scoring math
    and shapes.

    Args:
        top_k: Positive number of top token matches to retain per query token across all Q*N documents.
        chunk_elements: Element budget for the chunked matmul phase, see :func:`xtr_scores`.
    """

    def __init__(self, top_k: int = 256, *, chunk_elements: int | None = None) -> None:
        self.top_k = top_k
        self.chunk_elements = chunk_elements

    def get_config_dict(self) -> dict[str, int | None]:
        return {"top_k": self.top_k, "chunk_elements": self.chunk_elements}

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return xtr_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            top_k=self.top_k,
            chunk_elements=self.chunk_elements,
        )


class XTRKDScores(XTRScores):
    """Configured, reusable :func:`xtr_kd_scores` callable (KD ``(Q, N)`` output). See :class:`XTRScores`."""

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return xtr_kd_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            top_k=self.top_k,
            chunk_elements=self.chunk_elements,
        )


def _streaming_xtr_scores(
    q: torch.Tensor,
    d: torch.Tensor,
    queries_mask: torch.Tensor,
    documents_mask: torch.Tensor,
    top_k: int,
    chunk_elements: int,
) -> torch.Tensor | None:
    """Score finite CPU eager inputs without retaining the full token-score grid.

    Return None to retain legacy exceptional behavior or avoid an obviously adverse shape.
    Selection is nondifferentiable in both implementations. Only the selected per-document
    winners need a differentiable dot product; ties within a document keep the first token's
    gradient, matching torch.max. Gather/scatter and matmul accumulation order can differ in
    reduced precision, so gradient equivalence is numerical, not bitwise.
    """
    Q, T, H = q.shape
    D, N, L, _ = d.shape
    B = D * N
    # Each winner gathers both a query and a document vector for the differentiable bmm.
    # Short documents with large embedding dimensions can cost more than the dense grid.
    if 2 * min(top_k, B) * H >= B * L:
        return None
    qf, df = q.reshape(Q * T, H), d.reshape(B * L, H)
    # Preserve legacy exceptional and nonfinite behavior, including dense backward's 0*Inf.
    if not Q or not T or not B or not L or not torch.isfinite(q).all() or not torch.isfinite(d).all():
        return None
    qm = q.any(-1) if queries_mask is None else queries_mask
    dm = d.any(-1) if documents_mask is None else documents_mask
    dm = dm.reshape(-1).bool()
    k = min(top_k, len(df))
    keep = min(k + 1, len(df))
    with torch.no_grad():
        vals = None
        ids = torch.empty(Q * T, 0, dtype=torch.long, device=q.device)
        for begin, end in _chunk_ranges([L] * B, Q * T + H, chunk_elements):
            lo, hi = begin * L, end * L
            scores = qf @ df[lo:hi].T
            sentinel = torch.finfo(scores.dtype).min
            if not torch.isfinite(scores).all() or ((scores == sentinel) & dm[lo:hi]).any():
                return None
            if vals is None:
                vals = scores.new_empty(Q * T, 0)
            scores.masked_fill_(~dm[lo:hi], sentinel)
            local_values, local_ids = scores.topk(min(keep, hi - lo), dim=-1, sorted=False)
            merged = torch.cat((vals, local_values), dim=-1)
            merged_ids = torch.cat((ids, local_ids + lo), dim=-1)
            vals, ix = merged.topk(min(keep, merged.shape[-1]), dim=-1, sorted=True)
            ids = merged_ids.gather(-1, ix)
        if not torch.isfinite(vals).all():
            return None
        # Resolve ambiguous real cutoff ties with legacy topk on one complete row. Mask-only
        # and inactive-query ties cannot change the output and do not need this extra work.
        if keep > k:
            ambiguous = (
                ((vals[:, k - 1] == vals[:, k]) & (vals[:, k] != sentinel) & qm.reshape(-1).bool()).nonzero().flatten()
            )
            for r in ambiguous:
                # torch.topk's arbitrary boundary-tie membership is reproduced on a full row,
                # never on the full query-token by document-token score matrix.
                full_row = qf[r : r + 1] @ df.T
                full_row.masked_fill_(~dm, sentinel)
                rv, ri = full_row.topk(k, dim=-1, sorted=False)
                vals[r, :k], ids[r, :k] = rv[0], ri[0]
        vals, ids = vals[:, :k], ids[:, :k]
        valid = dm[ids] & qm.reshape(-1, 1).bool()
        docids = ids // L
        maxima = vals.new_full((Q * T, B), sentinel)
        maxima.scatter_reduce_(1, docids, vals, reduce="amax", include_self=True)
        # max(dim) assigns gradients only to the first winning token, not equally to ties.
        wins = valid & (vals == maxima.gather(1, docids))
        winner_ids = torch.full((Q * T, B), len(df), dtype=torch.long, device=q.device)
        winner_ids.scatter_reduce_(1, docids, torch.where(wins, ids, len(df)), reduce="amin", include_self=True)
        row, col = (winner_ids < len(df)).nonzero(as_tuple=True)
        token = winner_ids[row, col]
        alpha = (winner_ids < len(df)).reshape(Q, T, B)
    # Differentiable dot products only for per-query-token, per-document winners.
    dot = torch.bmm(qf[row].unsqueeze(1), df[token].unsqueeze(-1)).reshape(-1).float()
    out = torch.zeros(Q * T, B, dtype=torch.float32, device=q.device).index_put((row, col), dot)
    out = out.reshape(Q, T, B) * qm.unsqueeze(-1)
    score = out.sum(1) / alpha.float().sum(1).clamp(min=1)
    return _fill_empty_document_scores(score, ~dm.reshape(B, L).any(-1))
