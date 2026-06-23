# Training Overview

Training a multi-vector encoder follows the same workflow as :class:`~sentence_transformers.SparseEncoder`
and :class:`~sentence_transformers.SentenceTransformer`: pick a dataset, pick a loss, configure
:class:`~sentence_transformers.MultiVectorEncoderTrainingArguments`, and call
:class:`~sentence_transformers.MultiVectorEncoderTrainer`. The model produces token-level vectors that are
scored with MaxSim (or XTR-style global top-k) during the loss.

## Dataset formats

The following column conventions are recognized by
:class:`~sentence_transformers.multi_vector_encoder.MultiVectorEncoderDataCollator`. The collator routes
columns by position: the first tokenized column is the *query* side and the rest are the *document* side.
Columns ending in ``_id`` or ``_ids`` are skipped. Override the routing by passing ``router_mapping`` to
the trainer. Each side then applies whatever the model is configured with: optional prompt prefix
(e.g. ``"[Q] "`` / ``"[D] "``), optional ``query_length`` / ``document_length`` truncation,
mask-token query expansion (when ``query_length`` is set), and document-side skiplist masking.

| Format | Columns | Loss |
|---|---|---|
| Pair | `anchor, positive` | :class:`MultiVectorMultipleNegativesRankingLoss` |
| Triplet | `anchor, positive, negative` | :class:`MultiVectorMultipleNegativesRankingLoss` |
| Multi-negative | `anchor, positive, negative_1, ..., negative_n` | :class:`MultiVectorMultipleNegativesRankingLoss` |
| KD (per-row N candidates) | `query, documents, scores` where `documents` is a `list[str]` of size N and `scores` is a `list[float]` of size N | :class:`MultiVectorDistillKLDivLoss` |
| Margin-MSE | `query, positive, negative` (single) or `query, positive, negative_1..k` with `label` of shape `(bs,)` or `(bs, k)` | :class:`MultiVectorMarginMSELoss` |

For KD with externally-stored query / document texts (the typical MS MARCO BGE-distillation flow), use
:class:`~sentence_transformers.multi_vector_encoder.KDProcessing` to resolve IDs against query and document
datasets on the fly.

## Losses

| Loss | Purpose |
|---|---|
| `MultiVectorMultipleNegativesRankingLoss` | In-batch InfoNCE contrastive. The default. |
| `CachedMultiVectorMultipleNegativesRankingLoss` | GradCache version for very large effective batch sizes. |
| `MultiVectorDistillKLDivLoss` | KL divergence to teacher scores (MS MARCO BGE-style distillation). |
| `MultiVectorMarginMSELoss` | Pairwise margin-MSE distillation from a cross-encoder teacher. |

All losses accept a `score_metric=` kwarg. Pass
:class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` to switch from ColBERT-style MaxSim
to XTR-style global top-k scoring without changing the loss.

## Evaluators

| Evaluator | Use case |
|---|---|
| `MultiVectorInformationRetrievalEvaluator` | Standard IR (MRR, NDCG, Recall, Accuracy) on a query / corpus / qrels triple. |
| `MultiVectorNanoBEIREvaluator` | Quick NanoBEIR sweep across 13 small BEIR subsets. |
| `MultiVectorTripletEvaluator` | Triplet accuracy: `MaxSim(a, p) > MaxSim(a, n) + margin`. |
| `MultiVectorDistillationEvaluator` | Spearman correlation + KL divergence vs teacher scores. |
| `MultiVectorRerankingEvaluator` | Reranking quality given a candidate list per query. |

## Choosing the InfoNCE temperature

The default is `scale=1.0` (i.e. `temperature=1.0`), matching PyLate. The dense
:class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` defaults to `scale=20.0` because cosine
similarity is bounded to `[-1, 1]` and needs amplification — but MaxSim is an *unbounded* sum over
query-token similarities (range `~[0, num_query_tokens]`), so it needs none, exactly as the dense loss
recommends `scale=1` for dot-product similarity. A large `scale` here would saturate the softmax and kill
gradients. Raise `scale` (lower `temperature`) only if you have a specific reason to sharpen the distribution.

## Example recipes

See [`examples/multi_vector_encoder/training/`](../../examples/multi_vector_encoder/training/) for full
training scripts:

- `msmarco/training_contrastive.py` — InfoNCE on MS MARCO triplets.
- `msmarco/training_kd.py` — KL distillation from BGE on `lightonai/ms-marco-en-bge`.
- `msmarco/training_cached_contrastive.py` — GradCache for large effective batch sizes.
- `xtr/training_contrastive.py` — XTR-style scoring via `score_metric=XTRScores()`.
