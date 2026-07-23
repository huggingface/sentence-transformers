# Multi-Vector-Encoder Evaluators

All evaluators live in `sentence_transformers.multi_vector_encoder.evaluation`. They mirror the bi-encoder evaluators but use MaxSim scoring end-to-end.

## Top-line pick

| Task | Evaluator |
|---|---|
| Retrieval on a suite of common English IR benchmarks | `MultiVectorNanoBEIREvaluator` |
| Custom retrieval corpus (your own docs / queries / qrels) | `MultiVectorInformationRetrievalEvaluator` |
| Distillation from a cross-encoder teacher | `MultiVectorDistillationEvaluator` |
| Reranking a fixed candidate list per query | `MultiVectorRerankingEvaluator` |
| Triplet accuracy (does anchor score positive > negative?) | `MultiVectorTripletEvaluator` |

**Default recommendation for training**: `MultiVectorNanoBEIREvaluator` on a subset of NanoBEIR datasets (e.g. `["msmarco", "nq", "fiqa2018"]`) during training, and the full suite at end-of-run. Cheap, well-calibrated, and the metric key format is stable.

## `MultiVectorNanoBEIREvaluator`

Runs the 13 NanoBEIR sub-datasets and returns nDCG@10 averaged across them.

```python
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

evaluator = MultiVectorNanoBEIREvaluator(
    dataset_names=["msmarco", "nq", "fiqa2018"],  # subset for training-time evals
    batch_size=16,
)
```

**primary_metric**: `"NanoBEIR_mean_maxsim_ndcg@10"` (unless you pass `aggregate_key="..."`, in which case it becomes `"NanoBEIR_{aggregate_key}_maxsim_ndcg@10"`). Note that `NanoBEIREvaluator` composes its display name from `aggregate_key` (and optionally `truncate_dim`), it does not accept a `name` kwarg.

**`metric_for_best_model` key**: `"eval_NanoBEIR_mean_maxsim_ndcg@10"` (add the `eval_` prefix, since Trainer adds it to evaluator metrics).

- `dataset_names`: list of NanoBEIR sub-datasets. Full list: `msmarco`, `nq`, `fiqa2018`, `hotpotqa`, `nfcorpus`, `arguana`, `scidocs`, `climatefever`, `dbpedia`, `fever`, `quoraretrieval`, `scifact`, `touche2020`. Note it is `quoraretrieval`, not `quora`: an invalid name raises at construction.
- `batch_size`: also drives corpus encoding, so scale to fit memory.
- English-only. For non-English retrieval, use `MultiVectorInformationRetrievalEvaluator` with your own corpus.

## `MultiVectorInformationRetrievalEvaluator`

Full IR evaluator over a corpus + queries + qrels you supply.

```python
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator

evaluator = MultiVectorInformationRetrievalEvaluator(
    queries={qid: text, ...},                       # dict of query_id to query_text
    corpus={did: text, ...},                        # dict of doc_id to doc_text
    relevant_docs={qid: {did1, did2, ...}, ...},    # qid to set of relevant doc_ids
    batch_size=16,
    name="my-eval",                                 # optional, prefixes the metric key
    write_csv=True,                                 # persists per-eval metrics
    ndcg_at_k=[10],                                 # customize k-values if you need @20/@100 etc.
)
```

**primary_metric**: `"maxsim_ndcg@10"` by default (built from `score_function="maxsim"` and `max(ndcg_at_k)`).

**`metric_for_best_model` key** with `name="my-eval"`:  `"eval_my-eval_maxsim_ndcg@10"`.

- `main_score_function` overrides the score used in `primary_metric` if you set a non-default. Default is `None`, which resolves to `maxsim` from the model's `similarity_fn_name` at call time. For MVE just leave it as `None`.
- Reports MRR@k, nDCG@k, Recall@k, Precision@k, MAP@k for the k-values you pass (`mrr_at_k`, `ndcg_at_k`, etc.).
- `document_chunk_size` (default 32) chunks the corpus during scoring. This is the primary memory knob for MaxSim: intermediates are `(num_queries, document_chunk_size, q_tokens, d_tokens)`, so halving it halves the peak footprint per chunk. Prefer dropping this before `batch_size` when hitting eval-time OOM.

## `MultiVectorDistillationEvaluator`

Regression of student MaxSim scores against teacher scores over a held-out set of `(query, doc, teacher_score)` rows. Use it when you're distilling.

**primary_metric**: `"spearman"` (correlation of student scores with teacher scores).

**`metric_for_best_model` key** with `name="my-eval"`: `"eval_my-eval_spearman"`.

- Pair with `MultiVectorMarginMSELoss` or `MultiVectorDistillKLDivLoss` during training.
- Higher is better (student ranking matches teacher).

## `MultiVectorRerankingEvaluator`

Reranks a fixed list of candidates per query. Same shape as the bi-encoder `RerankingEvaluator`, MaxSim-scored.

- **Data**: `samples = [{"query": ..., "positive": [...], "negative": [...]} , ...]`.
- Reports MAP, MRR@10, nDCG@10 by default.

## `MultiVectorTripletEvaluator`

Fraction of `(anchor, positive, negative)` triplets where `sim(anchor, positive) > sim(anchor, negative)`. Cheap sanity signal, not what you want to gate a real IR release on.

## Named-evaluator metric key format (universal)

If you pass `name="..."` to any of these, the metric key format is:

```
eval_{name}_{primary_metric}
```

If you don't pass `name`, the key is just `eval_{primary_metric}`. The trainer's `metric_for_best_model=...` must match exactly, or `load_best_model_at_end` selects the wrong checkpoint. **Run the evaluator once before training** to observe the exact key format: `primary_metric` is `None` until the first `evaluator(model)` call, which composes it from the resolved score function and the `ndcg_at_k` you passed, then prefixes `name`. The production template does exactly this before building `TrainingArguments`.

## Gotchas

- **Metric key mismatch on `metric_for_best_model`**: silent failure. Training runs to completion and `load_best_model_at_end` picks a stale checkpoint. Always print `evaluator(model)` output once before starting the trainer and confirm the key format.
- **Eval-time OOM**: MaxSim scoring builds `(num_queries, document_chunk_size, q_tokens, d_tokens)` intermediates. Drop `document_chunk_size` first (default 32, halving it halves the per-chunk peak), then `batch_size`. NanoBEIR corpora can be small enough that oversized `batch_size` bites before you'd expect.
- **`MultiVectorNanoBEIREvaluator` with a Non-English base**: it's English-only. You'll get zero or garbage scores. Use `MultiVectorInformationRetrievalEvaluator` with an in-domain corpus.
- **Distillation eval on the same data as training**: measures memorization, not generalization. Hold out a separate `(query, doc, teacher_score)` split.
