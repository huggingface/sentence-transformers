# Multi-Vector-Encoder Losses (ColBERT / late-interaction)

All losses live in `sentence_transformers.multi_vector_encoder.losses`.

Multi-vector models emit **one embedding per token** and score `(query, document)` via MaxSim (for each query token, take the max similarity to any document token, then sum across query tokens). This is fundamentally different from single-vector cosine, which changes what "temperature" means: MaxSim is an unbounded sum over query-token similarities, so bi-encoder scaling values like `scale=20.0` saturate the softmax here. **The default `scale=1.0` (temperature=1.0) is correct for MaxSim. Do not copy `scale=20.0` from bi-encoder MNRL.**

## Top-line decision table

| You have | Use |
|---|---|
| `(anchor, positive)` or `(anchor, positive, negative)` triplets | `MultiVectorMultipleNegativesRankingLoss` |
| Same, want effective batch size of 128+ | `CachedMultiVectorMultipleNegativesRankingLoss` |
| Cross-encoder teacher scores, `(query, positive, negative, score_diff)` | `MultiVectorMarginMSELoss` |
| Listwise distillation `(query, [doc_1..doc_N], teacher_scores)` | `MultiVectorDistillKLDivLoss` |

Hard-negative mining is essential for competitive results, because random in-batch negatives leave a lot on the table for late-interaction models. See `dataset_formats.md` (Hard-negative mining section) and `../scripts/mine_hard_negatives.py`.

## Contrastive losses

### `MultiVectorMultipleNegativesRankingLoss`

The default late-interaction contrastive loss. In-batch positives plus explicit hard negatives, scored with MaxSim (or XTR).

```python
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss

loss = MultiVectorMultipleNegativesRankingLoss(model=model)  # scale=1.0 (temperature=1.0) is the correct default
```

- **Data**: `(anchor, positive)` or `(anchor, positive, negative_1, ..., negative_n)`. The collator stamps each column's task (column 0 becomes query, others become document). Pass `task=...` on a column to override.
- Set `batch_sampler=BatchSamplers.NO_DUPLICATES` on training args (same reason as bi-encoder MNRL).
- `score_metric=colbert_scores` by default. Pass `XTRScores(k=...)` for the sparser XTR-style scoring (from `sentence_transformers.multi_vector_encoder.scoring`). XTR applies to training only: evaluation always scores with MaxSim (see Gotchas).
- `scale=1.0` (temperature=1.0) matches PyLate and is correct for MaxSim. **Do NOT set `scale=20.0`**, which saturates the softmax and destroys learning.

### `CachedMultiVectorMultipleNegativesRankingLoss`

GradCache variant: chunked embedding forward, cached gradients, and a second re-embedding pass. Decouples per-device batch size from effective in-batch negatives.

```python
loss = CachedMultiVectorMultipleNegativesRankingLoss(
    model=model,
    mini_batch_size=8,           # or use mini_batch_num_tokens=... for token-budgeted packing
    score_mini_batch_size=4,     # optional, smaller trims the transient (Q, Q*N, q_tok, d_tok) buffer
)
```

- **Incompatible with `gradient_checkpointing=True`** (same as every `Cached*` loss).
- `mini_batch_num_tokens=N` packs sequences until N real tokens per mini-batch instead of a fixed `mini_batch_size`. Big win on variable-length data with flash-attention / input flattening.
- `score_mini_batch_size` chunks the SCORING phase (which builds `(Q, Q*N, q_tokens, d_tokens)` intermediates) independently from the embedding phase. Drop it first when hitting OOM in the loss stage.
- `gather_across_devices=True` gathers document embeddings across DDP ranks. Enables cross-rank in-batch negatives.

## Distillation losses

Distillation is where multi-vector models learn most efficiently: cross-encoder teachers (e.g. `gte-modernbert-base`) score `(query, doc)` pairs offline, and the student MaxSim model regresses to that signal.

### `MultiVectorMarginMSELoss`

Regress the **margin** between positive and negative MaxSim scores against the teacher's margin.

```python
loss = MultiVectorMarginMSELoss(model=model)
```

- **Data**: `(query, positive, negative, score_diff)` where `score_diff = teacher_score(query, positive) - teacher_score(query, negative)`.
- Popular recipe from PyLate / colpali-engine.
- Teacher scores are precomputed once, stored as the label column. The loss does not run the teacher inline.
- The scoring override here is `similarity_fct` (defaults to `maxsim_pairwise`), not the `score_metric` used by the MNRL and KLDiv losses.

### `MultiVectorDistillKLDivLoss`

Listwise KL-div: student's softmax distribution over N candidates should match the teacher's.

```python
loss = MultiVectorDistillKLDivLoss(model=model)
```

- **Data**: `(query, [doc_1..doc_N], teacher_scores)`. One query per row with a flattened `N`-way document column and a `(batch, N)` teacher-scores label.
- Stronger training signal than `MarginMSE` when you have full `N`-way teacher scores (not just positive/negative margins).
- `score_metric` defaults to `colbert_kd_scores`, the listwise KD variant, not the `colbert_scores` used by the MNRL family. `XTRKDScores` is the XTR counterpart.
- `temperature` softens both distributions before the softmax, and the loss is scaled by `temperature ** 2` so gradient magnitudes stay comparable across temperatures. `normalize_scores=True` min-max normalises the student scores along the `N` dimension first.
- **OOM**: drop `per_device_train_batch_size` first and raise `gradient_accumulation_steps` to hold the effective batch. Only reduce `N` (candidate-list length) as a last resort, since lowering N changes the experiment.

## Data-shape gotchas

- **Column 0 is always the query.** Losses call `self.model(sf, task="query" if idx==0 else "document")`. Use the collator's `router_mapping` to override if you need a non-standard column layout.
- **Cross-column varlen `T`** is handled: the batch's positive and negative columns can have different token counts (Qwen2-VL family etc.). `stack_padded_token_embeddings` pads to the per-batch max before MaxSim.
- **Mask flow**: MVE reads the SCORING mask from the model OUTPUT dict (`MultiVectorMask` rewrites the input mask). Custom loss code that reads `sentence_features[i]["attention_mask"]` directly instead of `outputs[i]["attention_mask"]` will silently score against skiplisted tokens.

## Gotchas

- **`scale=20.0` copied from bi-encoder MNRL**: saturates the MaxSim softmax so the loss stays stuck and gradients vanish. Keep `scale=1.0`.
- **Missing `Normalize` at the token level in the pipeline**: `colbert_scores` assumes L2-normalized token embeddings. If your custom pipeline drops the token-level `Normalize`, either add one or pass `normalize_embeddings=True` semantics via a wrapper.
- **`CachedMultiVectorMultipleNegativesRankingLoss` + `gradient_checkpointing=True`**: crash. Pick one.
- **`MultiVectorMarginMSELoss` without precomputed `score_diff`**: label column must be populated from a teacher pass ahead of training. The loss does not compute the teacher inline.
- **Expecting XTR scoring at eval**: `XTRScores` is a train-only `score_metric`. XTR takes its top-k across the whole candidate set, so a `(query, document)` pair has no standalone score, which means it cannot be the model's `similarity_fn_name` and the evaluators reject it outright. Evaluation and inference score with MaxSim, also for XTR-trained models. That is by design, not a mismatch to fix. To score a fixed candidate set ad hoc, call `xtr_scores` directly.
- **Distillation with a weak teacher**: multi-vector students easily match a small-model teacher and then plateau. Use a strong cross-encoder (e.g. `gte-modernbert-base`, `mxbai-rerank-large-v2`) for the teacher pass.
