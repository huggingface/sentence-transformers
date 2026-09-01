# Multi-Vector-Encoder Losses (ColBERT / late-interaction)

All losses live in `sentence_transformers.multi_vector_encoder.losses`.

Multi-vector models emit **one embedding per token** and score `(query, document)` via MaxSim (for each query token, take the max similarity to any document token, then sum across query tokens). This is fundamentally different from single-vector cosine, which changes what "temperature" means: MaxSim is an unbounded sum over query-token similarities, so for it a scale near 1.0 makes sense. **The default `scale=1.0` (temperature=1.0) is correct for unnormalized MaxSim: do not copy `scale=20.0` from bi-encoder MNRL.** With MeanMaxSim scoring (`similarity_fct=mean_colbert_scores`, and set `model.similarity_fn_name = "meanmaxsim"` so evaluation matches), each score is divided by its query's token count, so a scale of roughly the average length is a reasonable start, although some works use larger scales.

## Top-line decision table

| You have | Use |
|---|---|
| `(anchor, positive)` or `(anchor, positive, negative)` triplets | `MultiVectorMultipleNegativesRankingLoss` |
| Same, want effective batch size of 128+ | `CachedMultiVectorMultipleNegativesRankingLoss` |
| Cross-encoder teacher scores, `(query, positive, negative, score_diff)` | `MultiVectorMarginMSELoss` |
| Listwise distillation `(query, [doc_1..doc_N], scores)` | `MultiVectorDistillKLDivLoss` |

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
- `similarity_fct=colbert_scores` by default. Pass `XTRScores(top_k=...)` for the sparser XTR-style scoring (from `sentence_transformers.multi_vector_encoder.scoring`). XTR applies to training only: evaluation always scores with MaxSim (see Gotchas).
- `scale=1.0` (temperature=1.0) matches PyLate and is correct for unnormalized MaxSim. **Do NOT copy `scale=20.0` from bi-encoder MNRL.** With MeanMaxSim (`similarity_fct=mean_colbert_scores`), raise the scale to roughly the average query length instead, which is the same objective at the same strength.

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
- `chunk_elements` is the second lever on that same phase, and it cuts the other axis: `score_mini_batch_size` chunks queries, `chunk_elements` chunks documents inside each MaxSim call. Pass it through the scorer, e.g. `similarity_fct=partial(colbert_scores, chunk_elements=1_000_000)`. Scores and gradients are unchanged (it is a pure memory knob), and it composes with `score_mini_batch_size`.
- `gather_across_devices=True` gathers document embeddings across DDP ranks. Enables cross-rank in-batch negatives.

## Distillation losses

Distillation is where multi-vector models learn most efficiently: cross-encoder teachers (e.g. `gte-modernbert-base`) score `(query, doc)` pairs offline, and the student MaxSim model regresses to that signal.

For the standard MS MARCO KD format (LightOn's `ms-marco-en-bge` and similar: `(query_id, document_ids, scores)` with separate `queries` / `documents` splits), you can use `sentence_transformers.util.resolve_ids` to join the IDs against the text splits at load time. It's a factory that returns a batched transform: pass a mapping from each input ID column to its lookup dataset, e.g. `dataset.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=32))` (lazy, no caching) or via `dataset.map(..., batched=True, remove_columns=["query_id", "document_ids"])` (eager, cached: `map` keeps input columns unless told to drop them, so pass the ID columns to `remove_columns`), yielding `(query, document_1, ..., document_32, scores)` rows ready for `MultiVectorDistillKLDivLoss`. The same mapping shape covers triplets with IDs (`{"query_id": queries, "positive_id": documents, "negative_id": documents}`) and ID-only contrastive datasets without a `scores` column.

### `MultiVectorMarginMSELoss`

Regress the **margin** between positive and negative MaxSim scores against the teacher's margin.

```python
loss = MultiVectorMarginMSELoss(model=model)
```

- **Data**: `(query, positive, negative, score_diff)` where `score_diff = teacher_score(query, positive) - teacher_score(query, negative)`. Raw teacher scores `[score(q, pos), score(q, neg_1), ...]` also work, converted to margins internally.
- Popular recipe from PyLate / colpali-engine.
- Teacher scores are precomputed once, stored as the label column. The loss does not run the teacher inline.
- Same `similarity_fct` convention as the other losses, but pairwise-shaped: defaults to `colbert_scores_pairwise`, or pass `xtr_scores_pairwise` for XTR.

### `MultiVectorDistillKLDivLoss`

Listwise KL-div: student's softmax distribution over N candidates should match the teacher's.

```python
loss = MultiVectorDistillKLDivLoss(model=model)
```

- **Data**: `(query, document_1, ..., document_N, scores)` where `scores` is a list of N teacher scores per row. One column per candidate document (the standard positional multi-column convention), and the label column must use a recognized name (`label`, `labels`, `score`, `scores`). `resolve_ids` expands a stored ID list into the numbered columns.
- Stronger training signal than `MarginMSE` when you have full `N`-way teacher scores (not just positive/negative margins).
- `similarity_fct` defaults to `colbert_kd_scores`, the listwise KD variant, not the `colbert_scores` used by the MNRL family. `XTRKDScores` is the XTR counterpart.
- `temperature` softens both distributions before the softmax, and can be overridden per side with `student_temperature` / `teacher_temperature`. Set `teacher_temperature` from the spread of your teacher's scores rather than from a value copied out of a paper: a float32 softmax underflows to exact zeros once a row's spread divided by the temperature exceeds about 100, and every candidate that underflows drops out of the KL, which is the ranking information you are distilling. Sharpening `student_temperature` far below the student's own score spread collapses that side too, and once both are one-hot the loss and its gradient are exactly zero. **Start at 1.0 / 1.0 and only move a side you have a measured reason to move.**
- The loss is scaled by the student temperature squared, which keeps gradient magnitudes comparable only for a shared temperature at or above 1.0 (the regime Hinton et al. cover). Below that the factor shrinks the loss instead, by 1e-6 at `student_temperature=0.001`: when combining KD with a contrastive loss, scale the KD weight back up accordingly.
- **OOM**: drop `per_device_train_batch_size` first and raise `gradient_accumulation_steps` to hold the effective batch. Only reduce `N` (candidate-list length) as a last resort, since lowering N changes the experiment.

## Data-shape gotchas

- **Column 0 is always the query.** Losses call `self.model(sf, task="query" if idx==0 else "document")`. Use the collator's `router_mapping` to override if you need a non-standard column layout.
- **Cross-column varlen `T`** is handled: the batch's positive and negative columns can have different token counts (Qwen2-VL family etc.). `stack_padded_token_embeddings` pads to the per-batch max before MaxSim.
- **Mask flow**: MVE reads the SCORING mask from the model OUTPUT dict (`MultiVectorMask` rewrites the input mask). Custom loss code that reads `sentence_features[i]["attention_mask"]` directly instead of `outputs[i]["attention_mask"]` will silently score against skiplisted tokens.

## Gotchas

- **`scale=20.0` copied from bi-encoder MNRL**: misscaled for the unnormalized MaxSim sum. Keep `scale=1.0` there. Only length-normalized MeanMaxSim wants a larger scale, roughly the average query length.
- **Missing `Normalize` at the token level in the pipeline**: `colbert_scores` assumes L2-normalized token embeddings. If your custom pipeline drops the token-level `Normalize`, either add one or pass `normalize_embeddings=True` semantics via a wrapper.
- **`CachedMultiVectorMultipleNegativesRankingLoss` + `gradient_checkpointing=True`**: crash. Pick one.
- **`MultiVectorMarginMSELoss` without precomputed `score_diff`**: label column must be populated from a teacher pass ahead of training. The loss does not compute the teacher inline.
- **Expecting XTR scoring at eval**: `XTRScores` is a train-only `similarity_fct`. XTR takes its top-k across the whole candidate set, so a `(query, document)` pair has no standalone score, which means it cannot be the model's `similarity_fn_name` and the evaluators reject it outright. Evaluation and inference score with MaxSim, also for XTR-trained models. That is by design, not a mismatch to fix. To score a fixed candidate set ad hoc, call `xtr_scores` directly.
- **Distillation with a weak teacher**: multi-vector students easily match a small-model teacher and then plateau. Use a strong cross-encoder (e.g. `gte-modernbert-base`, `mxbai-rerank-large-v2`) for the teacher pass.
