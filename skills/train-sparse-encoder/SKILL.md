---
name: train-sparse-encoder
description: Train or fine-tune SparseEncoder (SPLADE) sparse embedding models with the sentence-transformers library — sparse vectors over the vocabulary (interpretable token weights, inverted-index-compatible with Elasticsearch / OpenSearch / Lucene). Covers SPLADE architecture (Transformer + SpladePooling + FLOPS regularization), `SpladeLoss`-wrapped sparse losses, hard-negative mining, `SparseNanoBEIREvaluator` with sparsity tracking, distillation from cross-encoders, PEFT / LoRA, prompts, and Hugging Face Hub publishing. Use for any SPLADE / learned-sparse retriever training; for dense retrieval use `train-sentence-transformer`, for rerankers use `train-cross-encoder`.
---

# Train a SparseEncoder (SPLADE)

Fine-tune or train a `SparseEncoder` — a learned-sparse retrieval model that outputs a sparse vector over the vocabulary (interpretable token-level weights, compatible with inverted-index backends like Elasticsearch / OpenSearch / Lucene). This skill targets the SPLADE architecture: Transformer + SpladePooling on an MLM head. For dense retrieval use `train-sentence-transformer`; for rerankers use `train-cross-encoder`.

## Before Training

Defaults (override only if the user specifies otherwise): **local execution** (pitch HF Jobs only if local hardware can't fit the job — see Prerequisites below), **single run** (after it completes, propose experimentation if the user would benefit; iteration rules in `## Experimentation` near the end of this skill), **public Hub push at end-of-run wrapped in try-except** (see `## Saving, Hub Push`).

### End-of-run verdict block (mandatory)

Every training script must emit one VERDICT line a monitor / log scraper can pick up. Capture the pre-training baseline once (already done in the Quick Start `evaluator(model)` pass), then at end of run. Log **both** the retrieval metric and sparsity (`query_active_dims` / `document_active_dims`); a high nDCG with collapsed sparsity is not a win.

```python
result = evaluator(model)
score = result[evaluator.primary_metric]
delta = score - baseline_eval                         # baseline_eval was captured before training
verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
logging.info(
    f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f} "
    f"| query_active={result.get('query_active_dims', 'n/a')} doc_active={result.get('document_active_dims', 'n/a')}"
)
```

For experimentation mode (multiple iterations vs a stronger known baseline) and the `logs/experiments.md` append, see `## Experimentation`. Note: SPLADE's pre-training baseline is uninformative since fill-mask bases score near zero on retrieval — for any non-trivial SPLADE run, override `baseline_eval` with a published comparator score (see Experimentation).

## Key Directives

1. **Load the model in fp32; autocast bf16/fp16 at runtime.** Never pass `torch_dtype=torch.bfloat16` to `SparseEncoder(...)` — it puts Adam state in bf16 and silently degrades quality. Use `bf16=True` in TrainingArguments, and wrap out-of-trainer `evaluator(model)` calls in `torch.autocast("cuda", dtype=torch.bfloat16)` (mandatory with FA2; otherwise just a speedup).
2. **Verify VRAM fits; on Windows, overflow is silent.** When VRAM hits ~95% on Windows, the NVIDIA driver silently spills into host RAM via "shared memory" and training throughput drops 5-10x with no error (`nvidia-smi` shows 100% util either way). Run 1-2 steps and watch `tokens/sec`; if dropping, cut `per_device_train_batch_size` or `model.max_seq_length` first. `gradient_checkpointing=True` is a fallback (incompatible with `Cached*` losses).
3. **Base model needs an MLM head.** SPLADE uses the masked-LM head to produce token-level weights. Start from `distilbert/distilbert-base-uncased`, `bert-base-uncased`, or an existing `naver/splade-*` — **not** a pure `AutoModel` without MLM.
4. **Always wrap your base loss in `SpladeLoss`** for SPLADE. Without it you get no FLOPS regularization and embeddings end up dense, defeating the point. See `references/losses.md`.
5. **Monitor sparsity during training.** `SparseNanoBEIREvaluator` reports `query_active_dims` and `document_active_dims`. Target: queries ~30–50, docs ~150–250. Abort if sparsity collapses toward 0 or explodes toward vocab size — the regularizer is mistuned.
6. **`BatchSamplers.NO_DUPLICATES`** for `SparseMultipleNegativesRankingLoss` — duplicates create false negatives.
7. **Always include an evaluator and run it once on the base model before training** (under autocast for speed) — a fill-mask base scores near zero, so this both gives the baseline and confirms the pipeline works end-to-end. Set `load_best_model_at_end=True` + `metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10"`.
8. **Silence noisy HTTP loggers + write logs to `logs/{run_name}.log`** at the top of every script: `for L in ("httpx", "httpcore", "huggingface_hub", "urllib3"): logging.getLogger(L).setLevel(logging.WARNING)`. Otherwise HF download URLs flood the agent's context. Tee logs to a file so they survive the session.
9. **Default Hub push: end-of-run, public, wrapped in try-except** (matches the repo's example-script idiom; auth or naming failures don't lose the local checkpoint). On HF Jobs (ephemeral env) ALSO enable in-trainer push: `push_to_hub=True`, `hub_model_id=RUN_NAME`, `hub_strategy="every_save"`, plus `secrets={"HF_TOKEN": "$HF_TOKEN"}` in the job spec — otherwise weights are lost on timeout.

**Attention backend**: pass `model_kwargs={"attn_implementation": "flash_attention_2"}` for FA2 (needs `pip install flash-attn`, Ampere+ only) or `"kernels-community/flash-attn"` (precompiled via `pip install kernels`, easier install) when training long sequences or large batches. Default `sdpa` is fine for short sequences. Don't combine with `torch_dtype="bfloat16"` — let `bf16=True` autocast handle dtype (Directive 1).

## Prerequisites

```bash
pip install "sentence-transformers[train]>=5.0"
pip install trackio                                    # optional tracker; or wandb / tensorboard / mlflow
hf auth login                                          # or set HF_TOKEN with write scope (for Hub push)
```

Dataset must load via `datasets.load_dataset(...)`. Same formats as bi-encoder (pairs, triplets, scored pairs). See `references/dataset_formats.md`. GPU strongly recommended; see `references/hardware_guide.md`.

**Execution modes**: local (`python train.py`) is default. HF Jobs (`hf_jobs("uv", {...})`) scales out, requires a Pro/Team/Enterprise plan, and **mandates Hub push** — see `references/hf_jobs_execution.md`.

## Quick Start

Train a SPLADE model from DistilBERT on GooAQ `(question, answer)` pairs.

```python
from datasets import load_dataset
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss

model = SparseEncoder("distilbert/distilbert-base-uncased")

full = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
split = full.train_test_split(test_size=1_000, seed=12)
train_dataset, eval_dataset = split["train"], split["test"]

loss = SpladeLoss(
    model,
    loss=SparseMultipleNegativesRankingLoss(model),
    query_regularizer_weight=5e-5,
    document_regularizer_weight=3e-5,
)
evaluator = SparseNanoBEIREvaluator()
baseline_eval = evaluator(model)[evaluator.primary_metric]   # baseline: fill-mask base scores near zero, confirms pipeline works

args = SparseEncoderTrainingArguments(
    output_dir="models/distilbert-splade-gooaq",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0.1,
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=f"eval_{evaluator.primary_metric}",   # robust to evaluator swap; sparse defaults to a dot-product key
    greater_is_better=True,
    run_name="distilbert-splade-gooaq",
)

trainer = SparseEncoderTrainer(
    model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
    loss=loss, evaluator=evaluator,
)
trainer.train()
model.save_pretrained("models/distilbert-splade-gooaq/final")

try:
    model.push_to_hub("distilbert-splade-gooaq")        # public by default; uses your authenticated user
except Exception:
    import traceback
    logging.error(f"Hub push failed:\n{traceback.format_exc()}")
```

**Before committing to a long run**, smoke-test by adding `max_steps=1` to the training args and slicing the dataset (`train_dataset = train_dataset.select(range(10))`). Catches column-shape, dtype, and `SpladeLoss`-wrap issues in seconds.

See `scripts/train_example.py` for the full production template (with trackio, Hub push, best-checkpoint loading).

## Base Model Selection

SPLADE requires a fill-mask / `AutoModelForMaskedLM`-compatible checkpoint. Encoder-only MLM models work out of the box; decoder LLMs do **not**.

Sparse leaderboards rotate; don't trust any hardcoded "best" pick. Discover current options live — run **both** sort orders since most-downloaded surfaces proven options and trending surfaces recent SOTA that may not have download volume yet:

```bash
hf models list --filter sentence-transformers --filter sparse-encoder --sort downloads --limit 20
hf models list --filter sentence-transformers --filter sparse-encoder --sort trending  --limit 20

# Optional language narrowing (not all multilingual models tag each language, so missing matches doesn't mean the model can't handle that language — re-run without the filter to compare):
hf models list --filter sentence-transformers --filter sparse-encoder --filter <language-code> --sort trending --limit 20

hf models card <id> --text                        # confirm SPLADE arch + MLM head + languages
```

**Shape guidance:**

- **Continue from an existing SPLADE** beats fresh fill-mask base + 100k-500k pairs. Common namespaces as of 2026-Q2 (verify against the live discovery commands above — the field rotates):
  - English: `naver/splade-*` (the canonical family — `splade-cocondenser-ensembledistil`, `splade-v3`, `splade-v3-distilbert`), `opensearch-project/opensearch-neural-sparse-encoding-*` (incl. `-doc-v2-distill`, `-doc-v3-distill` / `-doc-v3-gte` for newer OpenSearch variants), `prithivida/Splade_PP_en_v*` (Splade++), `ibm-granite/granite-embedding-30m-sparse`.
  - Multilingual: `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1`.
- **Fresh-start English** (≥500k pairs): any encoder with an MLM head — `distilbert/distilbert-base-uncased`, `google-bert/bert-base-uncased`. Pure `AutoModel` checkpoints without MLM won't work (Directive 3). Discover MLM bases via `hf models list --filter fill-mask --sort downloads --limit 20`.
- **Fresh-start multilingual**: `FacebookAI/xlm-roberta-base` (has MLM head). For other multilingual MLM bases: `hf models list --filter fill-mask --filter <language-code> --sort downloads --limit 20`.

**Minimum dataset:** 500k+ triplets (with mined hard negatives) for a competitive SPLADE model. For domain adaptation on an existing SPLADE, 50k+ triplets is often enough.

## Dataset Preparation and Validation

**Column-matching rules** (the #1 silent training failure):

1. The label column must be named exactly `label`, `labels`, `score`, or `scores`. Any column with one of those names IS treated as the label, even unintentionally (e.g. a stray retrieval-score column from a previous step).
2. All non-label columns are inputs; column **order** matters, names don't. The first N columns map to the loss's N expected inputs.

Reshape with `dataset.rename_column(old, new)` / `select_columns([...])` / `remove_columns([...])` to fit. To inspect a dataset's columns / a few rows quickly without `load_dataset(...)`: `hf datasets sql "SELECT * FROM 'hf://datasets/<id>/<split>' LIMIT 5"` (DuckDB-backed; see `references/dataset_formats.md`).

For contrastive sparse training, mine hard negatives:

```bash
python scripts/mine_hard_negatives.py \
    --dataset sentence-transformers/gooaq \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --num-negatives 5 \
    --output-path data/gooaq-with-hard-negatives
```

**Cache mined hard negatives** so reruns skip the ~1hr mining pass (per 500k pairs even with FAISS). The CLI script writes to `--output-path` for free; in-script Python `mine_hard_negatives()` calls re-mine on every run unless wrapped:

```python
import os
from datasets import load_from_disk

CACHE = f"data/{RUN_NAME}-hard-negatives"
if os.path.isdir(CACHE):
    mined = load_from_disk(CACHE)
else:
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    mined = mine_hard_negatives(...)           # your mining call
    mined.save_to_disk(CACHE)
    del retriever                              # free retriever VRAM before training
    torch.cuda.empty_cache()
```

Delete the cache directory to remine.

Full details in `references/dataset_formats.md`.

## Loss Selection (top picks)

**Always wrap in `SpladeLoss` for SPLADE architecture.** `SpladeLoss` adds FLOPS regularization on top of the inner loss.

| Data shape | Recommended loss |
|---|---|
| `(anchor, positive)` or triplet, SPLADE | `SpladeLoss(loss=SparseMultipleNegativesRankingLoss(model), ...)` |
| Same, want effective batch size 256+ | `CachedSpladeLoss(...)` |
| `(text1, text2, score)` labeled pairs | `SparseCoSENTLoss` or `SparseCosineSimilarityLoss` |
| Distillation from cross-encoder teacher | `SparseMarginMSELoss` (wrap in `SpladeLoss`) |
| Listwise distillation | `SparseDistillKLDivLoss` |
| Explicit triplet | `SparseTripletLoss` |

**Tuning FLOPS regularization:**

```python
loss = SpladeLoss(
    model,
    loss=SparseMultipleNegativesRankingLoss(model),
    query_regularizer_weight=5e-5,        # queries should be sparser → higher weight
    document_regularizer_weight=3e-5,     # docs can be denser (more terms)
)
```

- Typical range: 1e-5 to 1e-4. Higher = sparser, lower recall; lower = denser, possibly better recall.
- Query weight should be **higher** than document weight (queries have fewer meaningful terms).
- `SparseEncoderTrainer` auto-registers a `SpladeRegularizerWeightSchedulerCallback` whenever the loss is a `SpladeLoss`, ramping weights from 0 to target over the first ~33% of training. The default ramp shape is `SchedulerType.QUADRATIC` (slower at first, then accelerating), not linear; pass `scheduler_type=SchedulerType.LINEAR` to the callback if you want a linear ramp. The ramp length and shape live on the callback (`SpladeRegularizerWeightSchedulerCallback(loss=..., warmup_ratio=..., scheduler_type=...)`), not on `SpladeLoss` itself; to override, instantiate the callback yourself and pass it via `callbacks=[...]`. Starting with full regularization at step 0 kills learning, so don't disable it without a reason.

Full catalog of sparse losses in `references/losses.md`.

## Evaluator Selection

| Task | Evaluator |
|---|---|
| Retrieval (nDCG, MRR) + sparsity tracking | `SparseNanoBEIREvaluator` |
| Retrieval on your own corpus | `SparseInformationRetrievalEvaluator` |
| STS / continuous similarity | `SparseEmbeddingSimilarityEvaluator` |
| Binary classification | `SparseBinaryClassificationEvaluator` |
| Reranking (from retrieval candidates) | `SparseRerankingEvaluator` |
| Hybrid BM25 + sparse retrieval | `ReciprocalRankFusionEvaluator` |
| MSE vs. teacher (distillation) | `SparseMSEEvaluator` |

**If you have held-out data from the training corpus**, build an in-domain `SparseInformationRetrievalEvaluator` from it and wrap with `SparseNanoBEIREvaluator` in `SequentialEvaluator`. Use the in-domain evaluator as `metric_for_best_model`; NanoBEIR alone measures generalization to a different corpus, and for SPLADE the distribution-shift penalty there often hides real in-domain signal.
| Triplet validation (`anchor`, `positive`, `negative`) | `SparseTripletEvaluator` |
| Cross-lingual translation alignment | `SparseTranslationEvaluator` |

Note: all sparse evaluators default to **dot-product similarity** (cosine is less meaningful on sparse vectors). See `references/evaluators.md`.

## Training Arguments

Key knobs:

- **Duration**: 1 epoch for large datasets; extend if the FLOPS scheduler hasn't converged (the regularizer ramp takes the first ~33% of training to reach target by default).
- **Batch size**: 32–64 typical for DistilBERT SPLADE — heavier than bi-encoder at the same batch because of the vocab-sized output.
- **In-batch negatives are per-device by default.** Single-GPU batch 32 = 31 negatives per anchor; 4× DDP at per-device 32 = still 31 unless you pass `gather_across_devices=True` to `SparseMultipleNegativesRankingLoss` (or its `Cached` variant). With it on 4× DDP at per-device 32 you get 127 negatives. Single-GPU runs ignore the flag.
- **Learning rate**: `2e-5` (full fine-tune), `1e-4+` (LoRA).
- **LR scaling on batch change**: when effective batch size shifts (more GPUs, larger `per_device_batch`, more `gradient_accumulation_steps`, or moving from local to cluster), center the new LR sweep at `lr_new = lr_old * sqrt(batch_ratio)`. Linear scaling is too aggressive for fine-tuning; sqrt is the safer default.
- **Sampler**: `BatchSamplers.NO_DUPLICATES` for `SparseMultipleNegativesRankingLoss`.
- **Best checkpoint**: `eval_strategy="steps"`, align `save_steps` with `eval_steps`, `load_best_model_at_end=True`, `metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10"`.
- **Live progress dashboard**: with `report_to="trackio"`, log `f"https://huggingface.co/spaces/{whoami().get('name')}/trackio"` right after building the trainer so the user can watch live. The Space auto-creates on the first run with a valid `HF_TOKEN`.

Full arg treatment + tracker setup: `references/training_args.md`.

## Running the Training

Use a PEP 723 header so the same script runs locally, via `uv run`, or on HF Jobs:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers[train]>=5.0", "trackio"]
# ///
```

- **Local**: `python train.py` or `uv run train.py`.
- **Multi-GPU**: `accelerate launch train.py`. No code changes.
- **HF Jobs**: `hf_jobs("uv", {...})` + `secrets={"HF_TOKEN": "$HF_TOKEN"}` + `push_to_hub=True` + `hub_strategy="every_save"`. Full playbook: `references/hf_jobs_execution.md`.

## Saving, Hub Push, Post-Training Evaluation

`model.save_pretrained(...)` writes the full folder with an auto-generated `README.md` (model card). **Hub push defaults to public, end-of-run, wrapped in try-except** so a Hub failure doesn't lose the local checkpoint:

```python
try:
    model.push_to_hub(RUN_NAME)                 # uses your authenticated user; public by default
except Exception:
    import traceback
    logging.error(f"Hub push failed:\n{traceback.format_exc()}")
```

If the user wants private, pass `private=True` to `push_to_hub`. To skip Hub push entirely, drop the call; the local checkpoint at `output_dir/final/` is unaffected.

For HF Jobs (ephemeral env), ALSO enable in-trainer push so checkpoints survive timeout: pass `push_to_hub=True`, `hub_model_id=RUN_NAME`, `hub_strategy="every_save"` in TrainingArguments and `secrets={"HF_TOKEN": "$HF_TOKEN"}` in the job spec. The four `hub_strategy` values: `"every_save"` (each checkpoint, mandatory for HF Jobs), `"end"` (final only), `"checkpoint"` (latest only, overwrite), `"all_checkpoints"` (each as a separate commit).

**Post-training evaluation** in three escalating tiers:

| Level | Tool | Time | When |
|---|---|---|---|
| Quick | `SparseNanoBEIREvaluator` on a held-out test split, recording **both** retrieval metrics AND `query_active_dims` / `document_active_dims` | seconds to minutes | Every run |
| Benchmark | Full BEIR / domain-specific IR (also via `mteb.evaluate(...)`) | 30 min to several hours | Before shipping |
| A/B | Production traffic replay, CTR / conversion | days to weeks | Before ramping traffic |

A high nDCG with collapsed sparsity is not a win; always report active-dim counts alongside the retrieval metric.

## Common Failure Modes

| Symptom | First thing to try |
|---|---|
| SPLADE embeddings end up dense (>1000 active dims) | Missing `SpladeLoss` wrapper, or regularizer weights too low. Set `query_regularizer_weight=5e-5`, `document_regularizer_weight=3e-5` minimum. |
| Model outputs all zeros | Regularizer too high (>1e-4), or training started with full regularization from step 0 (override the built-in scheduler at your peril). |
| OOM | Lower `per_device_train_batch_size`; switch to `CachedSpladeLoss`; enable `gradient_checkpointing` (incompatible with `Cached*` — pick one). |
| Metrics stuck | Check column/loss match (label column must be `label`/`labels`/`score`/`scores`); base model lacks MLM head; mine hard negatives. |
| `metric_for_best_model` mismatch | For sparse, the key is `eval_NanoBEIR_mean_dot_ndcg@10` (dot, not cosine). |
| Training crashed / want to resume | `trainer.train(resume_from_checkpoint=True)` auto-detects the latest checkpoint in `output_dir`; pass an explicit path to resume from a specific step. `IterableDataset` iteration order is **not** preserved across resumption — handle streaming-dataset positioning yourself. |
| Hub push 401/403 | `hf auth whoami`; token needs **write** scope. |

Full catalog in `references/troubleshooting.md`.

## Experimentation

After a single run completes, propose iteration if it would benefit the user (phrasing like "beat baseline X", "see how high you can push it", weak/marginal verdict). When iterating:

- **Change one variable per run.** Multi-variable deltas are unattributable.
- **Kill underperforming runs quickly** (don't wait for `EarlyStoppingCallback` after most of the budget is gone):
  - Baseline regression: first eval below the pre-training baseline → kill, setup is broken.
  - Plateau: 3 consecutive evals within ±0.5% and below target → kill.
  - Loss explosion: `nan` or >2× the baseline minimum → kill.
  - Sparsity collapse: `query_active_dims` → 0 or → vocab size → kill, FLOPS regularizer is mistuned.
  Save the partial checkpoint anyway: `trainer.save_model(...)`.
- **Stronger comparator (recommended for SPLADE)**: SPLADE's pre-training baseline is uninformative since fill-mask bases score near zero. Override `baseline_eval` with a published comparator score (e.g. `naver/splade-v3` on NanoBEIR dot nDCG@10) so the verdict grades against a known model. The +0.005 WIN threshold stays the same.
- **`logs/experiments.md` table**: append a row from the verdict block after every run so the table survives the chat session.

  ```python
  import os
  LOG = "logs/experiments.md"
  os.makedirs("logs", exist_ok=True)
  new_file = not os.path.isfile(LOG)
  with open(LOG, "a") as f:
      if new_file:
          f.write("| run_name | base_model | loss | data | key_change | best_metric | delta | verdict | notes |\n|---|---|---|---|---|---|---|---|---|\n")
      f.write(f"| {RUN_NAME} | <base> | <loss> | <data> | <key_change> | {score:.4f} | {delta:+.4f} | {verdict} | <notes> |\n")
  ```
- **Decision log**: if a result contradicts prior intuition, append a `## YYYY-MM-DD — DECISION:` paragraph (decision / evidence / hypothesis / status) to `DECISIONS.md` so future-you doesn't retry the same rejected lever.

## Advanced Patterns

| Pattern | Reference |
|---|---|
| **Distillation from cross-encoder** (`SparseMarginMSELoss` in `SpladeLoss`) | `scripts/train_distillation_example.py` |
| **Prompts / instructions** | `references/prompts_and_instructions.md` |
| **PEFT / LoRA adapters** | `../train-sentence-transformer/scripts/train_with_lora_example.py` (docstring covers sparse-encoder setup variant: same `TaskType.FEATURE_EXTRACTION`, swap `SentenceTransformer` -> `SparseEncoder` and the loss) |
| **Hard negative mining** | `scripts/mine_hard_negatives.py` + `references/dataset_formats.md` |
| **Multi-task / multi-dataset** | `../train-sentence-transformer/scripts/train_multi_dataset_example.py` (docstring covers per-dataset losses, single-loss + DatasetDict variant, samplers, gotchas; same trainer plumbing for `SparseEncoderTrainer`) |
| **Hyperparameter search** | `references/training_args.md#hyperparameter-search` |
| **Resume from checkpoint** | `references/training_args.md#resuming-training` |

## Resources

Reference docs in `references/` cover the SparseEncoder loss / evaluator catalogs plus the cross-cutting training args, dataset formats, hardware, HF Jobs, prompts, and troubleshooting material. Example scripts in `scripts/` (each with an editorial-recipe docstring), plus the hard-negative mining CLI at `scripts/mine_hard_negatives.py`. PEFT/LoRA and multi-dataset recipes live in the bi-encoder skill's example scripts (their docstrings cover sparse-encoder variants where they differ — wrap losses in `SpladeLoss` for distillation, swap `SentenceTransformer` -> `SparseEncoder` for PEFT); install `train-sentence-transformer` alongside this skill to read them locally.

Related skills: `train-sentence-transformer` (dense), `train-cross-encoder` (rerankers). External: [SPLADE paper](https://arxiv.org/abs/2107.05720) (anchors the architecture).
