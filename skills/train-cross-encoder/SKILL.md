---
name: train-cross-encoder
description: Train or fine-tune CrossEncoder reranker models with the sentence-transformers library. A CrossEncoder scores `(query, passage)` pairs jointly to re-rank retrieval results from a first-stage retriever (BM25 / bi-encoder). Covers pointwise (BCE, CrossEntropy), pairwise (RankNet), and listwise (LambdaLoss, ListNet, ListMLE, PListMLE) ranking losses, distillation (MarginMSE), hard-negative mining, CrossEncoderNanoBEIREvaluator, PEFT / LoRA, prompts, and Hugging Face Hub publishing. Use for any reranker / pair-classification task; for first-stage retrieval use `train-sentence-transformer`, for sparse retrieval use `train-sparse-encoder`.
---

# Train a CrossEncoder (Reranker)

Fine-tune or train a `CrossEncoder` — a reranker that takes a `(query, passage)` pair and outputs a single relevance score by letting the two inputs attend to each other jointly. Slower than bi-encoders (no pre-computed embeddings) but more accurate per parameter, so the standard pattern is two-stage: bi-encoder retrieves top-100, cross-encoder reranks. For first-stage retrieval use `train-sentence-transformer`; for learned sparse retrieval use `train-sparse-encoder`.

## Before Training

Defaults (override only if the user specifies otherwise): **local execution** (pitch HF Jobs only if local hardware can't fit the job — see Prerequisites below), **single run** (after it completes, propose experimentation if the user would benefit; iteration rules in `## Experimentation` near the end of this skill), **public Hub push at end-of-run wrapped in try-except** (see `## Saving, Hub Push`).

### End-of-run verdict block (mandatory)

Every training script must emit one VERDICT line a monitor / log scraper can pick up. Capture the pre-training baseline once (already done in the Quick Start `evaluator(model)` pass), then at end of run:

```python
score = evaluator(model)[evaluator.primary_metric]
delta = score - baseline_eval                         # baseline_eval was captured before training
verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")
```

For experimentation mode (multiple iterations vs a stronger known baseline) and the `logs/experiments.md` append, see `## Experimentation`. CE rerankers peak mid-training and regress, so `EarlyStoppingCallback(patience>=3)` is mandatory regardless of mode (Directive 8).

## Key Directives

1. **Load the model in fp32; autocast bf16/fp16 at runtime.** Never pass `torch_dtype=torch.bfloat16` to `CrossEncoder(...)` — it puts Adam state in bf16 and silently degrades quality. Use `bf16=True` in TrainingArguments, and wrap out-of-trainer `evaluator(model)` calls in `torch.autocast("cuda", dtype=torch.bfloat16)` (mandatory with FA2; otherwise just a speedup).
2. **Verify VRAM fits; on Windows, overflow is silent.** When VRAM hits ~95% on Windows, the NVIDIA driver silently spills into host RAM via "shared memory" and training throughput drops 5-10x with no error (`nvidia-smi` shows 100% util either way). Run 1-2 steps and watch `tokens/sec`; if dropping, cut `per_device_train_batch_size` or `model.max_seq_length` first. `gradient_checkpointing=True` is a fallback (incompatible with `Cached*` losses).
3. **Pick a loss that matches the data shape** — don't reshape data to fit a preferred loss. See `references/losses.md`.
4. **Hard negatives are essential.** Random negatives teach a reranker nothing. Mine with `scripts/mine_hard_negatives.py` using a retriever model.
5. **`num_labels=1` for BCE; `num_labels>=2` for CrossEntropy.** Mismatch silently produces the wrong behavior. With BCE, set `pos_weight=torch.tensor(num_hard_negatives)` (typical: 5) so positives aren't under-weighted.
6. **For distillation / listwise / pairwise training, set `activation_fn=nn.Identity()`** when constructing `CrossEncoder(...)`. The default `Sigmoid` (with `num_labels=1`) saturates raw logits >5 to ~1.0, which silently destroys eval ranking (training loss looks fine, eval nDCG crashes from e.g. 0.59 to 0.14). Required for `MSELoss`, `MarginMSELoss`, `LambdaLoss`, `RankNetLoss`, `ListNetLoss`, `ListMLELoss`, `PListMLELoss`. Keep the default `Sigmoid` only for `BinaryCrossEntropyLoss`.
7. **Always include an evaluator and run it once on the base model before training** (under autocast for speed) — that's your baseline. Set `load_best_model_at_end=True` + `metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10"` (the `R100` matches the default `rerank_k=100`; e.g. `R50` if you change it).
8. **CE peaks mid-training**, so use `EarlyStoppingCallback(patience>=3)` and don't trust the final checkpoint.
9. **Silence noisy HTTP loggers + write logs to `logs/{run_name}.log`** at the top of every script: `for L in ("httpx", "httpcore", "huggingface_hub", "urllib3"): logging.getLogger(L).setLevel(logging.WARNING)`. Otherwise HF download URLs flood the agent's context. Tee logs to a file so they survive the session.
10. **Default Hub push: end-of-run, public, wrapped in try-except** (matches the repo's example-script idiom; auth or naming failures don't lose the local checkpoint). On HF Jobs (ephemeral env) ALSO enable in-trainer push: `push_to_hub=True`, `hub_model_id=RUN_NAME`, `hub_strategy="every_save"`, plus `secrets={"HF_TOKEN": "$HF_TOKEN"}` in the job spec — otherwise weights are lost on timeout.

**Attention backend**: pass `model_kwargs={"attn_implementation": "flash_attention_2"}` for FA2 (needs `pip install flash-attn`, Ampere+ only) or `"kernels-community/flash-attn"` (precompiled via `pip install kernels`, easier install) when training long sequences or large batches. Default `sdpa` is fine for short sequences. Don't combine with `torch_dtype="bfloat16"` — let `bf16=True` autocast handle dtype (Directive 1).

## Prerequisites

```bash
pip install "sentence-transformers[train]>=5.0"
pip install trackio                                    # optional tracker; or wandb / tensorboard / mlflow
hf auth login                                          # or set HF_TOKEN with write scope (for Hub push)
```

Dataset must load via `datasets.load_dataset(...)`. BCE expects `(query, passage, label)`; listwise expects `(query, passages[], labels[])`. See `references/dataset_formats.md`. GPU strongly recommended; see `references/hardware_guide.md`.

**Execution modes**: local (`python train.py`) is default. HF Jobs (`hf_jobs("uv", {...})`) scales out, requires a Pro/Team/Enterprise plan, and **mandates Hub push** — see `references/hf_jobs_execution.md`.

## Quick Start

Train a reranker with `BinaryCrossEntropyLoss` on GooAQ `(question, answer)` pairs. Mine 5 hard negatives per positive to produce labeled-pair data — `output_format="labeled-pair"` returns `(question, answer, label)`, the columns BCE consumes.

```python
import torch
from datasets import load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.util import mine_hard_negatives

model = CrossEncoder("microsoft/MiniLM-L12-H384-uncased", num_labels=1)

pairs = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
retriever = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
labeled = mine_hard_negatives(
    dataset=pairs,
    model=retriever,
    num_negatives=5,
    range_min=10, range_max=100,
    output_format="labeled-pair",   # (question, answer, label) with label in {0, 1}
    use_faiss=True,
)
split = labeled.train_test_split(test_size=1_000, seed=12)
train_dataset, eval_dataset = split["train"], split["test"]

loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(5.0))   # =num_negatives
evaluator = CrossEncoderNanoBEIREvaluator()
baseline_eval = evaluator(model)[evaluator.primary_metric]   # baseline: measure before training (used by verdict block)

args = CrossEncoderTrainingArguments(
    output_dir="models/minilm-gooaq-ce",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=0.1,
    bf16=True,
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=f"eval_{evaluator.primary_metric}",   # robust to evaluator swap
    greater_is_better=True,
    run_name="minilm-gooaq-ce",
)

trainer = CrossEncoderTrainer(
    model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
    loss=loss, evaluator=evaluator,
)
trainer.train()
model.save_pretrained("models/minilm-gooaq-ce/final")

try:
    model.push_to_hub("minilm-gooaq-ce")            # public by default; uses your authenticated user
except Exception:
    import traceback
    logging.error(f"Hub push failed:\n{traceback.format_exc()}")
```

**Before committing to a long run**, smoke-test by adding `max_steps=1` to the training args and slicing the dataset (`train_dataset = train_dataset.select(range(10))`). Catches column-shape, dtype, and loss-mismatch issues in seconds.

See `scripts/train_example.py` for the full production template (with trackio, Hub push, early stopping).

## Base Model Selection

Reranker leaderboards rotate every few months; don't trust any hardcoded "best" pick. Discover current options live — run **both** sort orders since most-downloaded surfaces proven options and trending surfaces recent SOTA that may not have download volume yet:

```bash
hf models list --filter sentence-transformers --filter text-ranking --sort downloads --limit 20
hf models list --filter sentence-transformers --filter text-ranking --sort trending  --limit 20

# Optional language narrowing (not all multilingual models tag each language, so missing matches doesn't mean the model can't handle that language — re-run without the filter to compare):
hf models list --filter sentence-transformers --filter text-ranking --filter <language-code> --sort trending --limit 20

hf models card <id> --text                        # confirm training data, license, languages, max_seq_length
```

Cross-check the [MTEB Reranking leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (pick the language / task tab matching your use case) before committing to a multi-hour run.

**Shape guidance:**

- **Continue from an existing reranker** beats fresh-start + 100k–500k pairs in most domains; default to this unless you have a strong reason otherwise. Common namespaces as of 2026-Q2 (verify against the live discovery commands above — the field rotates):
  - English encoder rerankers: `cross-encoder/ms-marco-*`, `BAAI/bge-reranker-*`, `mixedbread-ai/mxbai-rerank-*-v1` / `-v2`, `Alibaba-NLP/gte-reranker-modernbert-*`, `ibm-granite/granite-embedding-reranker-english-*`.
  - Multilingual encoder rerankers: `cross-encoder/mmarco-*`, `BAAI/bge-reranker-v2-m3`, `Alibaba-NLP/gte-multilingual-reranker-*`, `ibm-granite/granite-embedding-reranker-multilingual-*`.
  - Decoder LLM rerankers (multilingual; `num_labels=1` last-token-style scoring): `Qwen/Qwen3-Reranker-*` (0.6B / 4B / 8B), `Qwen/Qwen3-VL-Reranker-*` (multimodal).
- **Fresh-start** is right when (a) you have a strong domain-fit reason (e.g. PubMedBERT for biomedical) AND (b) ≥500k labeled pairs after mining. Pick any strong MLM encoder in your target language. Common picks: `microsoft/MiniLM-L12-H384-uncased`, `answerdotai/ModernBERT-base` / `-large`, `jhu-clsp/ettin-encoder-*` (17m / 32m / 68m / 150m / 400m / 1b — paired encoder family), `FacebookAI/xlm-roberta-base` (multilingual), `microsoft/mdeberta-v3-base` (multilingual), `jhu-clsp/mmBERT-base` / `-small` (multilingual).
- **Classification cross-encoder**: same as fresh-start; pass `num_labels >= 2`.

Encoder-only bases are still the latency-efficient default (bidirectional attention is well-suited to the reranking use case at small parameter counts), but decoder LLM rerankers (`Qwen3-Reranker-*`, `Qwen3-VL-Reranker-*`) are now competitive at the top of MTEB Reranking when latency / memory budget allows.

**Minimum dataset:** 500k+ labeled `(query, passage, label)` tuples for a production reranker. For continue-training on domain data, 10k–100k labeled pairs is usually enough.

## Dataset Preparation and Validation

**Column-matching rules** (the #1 silent training failure):

1. The label column must be named exactly `label`, `labels`, `score`, or `scores`. Any column with one of those names IS treated as the label, even unintentionally (e.g. a stray retrieval-score column from a previous step).
2. All non-label columns are inputs; column **order** matters, names don't. The first N columns map to the loss's N expected inputs.

Reshape with `dataset.rename_column(old, new)` / `select_columns([...])` / `remove_columns([...])` to fit. To inspect a dataset's columns / a few rows quickly without `load_dataset(...)`: `hf datasets sql "SELECT * FROM 'hf://datasets/<id>/<split>' LIMIT 5"` (DuckDB-backed; see `references/dataset_formats.md`).

Most rerankers need **labeled-pair data** produced by hard-negative mining:

```bash
python scripts/mine_hard_negatives.py \
    --dataset sentence-transformers/gooaq \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --num-negatives 5 \
    --range-min 10 --range-max 100 \
    --sampling-strategy top \
    --output-path data/gooaq-labeled-pairs
```

The `--range-min 10` skips the top-10 retrieved results (often actual positives) as negative candidates. Full details in `references/dataset_formats.md`.

**Cache mined hard negatives** so reruns skip the ~1hr mining pass (per 500k pairs even with FAISS). The CLI script writes to `--output-path` for free; in-script Python `mine_hard_negatives()` calls (like the Quick Start above) re-mine on every run unless wrapped:

```python
import os
from datasets import load_from_disk

CACHE = f"data/{RUN_NAME}-hard-negatives"
if os.path.isdir(CACHE):
    labeled = load_from_disk(CACHE)
else:
    retriever = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1")
    labeled = mine_hard_negatives(...)         # the call from the Quick Start
    labeled.save_to_disk(CACHE)
    del retriever                              # free retriever VRAM before training
    torch.cuda.empty_cache()
```

Delete the cache directory to remine (e.g. after changing `range_min` or the cross-encoder oracle).

## Loss Selection (top picks)

| Data shape | Recommended loss | Family |
|---|---|---|
| `(query, passage, label)` with `label ∈ {0, 1}` or `[0, 1]` | `BinaryCrossEntropyLoss` | Pointwise |
| `(query, passage, class_id)` multi-class | `CrossEntropyLoss` | Pointwise |
| `(query, positive)` pairs, contrastive | `CachedMultipleNegativesRankingLoss` | Contrastive |
| `(query, passages[], scores[])` — listwise, graded | `LambdaLoss` | Listwise |
| Same shape, simpler listwise | `ListMLELoss` or `ListNetLoss` | Listwise |
| `(query, positive, negative)` pairwise | `RankNetLoss` | Pairwise |
| `(query, positive, negative, score_diff)` from a stronger reranker | `MarginMSELoss` | Distillation |
| `(query, passage, teacher_score)` from a stronger reranker | `MSELoss` | Distillation |

Full catalog of 12 losses in `references/losses.md`.

## Evaluator Selection

| Task | Evaluator |
|---|---|
| Rerank retrieval results (nDCG, MRR) | `CrossEncoderNanoBEIREvaluator` |
| Rerank with your own candidates | `CrossEncoderRerankingEvaluator` |
| Binary / multi-class pair classification | `CrossEncoderClassificationEvaluator` |
| Continuous pair scoring (STS) | `CrossEncoderCorrelationEvaluator` |

**If you have held-out data from the training corpus**, build an in-domain `CrossEncoderRerankingEvaluator` from it and wrap with `CrossEncoderNanoBEIREvaluator` in `SequentialEvaluator`. Use the in-domain evaluator as `metric_for_best_model`. **Mandatory for any non-English / domain-specific task**, since NanoBEIR is English-only and a distribution-shift penalty there will hide real in-domain signal (e.g. training on medical 10-K filings + only evaluating on NanoBEIR's NFCorpus / SciFact).

Recipe — turn `(query, positive)` held-out pairs into a reranking eval set via `mine_hard_negatives`. The `n-tuple` columns are 1-indexed (`negative_1, ..., negative_N`) and the **positive column keeps its original input name** (e.g. if your dataset is `(query, answer)` it stays `answer`, not `positive`). The `documents=[positive] + [negs]` form below sidesteps that by naming the positive column once and bundling everything into a single candidate list:

```python
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator

POS_COL = "answer"   # whatever your dataset's positive column is called
NUM_NEGS = 10

valid_ntuple = mine_hard_negatives(
    dataset=valid_pairs, model=retriever,
    num_negatives=NUM_NEGS, range_min=10, range_max=100,
    output_format="n-tuple", use_faiss=True,
    # Pass query_prompt= / corpus_prompt= here too if your retriever needs them
    # (E5, Ruri, BGE-M3, Qwen3-Embedding, etc.); see references/dataset_formats.md.
)
samples = [
    {
        "query": r["query"],
        "positive": [r[POS_COL]],
        "documents": [r[POS_COL]] + [r[f"negative_{i}"] for i in range(1, NUM_NEGS + 1)],
    }
    for r in valid_ntuple
]
evaluator = CrossEncoderRerankingEvaluator(samples=samples, name="in-domain")
# metric_for_best_model = f"eval_{evaluator.primary_metric}"  # e.g. "eval_in-domain_ndcg@10"
```

`documents` + default `always_rerank_positives=True` grades pure reranker quality (the positive is in the candidate list, so the reranker is judged on whether it ranks it on top). Pass `always_rerank_positives=False` and OMIT the positive from `documents` to grade end-to-end retriever+reranker quality instead (a positive the retriever missed counts as rank N+1). See `references/evaluators.md`.

## Training Arguments

The quick-start snippet above is pared down; `scripts/train_example.py` adds tracker, early stopping, Hub push, and model-card metadata. Key knobs:

- **Duration**: 1 epoch for large (>500k) datasets; 3–10 for small. CE peaks mid-training — `EarlyStoppingCallback(patience=3)` is strongly recommended.
- **Batch size**: 32–128 typical (matters less for quality than for bi-encoder MNRL).
- **Learning rate**: `2e-5` (full fine-tune), `1e-4+` (LoRA).
- **LR scaling on batch change**: when effective batch size shifts (more GPUs, larger `per_device_batch`, more `gradient_accumulation_steps`, or moving from local to cluster), center the new LR sweep at `lr_new = lr_old * sqrt(batch_ratio)`. Linear scaling is too aggressive for fine-tuning; sqrt is the safer default.
- **Best checkpoint**: `eval_strategy="steps"`, align `save_steps` with `eval_steps`, `load_best_model_at_end=True`, `metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10"`.
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
- **HF Jobs**: `hf_jobs("uv", {"script": <contents>, "flavor": "a10g-large", "timeout": "3h", "secrets": {"HF_TOKEN": "$HF_TOKEN"}})` + `push_to_hub=True` + `hub_strategy="every_save"`. Full playbook: `references/hf_jobs_execution.md`.

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
| Quick | Training-time evaluator on a held-out test split | seconds to minutes | Every run |
| Benchmark | MTEB / BEIR / domain-specific IR | 30 min to several hours | Before shipping |
| A/B | Production traffic replay, CTR / conversion | days to weeks | Before ramping traffic |

For MTEB: `mteb.evaluate(model, mteb.get_tasks(task_types=["Retrieval"], languages=["eng"]))` (mteb v2.x API is functional, not class-based). Always evaluate **both** the trained model and the base model on the same task list — a positive delta is the only honest signal.

**Sanity-check `predict()` before pushing**, especially under BCE with high `pos_weight` or any non-BCE loss — catches logit saturation that the eval evaluator might also be papering over:

```python
sample_scores = model.predict([("query about X", "highly relevant passage"), ("query about X", "totally unrelated passage")])
assert sample_scores.min() < 0.99 and sample_scores.max() > 0.01, f"Logit saturation suspected: {sample_scores}"
```

## Common Failure Modes

| Symptom | First thing to try |
|---|---|
| OOM | Lower `per_device_train_batch_size`; enable `gradient_checkpointing`; switch to `CachedMultipleNegativesRankingLoss` for contrastive training. |
| Loss → NaN | Drop LR; enable `warmup_steps=0.1`; fp16 → bf16. |
| Metrics stuck at baseline | Check column/loss match (label column must be `label`/`labels`/`score`/`scores`; column order matches loss inputs); mine hard negatives (rerankers **cannot** learn from random negatives); verify `metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10"`. |
| Eval nDCG crashes after distillation/listwise/pairwise | Default `Sigmoid` activation saturated logits. Set `activation_fn=nn.Identity()` (Directive 6). |
| BCE underweights positives | Set `pos_weight=torch.tensor(num_hard_negatives)`. |
| Training hangs at first eval | Missing `eval_dataset` with `eval_strategy="steps"`. |
| `num_labels` mismatch error | `num_labels=1` → `BinaryCrossEntropyLoss`; `num_labels>=2` → `CrossEntropyLoss`. |
| Reranked nDCG below first-stage retriever on 1+ eval dataset | (a) Confirm hard negatives aren't false positives by re-mining with a cross-encoder oracle filter (`mine_hard_negatives(cross_encoder=..., max_score=0.9)`); (b) re-evaluate the best checkpoint with `activation_fn=nn.Identity()` even under BCE — high `pos_weight` can push positive logits past Sigmoid saturation; (c) consider continuing from a pretrained reranker instead of a base encoder (Base Model Selection). |
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
  Save the partial checkpoint anyway: `trainer.save_model(...)`.
- **Stronger comparator (optional)**: override `baseline_eval` with a published baseline's score (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` = 0.6420 NanoBEIR R100 nDCG@10) so the verdict grades against a known model instead of the pre-training base. The +0.005 WIN threshold stays the same.
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
| **Hard-negative mining with cross-encoder as oracle** | `scripts/mine_hard_negatives.py --cross-encoder ...` + `references/dataset_formats.md` |
| **Distillation** (from stronger reranker to smaller, MarginMSE) | `scripts/train_distillation_example.py` |
| **Listwise training** (LambdaLoss with graded relevance) | `scripts/train_listwise_example.py` |
| **Prompts / instructions** | `references/prompts_and_instructions.md` |
| **PEFT / LoRA adapters** | `../train-sentence-transformer/scripts/train_with_lora_example.py` (docstring covers cross-encoder setup variant: `TaskType.SEQ_CLS`. Community examples are sparse for cross-encoders; validate with a smoke test.) |
| **Hyperparameter search** | `references/training_args.md#hyperparameter-search` |
| **Resume from checkpoint** | `references/training_args.md#resuming-training` |

## Resources

Reference docs in `references/` cover the CrossEncoder loss / evaluator catalogs plus the cross-cutting training args, dataset formats, hardware, HF Jobs, prompts, and troubleshooting material. Example scripts in `scripts/` (each with an editorial-recipe docstring), plus the hard-negative mining CLI at `scripts/mine_hard_negatives.py`. PEFT/LoRA and multi-dataset recipes live in the bi-encoder skill's example scripts (their docstrings cover cross-encoder variants where they differ); install `train-sentence-transformer` alongside this skill to read them locally.

Related skills: `train-sentence-transformer` (first-stage retrieval), `train-sparse-encoder` (SPLADE).
