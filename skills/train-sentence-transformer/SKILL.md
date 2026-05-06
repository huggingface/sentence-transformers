---
name: train-sentence-transformer
description: Train or fine-tune SentenceTransformer (bi-encoder) embedding models with the sentence-transformers library — fixed-dimension dense vectors for semantic similarity, retrieval, clustering, classification, paraphrase mining, and deduplication. Covers loss selection, hard-negative mining, NanoBEIR / STS evaluators, multi-task training, distillation, PEFT / LoRA, Matryoshka, prompts, and Hugging Face Hub publishing. Use for any bi-encoder / embedding model training; for rerankers use `train-cross-encoder`, for sparse retrieval use `train-sparse-encoder`.
---

# Train a SentenceTransformer (Bi-Encoder) Embedding Model

Fine-tune or train a `SentenceTransformer` — the bi-encoder architecture that maps each input to a fixed-dimension dense vector for similarity, retrieval, clustering, and classification. For one-shot `(query, passage)` pair scoring use `train-cross-encoder`; for learned sparse retrieval use `train-sparse-encoder`.

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

For experimentation mode (multiple iterations vs a stronger known baseline) and the `logs/experiments.md` append, see `## Experimentation`.

## Key Directives

1. **Load the model in fp32; autocast bf16/fp16 at runtime.** Never pass `torch_dtype=torch.bfloat16` — it puts Adam state in bf16 and silently degrades quality. Use `bf16=True` in TrainingArguments, and wrap out-of-trainer `evaluator(model)` calls in `torch.autocast("cuda", dtype=torch.bfloat16)` (mandatory with FA2; otherwise just a speedup).
2. **Verify VRAM fits; on Windows, overflow is silent.** When VRAM hits ~95% on Windows, the NVIDIA driver silently spills into host RAM via "shared memory" and training throughput drops 5-10x with no error (`nvidia-smi` shows 100% util either way). Run 1-2 steps and watch `tokens/sec`; if dropping, cut `per_device_train_batch_size` or `model.max_seq_length` first. `gradient_checkpointing=True` is a fallback (incompatible with `Cached*` losses).
3. **Pick a loss that matches the data shape** — don't reshape data to fit a preferred loss. See `references/losses.md`.
4. **`BatchSamplers.NO_DUPLICATES`** for MNRL / CachedMNRL / GIST / symmetric variants — duplicate anchors otherwise create false negatives.
5. **Hard negatives are the single highest-leverage lever for retrieval quality.** Mine them (`scripts/mine_hard_negatives.py`) when your training data is `(question, answer)` pairs.
6. **Always include an evaluator and run it once on the base model before training** (under autocast for speed) — that's your baseline. Set `load_best_model_at_end=True` + `metric_for_best_model=...`. `NanoBEIREvaluator` is the strong retrieval default; `EmbeddingSimilarityEvaluator` for STS.
7. **Silence noisy HTTP loggers + write logs to `logs/{run_name}.log`** at the top of every script: `for L in ("httpx", "httpcore", "huggingface_hub", "urllib3"): logging.getLogger(L).setLevel(logging.WARNING)`. Otherwise HF download URLs flood the agent's context. Tee logs to a file so they survive the session.
8. **Default Hub push: end-of-run, public, wrapped in try-except** (matches the repo's example-script idiom; auth or naming failures don't lose the local checkpoint). On HF Jobs (ephemeral env) ALSO enable in-trainer push: `push_to_hub=True`, `hub_model_id=RUN_NAME`, `hub_strategy="every_save"`, plus `secrets={"HF_TOKEN": "$HF_TOKEN"}` in the job spec — otherwise weights are lost on timeout.

**Attention backend**: pass `model_kwargs={"attn_implementation": "flash_attention_2"}` for FA2 (needs `pip install flash-attn`, Ampere+ only) or `"kernels-community/flash-attn"` (precompiled via `pip install kernels`, easier install) when training long sequences or large batches. Default `sdpa` is fine for short sequences. Don't combine with `torch_dtype="bfloat16"` — let `bf16=True` autocast handle dtype (Directive 1).

## Prerequisites

```bash
pip install "sentence-transformers[train]>=5.0"        # add [train,image] / [audio] / [video] for multimodal
pip install trackio                                    # optional tracker; or wandb / tensorboard / mlflow
hf auth login                                          # or set HF_TOKEN with write scope (for Hub push)
```

Dataset must load via `datasets.load_dataset(...)`. Column order must match the loss (names don't matter). GPU strongly recommended (CPU works only for demos and `StaticEmbedding`). See `references/hardware_guide.md` for sizing, multi-GPU, and HF Jobs flavors.

**Execution modes**: local (`python train.py` / `uv run train.py` / `accelerate launch train.py`) is default. HF Jobs (`hf_jobs("uv", {...})`) scales out, requires a Pro/Team/Enterprise plan, and **mandates Hub push** — see `references/hf_jobs_execution.md`.

## Quick Start

Fine-tune a bi-encoder on Natural Language Inference triplets with `MultipleNegativesRankingLoss` and evaluate with NanoBEIR:

```python
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("microsoft/mpnet-base")

train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(50_000))
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1_000))

loss = MultipleNegativesRankingLoss(model)
evaluator = NanoBEIREvaluator()
baseline_eval = evaluator(model)[evaluator.primary_metric]   # baseline: measure before training (used by verdict block)

args = SentenceTransformerTrainingArguments(
    output_dir="models/mpnet-base-all-nli",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
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
    metric_for_best_model=f"eval_{evaluator.primary_metric}",   # robust to evaluator swap
    greater_is_better=True,
    run_name="mpnet-base-all-nli",
)

trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
    loss=loss, evaluator=evaluator,
)
trainer.train()
model.save_pretrained("models/mpnet-base-all-nli/final")

# Default: push to Hub at end (public, under your authenticated user). Wrapped in
# try/except so a Hub auth or naming failure doesn't lose the local checkpoint.
try:
    model.push_to_hub("mpnet-base-all-nli")
except Exception:
    import traceback
    logging.error(f"Hub push failed:\n{traceback.format_exc()}")
```

**Before committing to a long run**, smoke-test by adding `max_steps=1` to the training args and slicing the dataset (`train_dataset = train_dataset.select(range(10))`). Catches column-shape, dtype, and loss-mismatch issues in seconds before you burn the full run.

See `scripts/train_example.py` for the full production template (with trackio, Hub push, best-checkpoint loading).

## Base Model Selection

Embedding leaderboards rotate every few months; don't trust any hardcoded "best" pick. Discover current options live — run **both** sort orders since most-downloaded surfaces proven options and trending surfaces recent SOTA that may not have download volume yet:

```bash
hf models list --filter sentence-transformers --sort downloads --limit 20
hf models list --filter sentence-transformers --sort trending  --limit 20

# Optional language narrowing (not all multilingual models tag each language, so missing matches doesn't mean the model can't handle that language — re-run without the filter to compare):
hf models list --filter sentence-transformers --filter <language-code> --sort trending --limit 20

hf models card <id> --text                        # confirm dimensions, max_seq_length, license, languages
```

Cross-check the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for current SOTA before committing.

**Shape guidance:**

- **Continue from an existing retriever** beats fresh-start + 100k–500k pairs. Common namespaces as of 2026-Q2 (verify against the live discovery commands above — the field rotates):
  - English encoder retrievers: `sentence-transformers/all-*` (MiniLM-L6-v2, mpnet-base-v2 still the most-downloaded models on the Hub), `BAAI/bge-*-en-v1.5` (small / base / large), `nomic-ai/nomic-embed-text-v1.5`, `mixedbread-ai/mxbai-embed-large-v1`, `Alibaba-NLP/gte-*`, `Snowflake/snowflake-arctic-embed-*`, `jinaai/jina-embeddings-v5-text-small` / `-nano`, `microsoft/harrier-oss-v1-270m` / `-0.6b`.
  - Multilingual encoder retrievers: `sentence-transformers/paraphrase-multilingual-*`, `intfloat/multilingual-e5-*` (small / base / large), `ibm-granite/granite-embedding-*-multilingual-r2`, `google/embeddinggemma-300m`, `voyageai/voyage-4-nano`.
  - Long documents (8k+): `nomic-ai/modernbert-embed-*`, `answerdotai/ModernBERT-large`.
  - Decoder LLM retrievers (multilingual; **need last-token pooling**): `Qwen/Qwen3-Embedding-*` (0.6B / 4B / 8B), `Qwen/Qwen3-VL-Embedding-*` (multimodal), `codefuse-ai/F2LLM-v2-*`.
- **Fresh-start English** (≥500k pairs + domain-fit reason): `microsoft/mpnet-base`, `answerdotai/ModernBERT-base`, `google-bert/bert-base-uncased`, `jhu-clsp/ettin-encoder-*` (17m / 32m / 68m / 150m / 400m / 1b — paired encoder family).
- **Fresh-start multilingual**: `FacebookAI/xlm-roberta-base` (MLM-only, needs contrastive training), `microsoft/mdeberta-v3-base`, `jhu-clsp/mmBERT-base` / `-small`.
- **CPU / small footprint (`StaticEmbedding`)**: `StaticEmbedding(tokenizer, embedding_dim=...)`. **Model size = `vocab_size × dim × 4 bytes`** — pick a small-vocab tokenizer or you get a giant model: 30k-vocab `bert-base-uncased` × 128 dim ≈ 15 MB; **250k-vocab `paraphrase-multilingual-MiniLM-L12-v2` × 256 dim ≈ 256 MB**. Random init needs 1M+ pairs; warm-start (`StaticEmbedding.from_distillation(...)`) helps under ~100k pairs.

Architecture variants (encoder / decoder / static / Router), pooling rules, and decoder-vs-encoder setup paths: `references/model_architectures.md`.

## Dataset Preparation and Validation

**Column-matching rules** (the #1 silent training failure):

1. The label column must be named exactly `label`, `labels`, `score`, or `scores`. Any column with one of those names IS treated as the label, even unintentionally (e.g. a stray retrieval-score column from a previous step).
2. All non-label columns are inputs; column **order** matters, names don't. The first N columns map to the loss's N expected inputs.

Reshape with `dataset.rename_column(old, new)` / `select_columns([...])` / `remove_columns([...])` to fit. To inspect a dataset's columns / a few rows quickly without `load_dataset(...)`: `hf datasets sql "SELECT * FROM 'hf://datasets/<id>/<split>' LIMIT 5"` (DuckDB-backed; see `references/dataset_formats.md`).

If you're doing contrastive training, mine hard negatives:

```bash
python scripts/mine_hard_negatives.py --dataset ... --model sentence-transformers/all-MiniLM-L6-v2 --num-negatives 5 --output-path data/...
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

Per-loss column layouts + mining filter options: `references/dataset_formats.md`.

## Loss Selection (top picks)

| Data shape | Recommended loss | Notes |
|---|---|---|
| `(anchor, positive)` | `MultipleNegativesRankingLoss` | Default for retrieval. Use `NO_DUPLICATES` sampler. |
| `(anchor, positive)` + big batches | `CachedMultipleNegativesRankingLoss` | GradCache: effective batch 1024+ on a 24GB GPU. |
| `(anchor, positive, negative)` | `MultipleNegativesRankingLoss` or cached variant | Triplets improve quality when negatives are hard. |
| `(text1, text2, score)` with `score ∈ [-1, 1]` or `[0, 1]` | `CoSENTLoss` | Best continuous regression loss. |
| `(text1, text2, label)` with `label ∈ {0, 1}` | `OnlineContrastiveLoss` | Robust to noisy labels. |
| Teacher-student distillation | `MSELoss` (embedding) or `MarginMSELoss` (score) | See `scripts/train_distillation_example.py`. |
| Multiple output dimensions from one training | Wrap any loss in `MatryoshkaLoss` | Enables `truncate_dim` at inference. |

Full catalog of ~28 losses in `references/losses.md`.

## Evaluator Selection

| Task | Evaluator |
|---|---|
| Retrieval (nDCG, MRR, Recall) | `NanoBEIREvaluator`, `InformationRetrievalEvaluator` |
| STS / continuous similarity | `EmbeddingSimilarityEvaluator` |
| Binary classification | `BinaryClassificationEvaluator` |
| Triplet accuracy | `TripletEvaluator` |
| Reranking (from retrieval candidates) | `RerankingEvaluator` |
| MSE vs. teacher | `MSEEvaluator` |
| Paraphrase mining | `ParaphraseMiningEvaluator` |

**If you have held-out data from the training corpus**, build an in-domain `InformationRetrievalEvaluator` (or `EmbeddingSimilarityEvaluator` for STS-style training data) from it and wrap with `NanoBEIREvaluator` in `SequentialEvaluator`. Use the in-domain evaluator as `metric_for_best_model`; NanoBEIR alone measures generalization to a *different* corpus and a distribution-shift penalty there can hide real in-domain signal.

Wrap multiple in `SequentialEvaluator` to track them together. See `references/evaluators.md`.

## Training Arguments

The quick-start snippet above is pared down; `scripts/train_example.py` adds tracker, early stopping, Hub push, and model-card metadata. Key knobs:

- **Duration**: 1 epoch for large (>500k) datasets; 3–10 for small.
- **Batch size**: push as high as VRAM allows for MNRL-family losses. `CachedMultipleNegativesRankingLoss` decouples effective batch from VRAM.
- **In-batch negatives are per-device by default.** Single-GPU batch 64 = 63 negatives per anchor; 4× DDP at per-device 64 = still 63 unless you pass `gather_across_devices=True` to `MultipleNegativesRankingLoss` / `MultipleNegativesSymmetricRankingLoss` / `GISTEmbedLoss` / their `Cached*` variants. With `gather_across_devices=True` on 4× DDP at per-device 64 you get 255 negatives. Single-GPU runs ignore the flag.
- **Learning rate**: `2e-5` (BERT-family full fine-tune), `1e-4+` (LoRA), `2e-1` (`StaticEmbedding` from scratch).
- **LR scaling on batch change**: when effective batch size shifts (more GPUs, larger `per_device_batch`, more `gradient_accumulation_steps`, or moving from local to cluster), center the new LR sweep at `lr_new = lr_old * sqrt(batch_ratio)`. Linear scaling is too aggressive for fine-tuning; sqrt is the safer default for both contrastive (MNRL family) and listwise losses.
- **Sampler**: `batch_sampler=BatchSamplers.NO_DUPLICATES` for contrastive losses.
- **Best checkpoint**: `eval_strategy="steps"`, align `save_steps` with `eval_steps`, `load_best_model_at_end=True`, `metric_for_best_model="eval_<primary_metric>"`.
- **Live progress dashboard**: with `report_to="trackio"`, log `f"https://huggingface.co/spaces/{whoami().get('name')}/trackio"` right after building the trainer so the user can watch live. The Space auto-creates on the first run with a valid `HF_TOKEN`.

Full arg treatment + HPO + resume + tracker setup: `references/training_args.md`.

## Running the Training

Use a PEP 723 header so the same script runs locally, via `uv run`, or on HF Jobs:

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["sentence-transformers[train]>=5.0", "trackio"]
# ///
```

- **Local**: `python train.py` or `uv run train.py`.
- **Multi-GPU**: `accelerate launch train.py`. No code changes; `per_device_train_batch_size` stays per-GPU.
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

## Common Failure Modes

| Symptom | First thing to try |
|---|---|
| OOM | Lower `per_device_train_batch_size`; enable `gradient_checkpointing`; switch to a `Cached*` loss. |
| Loss → NaN | Drop LR; enable `warmup_steps=0.1`; fp16 → bf16 if GPU supports it. |
| Metrics stuck at baseline | Check column/loss match (label column must be `label`/`labels`/`score`/`scores`; column order matches loss inputs); mine hard negatives; verify `metric_for_best_model` spelling. |
| Training hangs at first eval | Missing/empty `eval_dataset` with `eval_strategy="steps"`. |
| `Cached*` loss crashes | Incompatible with `gradient_checkpointing=True`. Disable one. |
| Multi-GPU hangs at start | Set `batch_sampler=BatchSamplers.NO_DUPLICATES`; ensure all processes see the same dataset. |
| Training crashed / want to resume | `trainer.train(resume_from_checkpoint=True)` auto-detects the latest checkpoint in `output_dir`; pass an explicit path to resume from a specific step. `IterableDataset` iteration order is **not** preserved across resumption — handle streaming-dataset positioning yourself. |
| Hub push 401/403 | `hf auth whoami`; token needs **write** scope; on HF Jobs pass `secrets={"HF_TOKEN": "$HF_TOKEN"}`. |

Full catalog in `references/troubleshooting.md`.

## Experimentation

After a single run completes, propose iteration if it would benefit the user (phrasing like "beat baseline X", "see how high you can push it", weak/marginal verdict). When iterating:

- **Change one variable per run.** Multi-variable deltas are unattributable.
- **Kill underperforming runs quickly** (don't wait for `EarlyStoppingCallback` after most of the budget is gone):
  - Baseline regression: first eval below the pre-training baseline → kill, setup is broken.
  - Plateau: 3 consecutive evals within ±0.5% and below target → kill.
  - Loss explosion: `nan` or >2× the baseline minimum → kill.
  Save the partial checkpoint anyway: `trainer.save_model(...)`.
- **Stronger comparator (optional)**: override `baseline_eval` with a published baseline's score (e.g. `intfloat/multilingual-e5-small` on NanoBEIR) so the verdict grades against a known model instead of the pre-training base. The +0.005 WIN threshold stays the same.
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

| Pattern | Script + Reference |
|---|---|
| **Matryoshka** (train once, deploy at multiple dims) | `scripts/train_matryoshka_example.py` + `references/losses.md#matryoshkaloss` |
| **Multi-task / multi-dataset** | `scripts/train_multi_dataset_example.py` (docstring covers per-dataset losses, single-loss + DatasetDict variant, samplers, gotchas) |
| **PEFT / LoRA adapters** | `scripts/train_with_lora_example.py` (docstring covers when to use, hyperparams, QLoRA, adapter sharing, gotchas) |
| **Distillation** (teacher -> student) | `scripts/train_distillation_example.py` (docstring covers Embedding MSE / Margin MSE / Listwise KL patterns, dim-mismatch projection, layer pruning, CrossEncoder-student Identity activation) |
| **Multilingual** (extend English teacher to other languages) | `scripts/train_make_multilingual_example.py` (docstring covers parallel-data sources, when to use, teacher / student picks) |
| **Static embedding** (~80x faster, train from scratch) | `scripts/train_static_embedding_example.py` + `references/model_architectures.md#static-embeddings` |
| **Prompts / instructions** | `references/prompts_and_instructions.md` |
| **Hard negative mining** | `scripts/mine_hard_negatives.py` + `references/dataset_formats.md` |
| **Hyperparameter search** | `references/training_args.md#hyperparameter-search` |
| **Resume from checkpoint** | `references/training_args.md#resuming-training` |

## Resources

Reference docs in `references/` cover bi-encoder loss / evaluator / model-architecture catalogs plus the cross-cutting training args, dataset formats, hardware, HF Jobs, prompts, and troubleshooting material. Example scripts in `scripts/` (each with an editorial-recipe docstring), plus the hard-negative mining CLI at `scripts/mine_hard_negatives.py`.

Related skills: `train-cross-encoder` (rerankers), `train-sparse-encoder` (SPLADE). External: [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for benchmarking the trained model.
