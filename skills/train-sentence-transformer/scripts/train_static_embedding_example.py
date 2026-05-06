#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=5.0",
#     "datasets>=2.19.0",
#     "accelerate>=0.26.0",
#     "trackio",
#     "tokenizers>=0.20",
# ]
# ///
"""Train a StaticEmbedding model from scratch on a large contrastive dataset.

StaticEmbedding is a token-bag model: a per-token embedding table averaged over
the tokens of an input. No transformer, no attention. Inference is ~20x faster
on GPU and ~80x faster on CPU than a small encoder, with surprisingly competitive
quality on retrieval benchmarks when trained on >=1M contrastive pairs.

Demonstrates:
- Random-init StaticEmbedding (with >=1M training samples this beats from_model2vec
  and from_distillation warm-starts; for smaller datasets, see model_architectures.md)
- MultipleNegativesRankingLoss wrapped in MatryoshkaLoss for nested embedding dims
- Large batch size (1024+) with a high LR (~2e-1, multiple orders of magnitude
  higher than encoder fine-tuning) since the loss surface for a freshly
  initialized embedding table is much flatter than for a pretrained encoder
- BatchSamplers.NO_DUPLICATES (load-bearing for in-batch negatives with duplicated
  anchors)
- NanoBEIREvaluator at full embedding dim
- Auto model card + optional Hub push

Run locally (CPU works for inference, but training needs a GPU for batch=1024+):
    pip install "sentence-transformers[train]>=5.0"
    python train_static_embedding_example.py

Multi-GPU:
    accelerate launch train_static_embedding_example.py

Hugging Face Jobs: paste this file's contents as the `script` in hf_jobs(...).

References:
- HF blog post: https://huggingface.co/blog/static-embeddings
- Module docs: sentence_transformers.sentence_transformer.modules.StaticEmbedding
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import nullcontext

import datasets
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import (
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import StaticEmbedding


def autocast_ctx():
    """bf16/fp16 autocast for evaluator calls outside the trainer."""
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def log_trackio_dashboard():
    """Surface the Trackio dashboard URL so the user can watch training live."""
    try:
        from huggingface_hub import whoami

        hf_user = whoami().get("name")
        if hf_user:
            logging.info(
                f"Trackio dashboard (live training progress): https://huggingface.co/spaces/{hf_user}/trackio"
            )
    except Exception:
        pass


TOKENIZER_NAME = "google-bert/bert-base-uncased"
EMBEDDING_DIM = 1024
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64, 32]  # ordered largest-first per MatryoshkaLoss

OUTPUT_DIR = "models/static-embedding-bert-uncased"
RUN_NAME = "static-embedding-bert-uncased"


def setup_logging():
    """Configure logging + TF32. Tees to logs/{RUN_NAME}.log and silences HTTP spam."""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(f"logs/{RUN_NAME}.log")],
        force=True,
    )
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def load_pair_dataset() -> datasets.Dataset:
    """Load a contrastive-pair dataset for training.

    StaticEmbedding starts from random initialization, so it needs *a lot* of
    contrastive signal to converge. GooAQ alone provides ~3M (question, answer)
    pairs, comfortably over the >=1M threshold below which a warm-start would
    beat random init. For stronger production models, concatenate more sources
    (NaturalQuestions, MSMARCO, MIRACL, etc.) and shuffle, in the same family of
    sources used in `sentence-transformers/static-retrieval-mrl-en-v1`.
    """
    return (
        load_dataset("sentence-transformers/gooaq", split="train")
        .rename_columns({"question": "anchor", "answer": "positive"})
        .select_columns(["anchor", "positive"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", type=str, default=None, help="Skip training; load this saved model and run only the evaluator."
    )
    cli, _ = parser.parse_known_args()

    setup_logging()

    if cli.eval_only:
        logging.info(f"Eval-only mode: loading model from {cli.eval_only}")
        model = SentenceTransformer(cli.eval_only)
        evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
        with autocast_ctx():
            evaluator(model)
        return

    logging.info(f"Building StaticEmbedding from {TOKENIZER_NAME} tokenizer (dim={EMBEDDING_DIM}, random init)")
    # Random init beats from_model2vec / from_distillation when the training set
    # has >=1M pairs. For smaller datasets (<100k), warm-start instead via
    # StaticEmbedding.from_model2vec("minishlab/potion-base-8M") or
    # StaticEmbedding.from_distillation("sentence-transformers/all-MiniLM-L6-v2", ...)
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    static_embedding = StaticEmbedding(tokenizer, embedding_dim=EMBEDDING_DIM)
    model = SentenceTransformer(
        modules=[static_embedding],
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"Static embedding ({EMBEDDING_DIM}d) trained on contrastive pairs",
        ),
    )

    logging.info("Loading + concatenating training datasets")
    full = load_pair_dataset()
    split = full.train_test_split(test_size=10_000, seed=12)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logging.info(f"  train: {len(train_dataset):,} rows | eval: {len(eval_dataset):,} rows")
    logging.info(f"  columns: {train_dataset.column_names}")

    inner = MultipleNegativesRankingLoss(model)
    loss = MatryoshkaLoss(model, inner, matryoshka_dims=MATRYOSHKA_DIMS)

    evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"])
    logging.info("Baseline evaluation (random init scores near zero, confirming the pipeline runs):")
    with autocast_ctx():
        baseline_eval = evaluator(model)[evaluator.primary_metric]

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2048,
        per_device_eval_batch_size=2048,
        learning_rate=2e-1,  # ~10000x higher than encoder fine-tuning, by design
        weight_decay=0.0,  # weight decay on a token-bag is usually harmful
        warmup_steps=0.1,
        lr_scheduler_type="linear",
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_cosine_ndcg@10",
        greater_is_better=True,
        report_to="trackio",
        run_name=RUN_NAME,
        seed=12,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    log_trackio_dashboard()
    trainer.train()

    logging.info("Post-training evaluation:")
    with autocast_ctx():
        score = evaluator(model)[evaluator.primary_metric]
    delta = score - baseline_eval
    verdict = "WIN" if delta >= 0.005 else "MARGINAL" if delta >= 0 else "REGRESSION"
    logging.info(f"VERDICT: {verdict} | score={score:.4f} | baseline={baseline_eval:.4f} | delta={delta:+.4f}")

    final_dir = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_dir)
    logging.info(f"Saved final model to {final_dir}")

    try:
        model.push_to_hub(RUN_NAME)
        logging.info(f"Pushed model to https://huggingface.co/{RUN_NAME}")
    except Exception:
        import traceback

        logging.error(f"Hub push failed:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
