"""Train text/image pairs with lazy preprocessing inside GradCache mini-batches.

Preprocessing must be deterministic. Run with --help for the JSONL format.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="A local/Hub model supporting text and image inputs.")
    parser.add_argument(
        "--data", type=Path, required=True, help='JSONL rows: {"query": "text", "image": "local/path.jpg"}.'
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--eager", action="store_true", help="Reference run: preprocess the full batch before GradCache."
    )
    args = parser.parse_args()

    with args.data.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    dataset = Dataset.from_dict(
        {
            "query": [row["query"] for row in rows],
            "image": [{"image": str((args.data.parent / row["image"]).resolve())} for row in rows],
        }
    )
    model = SentenceTransformer(args.model, device=args.device)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        use_cpu=args.device == "cpu",
        lazy_preprocessing=not args.eager,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        dataloader_num_workers=0,
        save_strategy="no",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size),
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    trainer.train()
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        print(f"peak_allocated_mib={torch.cuda.max_memory_allocated(model.device) / 1024**2:.2f}")
    print(f"train_seconds={time.perf_counter() - started:.3f}")
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
