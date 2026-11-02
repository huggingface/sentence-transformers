"""Evaluate a multi-vector model across document token budgets on NanoBEIR."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", help="A Hugging Face model ID or local model directory.")
    cli = parser.parse_args()

    model = MultiVectorEncoder(cli.model_name, model_kwargs={"torch_dtype": "float32"})
    model_path = Path(cli.model_name)
    model_name = model_path.parent.name if model_path.name == "final" else model_path.name
    output_dir = Path("models") / model_name / "eval-token-pooling"
    token_poolings = {
        "unpooled": None,
        "max_tokens_64": HierarchicalTokenPooling(max_tokens=64),
        "max_tokens_32": HierarchicalTokenPooling(max_tokens=32),
        "max_tokens_16": HierarchicalTokenPooling(max_tokens=16),
        "max_tokens_8": HierarchicalTokenPooling(max_tokens=8),
        "max_tokens_4": HierarchicalTokenPooling(max_tokens=4),
        "pool_factor_2": HierarchicalTokenPooling(pool_factor=2),
        "pool_factor_3": HierarchicalTokenPooling(pool_factor=3),
        "pool_factor_4": HierarchicalTokenPooling(pool_factor=4),
        "pool_factor_5": HierarchicalTokenPooling(pool_factor=5),
        "pool_factor_6": HierarchicalTokenPooling(pool_factor=6),
    }

    scores = {}
    for name, token_pooling in token_poolings.items():
        setting_dir = output_dir / name
        setting_dir.mkdir(parents=True, exist_ok=True)
        evaluator = MultiVectorNanoBEIREvaluator(
            batch_size=16,
            corpus_chunk_size=1024,
            chunk_elements=100_000_000,
            token_pooling=token_pooling,
        )
        results = evaluator(model, output_path=str(setting_dir))
        assert evaluator.primary_metric is not None
        scores[name] = results[evaluator.primary_metric]

    print("\nMean NanoBEIR nDCG@10 (%)")
    for name, score in scores.items():
        print(f"{name:>13}: {100 * score:.2f}")
    print(f"\nEvaluation CSV files: {output_dir}")


if __name__ == "__main__":
    main()
