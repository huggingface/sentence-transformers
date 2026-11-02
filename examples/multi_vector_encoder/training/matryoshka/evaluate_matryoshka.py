"""Evaluate a multi-vector model across token embedding dimensions on NanoBEIR."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", help="A Hugging Face model ID or local model directory.")
    cli = parser.parse_args()

    model = MultiVectorEncoder(cli.model_name, model_kwargs={"torch_dtype": "float32"})
    model.truncate_dim = None
    model_path = Path(cli.model_name)
    model_name = model_path.parent.name if model_path.name == "final" else model_path.name
    output_dir = Path("models") / model_name / "eval-matryoshka"
    dimensions = {"full": None, "dim_96": 96, "dim_64": 64, "dim_32": 32, "dim_16": 16}

    scores = {}
    for name, truncate_dim in dimensions.items():
        setting_dir = output_dir / name
        setting_dir.mkdir(parents=True, exist_ok=True)
        evaluator = MultiVectorNanoBEIREvaluator(
            batch_size=8,
            corpus_chunk_size=512,
            chunk_elements=20_000_000,
            truncate_dim=truncate_dim,
        )
        results = evaluator(model, output_path=str(setting_dir))
        assert evaluator.primary_metric is not None
        scores[name] = results[evaluator.primary_metric]

    print("\nMean NanoBEIR nDCG@10 (%)")
    for name, score in scores.items():
        print(f"{name:>6}: {100 * score:.2f}")
    print(f"\nEvaluation CSV files: {output_dir}")


if __name__ == "__main__":
    main()
