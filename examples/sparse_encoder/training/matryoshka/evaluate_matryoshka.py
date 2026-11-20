# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]",
# ]
# ///
"""Evaluate a sparse model across active dimension budgets on NanoBEIR."""

import argparse
import logging
from pathlib import Path

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_name", help="A Hugging Face model ID or local model directory.")
    cli = parser.parse_args()

    model = SparseEncoder(cli.model_name, model_kwargs={"torch_dtype": "float32"})
    model.max_active_dims = None
    model_path = Path(cli.model_name)
    model_name = model_path.parent.name if model_path.name == "final" else model_path.name
    output_dir = Path("models") / model_name / "eval-matryoshka"
    budgets = {"unpruned": None, "256": 256, "128": 128, "64": 64}

    for name, max_active_dims in budgets.items():
        setting_dir = output_dir / name
        setting_dir.mkdir(parents=True, exist_ok=True)
        evaluator = SparseNanoBEIREvaluator(
            max_active_dims=max_active_dims,
            batch_size=32,
            show_progress_bar=True,
        )
        evaluator(model, output_path=str(setting_dir))


if __name__ == "__main__":
    main()
