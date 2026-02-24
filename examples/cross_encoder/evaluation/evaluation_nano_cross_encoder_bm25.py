"""Simple CrossEncoder NanoBEIR reranking example.

Run:
  uv run --with datasets python examples/cross_encoder/evaluation/evaluation_nano_cross_encoder_bm25.py
"""

import logging

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.evaluation.nano_beir import DATASET_NAME_TO_HUMAN_READABLE

logging.basicConfig(format="%(message)s", level=logging.INFO)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DATASET_ID = "sentence-transformers/NanoBEIR-en"
DATASET_SPLITS = ["msmarco", "nq"]
RERANK_K = 100

model = CrossEncoder(MODEL_NAME)
evaluator = CrossEncoderNanoBEIREvaluator(
    dataset_id=DATASET_ID,
    dataset_names=DATASET_SPLITS,
    candidate_subset_name="bm25",
    rerank_k=RERANK_K,
    at_k=10,
    batch_size=32,
    show_progress_bar=False,
)

results = evaluator(model)
if evaluator.primary_metric is None:
    raise ValueError("Expected evaluator.primary_metric to be set after evaluation.")

primary_metric = evaluator.primary_metric
if primary_metric not in results:
    primary_metric = f"{evaluator.name}_{primary_metric}"
if primary_metric not in results:
    raise ValueError(f"Primary metric key not found: {primary_metric}")

"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: cross-encoder/ms-marco-MiniLM-L6-v2
  Dataset: sentence-transformers/NanoBEIR-en
  Splits: ['msmarco', 'nq']
  Split scores:
  - NanoMSMARCO_R100_ndcg@10 = 0.6686
  - NanoNQ_R100_ndcg@10 = 0.7599
  Primary metric key: NanoBEIR_R100_mean_ndcg@10
  Primary metric value: 0.7142
"""

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_ID}")
print(f"Splits: {DATASET_SPLITS}")
metric_suffix = primary_metric.split("_mean_", maxsplit=1)[1]
print("Split scores:")
for split_name in DATASET_SPLITS:
    human_readable = DATASET_NAME_TO_HUMAN_READABLE[split_name.lower()]
    split_key = f"Nano{human_readable}_R{RERANK_K}_{metric_suffix}"
    if split_key in results:
        print(f"- {split_key} = {float(results[split_key]):.4f}")
print(f"Primary metric key: {primary_metric}")
print(f"Primary metric value: {float(results[primary_metric]):.4f}")
