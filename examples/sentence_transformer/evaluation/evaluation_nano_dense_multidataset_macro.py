"""Simple dense multi-dataset Nano macro example.

Run:
  uv run --with datasets python examples/sentence_transformer/evaluation/evaluation_nano_dense_multidataset_macro.py
"""

import logging

import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

MODEL_NAME = "intfloat/multilingual-e5-small"
MULTILINGUAL_NANOBEIR_DATASET_IDS = [
    "sentence-transformers/NanoBEIR-en",
    "LiquidAI/NanoBEIR-ja",
]
CUSTOM_DATASET_IDS = [
    "hotchpotch/NanoCodeSearchNet",
]


def evaluate_dataset(model: SentenceTransformer, dataset_id: str) -> tuple[str, str, float]:
    evaluator = NanoEvaluator(
        dataset_id=dataset_id,
        dataset_names=None,
        batch_size=32,
        show_progress_bar=False,
    )
    results = evaluator(model)
    if evaluator.primary_metric is None:
        raise ValueError(f"Expected evaluator.primary_metric for dataset_id={dataset_id}")
    return dataset_id, evaluator.primary_metric, float(results[evaluator.primary_metric])


model = SentenceTransformer(MODEL_NAME)

multilingual_results = [evaluate_dataset(model, dataset_id) for dataset_id in MULTILINGUAL_NANOBEIR_DATASET_IDS]
custom_results = [evaluate_dataset(model, dataset_id) for dataset_id in CUSTOM_DATASET_IDS]

multilingual_scores = [score for _, _, score in multilingual_results]
custom_scores = [score for _, _, score in custom_results]

multilingual_macro = float(np.mean(multilingual_scores))
custom_macro = float(np.mean(custom_scores))
group_macro = float(np.mean([multilingual_macro, custom_macro]))

"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: intfloat/multilingual-e5-small
  Multilingual dataset scores:
  - sentence-transformers/NanoBEIR-en | NanoBEIR-en_mean_cosine_ndcg@10 = 0.5542
  - LiquidAI/NanoBEIR-ja | NanoBEIR-ja_mean_cosine_ndcg@10 = 0.4985
  Custom dataset scores:
  - hotchpotch/NanoCodeSearchNet | NanoCodeSearchNet_mean_cosine_ndcg@10 = 0.7381
  Multilingual macro mean: 0.5263
  Custom macro mean: 0.7381
  Group macro mean: 0.6322
"""

print(f"Model: {MODEL_NAME}")
print("Multilingual dataset scores:")
for dataset_id, metric_key, score in multilingual_results:
    print(f"- {dataset_id} | {metric_key} = {score:.4f}")
print("Custom dataset scores:")
for dataset_id, metric_key, score in custom_results:
    print(f"- {dataset_id} | {metric_key} = {score:.4f}")
print(f"Multilingual macro mean: {multilingual_macro:.4f}")
print(f"Custom macro mean: {custom_macro:.4f}")
print(f"Group macro mean: {group_macro:.4f}")
