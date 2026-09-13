"""
This example evaluates two small models on two NanoBEIR datasets with bootstrap confidence intervals,
then tests whether the difference between the two models is significant on the same queries.

NanoBEIR datasets have roughly 50 queries each, so nDCG@10 differences of 0.5-1 point are often within
the noise. Passing `bootstrap_resamples` to the evaluator adds `_ci_low` / `_ci_high` keys next to every
metric, and each sub-evaluator exposes its per-query values in `per_query_metrics`, which can be fed to
`paired_bootstrap_test` to compare two models query by query.

Usage:
python evaluation_nanobeir_confidence_intervals.py
OR
python evaluation_nanobeir_confidence_intervals.py model_name_a model_name_b
"""

import logging
import sys

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.util import paired_bootstrap_test

# Limit torch to 4 threads
torch.set_num_threads(4)

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

model_name_a = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
model_name_b = sys.argv[2] if len(sys.argv) > 2 else "sentence-transformers/static-retrieval-mrl-en-v1"

# Short dataset names; the results are reported as NanoMSMARCO and NanoNQ
dataset_names = ["msmarco", "nq"]
n_resamples = 1000

# Per-query nDCG@10 per model and dataset, used for the paired test below
per_query_ndcg = {}
for model_name in [model_name_a, model_name_b]:
    model = SentenceTransformer(model_name)
    # Use a fresh evaluator per model, so that each one keeps its own per-query metrics
    evaluator = NanoBEIREvaluator(dataset_names=dataset_names, bootstrap_resamples=n_resamples)
    results = evaluator(model)

    # The evaluator scores with the model's similarity function, "cosine" for both default models
    score_fn = model.similarity_fn_name
    print(f"\n{model_name}")
    for prefix in [evaluator.name] + [sub_evaluator.name for sub_evaluator in evaluator.evaluators]:
        value = results[f"{prefix}_{score_fn}_ndcg@10"]
        ci_low = results[f"{prefix}_{score_fn}_ndcg@10_ci_low"]
        ci_high = results[f"{prefix}_{score_fn}_ndcg@10_ci_high"]
        print(f"  {prefix:<14} nDCG@10: {value:.4f} (95% CI: {ci_low:.4f} to {ci_high:.4f})")

    # Per-query values are aligned with each sub-evaluator's `queries_ids`, so two runs on the same
    # dataset can be compared query by query
    per_query_ndcg[model_name] = {
        sub_evaluator.name: sub_evaluator.per_query_metrics[score_fn]["ndcg@10"]
        for sub_evaluator in evaluator.evaluators
    }

print(f"\nPaired bootstrap test, {model_name_b} minus {model_name_a}:")
for dataset_name in per_query_ndcg[model_name_a]:
    result = paired_bootstrap_test(
        per_query_ndcg[model_name_a][dataset_name],
        per_query_ndcg[model_name_b][dataset_name],
        n_resamples=n_resamples,
    )
    # The confidence interval is the primary output: the difference is significant when it excludes zero
    verdict = "significant" if result.ci_low > 0 or result.ci_high < 0 else "not significant"
    print(
        f"  {dataset_name:<14} nDCG@10 difference: {result.difference:+.4f} "
        f"(95% CI: {result.ci_low:+.4f} to {result.ci_high:+.4f}), p = {result.p_value:.3f}, "
        f"{verdict} over {result.n_samples} queries"
    )
