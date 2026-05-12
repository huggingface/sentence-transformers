import logging

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")


evaluator = SparseNanoBEIREvaluator(
    dataset_names=None,  # None means evaluate on all datasets
    show_progress_bar=True,
    batch_size=16,
)

# Run evaluation
results = evaluator(model)
"""
Average Queries: 49.92307692307692
Average Corpus: 4334.7692307692305

Aggregated for Score Function: dot
Accuracy@1: 59.18%
Accuracy@3: 75.37%
Accuracy@5: 80.76%
Accuracy@10: 86.92%
Precision@1: 59.18%
Recall@1: 35.62%
Precision@3: 36.26%
Recall@3: 50.85%
Precision@5: 27.75%
Recall@5: 56.57%
Precision@10: 19.24%
Recall@10: 64.31%
MRR@10: 0.6847
NDCG@10: 0.6217
MAP@100: 0.5440
Model Query Sparsity: Active Dimensions: 72.7, Sparsity Ratio: 0.9976
Model Corpus Sparsity: Active Dimensions: 165.9, Sparsity Ratio: 0.9946
Average FLOPS: 3.79
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: NanoBEIR_mean_dot_ndcg@10
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6217
