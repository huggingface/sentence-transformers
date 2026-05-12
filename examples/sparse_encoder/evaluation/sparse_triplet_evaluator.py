import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load triplets from the AllNLI dataset
# The dataset contains triplets of (anchor, positive, negative) sentences
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# Initialize the SparseTripletEvaluator
evaluator = SparseTripletEvaluator(
    anchors=dataset[:1000]["anchor"],
    positives=dataset[:1000]["positive"],
    negatives=dataset[:1000]["negative"],
    name="all_nli_dev",
    batch_size=32,
    show_progress_bar=True,
)

# Run the evaluation
results = evaluator(model)
"""
TripletEvaluator: Evaluating the model on the all_nli_dev dataset:
Accuracy Dot Similarity:        85.40%
Model Anchor Sparsity: Active Dimensions: 103.0, Sparsity Ratio: 0.9966
Model Positive Sparsity: Active Dimensions: 67.4, Sparsity Ratio: 0.9978
Model Negative Sparsity: Active Dimensions: 65.9, Sparsity Ratio: 0.9978
"""
# Print the results
print(f"Primary metric: {evaluator.primary_metric}")
# => Primary metric: all_nli_dev_dot_accuracy
print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8540
