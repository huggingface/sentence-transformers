import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseBinaryClassificationEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize the SPLADE model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

# Initialize the evaluator
binary_acc_evaluator = SparseBinaryClassificationEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    labels=eval_dataset["label"],
    name="quora_duplicates_dev",
    show_progress_bar=True,
    similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
)
results = binary_acc_evaluator(model)
"""
Accuracy with Cosine-Similarity:             75.00      (Threshold: 0.8668)
F1 with Cosine-Similarity:                   67.22      (Threshold: 0.5974)
Precision with Cosine-Similarity:            54.18
Recall with Cosine-Similarity:               88.51
Average Precision with Cosine-Similarity:    67.81
Matthews Correlation with Cosine-Similarity: 49.56

Accuracy with Dot-Product:             76.50    (Threshold: 23.4236)
F1 with Dot-Product:                   67.00    (Threshold: 19.0095)
Precision with Dot-Product:            55.93
Recall with Dot-Product:               83.54
Average Precision with Dot-Product:    65.89
Matthews Correlation with Dot-Product: 48.88

Accuracy with Euclidean-Distance:             67.70     (Threshold: -10.0041)
F1 with Euclidean-Distance:                   48.60     (Threshold: -0.1876)
Precision with Euclidean-Distance:            32.13
Recall with Euclidean-Distance:               99.69
Average Precision with Euclidean-Distance:    20.52
Matthews Correlation with Euclidean-Distance: -4.59

Accuracy with Manhattan-Distance:             67.70     (Threshold: -103.0263)
F1 with Manhattan-Distance:                   48.60     (Threshold: -0.8532)
Precision with Manhattan-Distance:            32.13
Recall with Manhattan-Distance:               99.69
Average Precision with Manhattan-Distance:    21.05
Matthews Correlation with Manhattan-Distance: -4.59

Model Sparsity: Active Dimensions: 61.2, Sparsity Ratio: 0.9980
"""
# Print the results
print(f"Primary metric: {binary_acc_evaluator.primary_metric}")
# => Primary metric: quora_duplicates_dev_max_ap
print(f"Primary metric value: {results[binary_acc_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6781
