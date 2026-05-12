import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load any dataset with some texts
dataset = load_dataset("sentence-transformers/stsb", split="validation")
sentences = dataset["sentence1"] + dataset["sentence2"]

# Given queries, a corpus and a mapping with relevant documents, the SparseMSEEvaluator computes different MSE metrics.
mse_evaluator = SparseMSEEvaluator(
    source_sentences=sentences,
    target_sentences=sentences,
    teacher_model=teacher_model,
    name="stsb-dev",
)
results = mse_evaluator(student_model)
"""
MSE evaluation (lower = better) on the stsb-dev dataset:
MSE (*100):     0.034905
Model Sparsity: Active Dimensions: 54.6, Sparsity Ratio: 0.9982
"""
# Print the results
print(f"Primary metric: {mse_evaluator.primary_metric}")
# => Primary metric: stsb-dev_negative_mse
print(f"Primary metric value: {results[mse_evaluator.primary_metric]:.4f}")
# => Primary metric value: -0.0349
