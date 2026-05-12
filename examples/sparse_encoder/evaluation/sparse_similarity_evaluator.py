import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
model.similarity_fn_name = "cosine"  # even though the model is trained with dot, we need to set it to cosine for evaluation as the score in the dataset is cosine similarity

# Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# Initialize the evaluator
dev_evaluator = SparseEmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts_dev",
)
results = dev_evaluator(model)
"""
EmbeddingSimilarityEvaluator: Evaluating the model on the sts_dev dataset:
Cosine-Similarity:      Pearson: 0.8429 Spearman: 0.8366
Model Sparsity: Active Dimensions: 78.3, Sparsity Ratio: 0.9974
"""
# Print the results
print(f"Primary metric: {dev_evaluator.primary_metric}")
# => Primary metric: sts_dev_spearman_cosine
print(f"Primary metric value: {results[dev_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.8366
