import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTranslationEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model, not mutilingual but hope to see some on the hub soon
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a parallel sentences dataset
dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

# Initialize the TranslationEvaluator using the same texts from two languages
translation_evaluator = SparseTranslationEvaluator(
    source_sentences=dataset["english"],
    target_sentences=dataset["non_english"],
    name="news-commentary-en-nl",
)
results = translation_evaluator(model)
"""
Evaluating translation matching Accuracy of the model on the news-commentary-en-nl dataset:
Accuracy src2trg: 41.40
Accuracy trg2src: 47.60
Model Sparsity: Active Dimensions: 112.3, Sparsity Ratio: 0.9963
"""
# Print the results
print(f"Primary metric: {translation_evaluator.primary_metric}")
# => Primary metric: news-commentary-en-nl_mean_accuracy
print(f"Primary metric value: {results[translation_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.4450
