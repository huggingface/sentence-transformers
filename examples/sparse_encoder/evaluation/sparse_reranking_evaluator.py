import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseRerankingEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load a dataset with queries, positives, and negatives
eval_dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation").select(range(1000))

samples = [
    {
        "query": sample["query"],
        "positive": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if is_selected
        ],
        "negative": [
            text
            for is_selected, text in zip(sample["passages"]["is_selected"], sample["passages"]["passage_text"])
            if not is_selected
        ],
    }
    for sample in eval_dataset
]


# Now evaluate using only the documents from the 1000 samples
reranking_evaluator = SparseRerankingEvaluator(
    samples=samples,
    name="ms_marco_dev_small",
    show_progress_bar=True,
    batch_size=32,
)

results = reranking_evaluator(model)
"""
RerankingEvaluator: Evaluating the model on the ms_marco_dev_small dataset:
Queries: 967     Positives: Min 1.0, Mean 1.1, Max 3.0   Negatives: Min 1.0, Mean 7.1, Max 9.0
MAP: 53.41
MRR@10: 54.14
NDCG@10: 65.06
Model Query Sparsity: Active Dimensions: 42.2, Sparsity Ratio: 0.9986
Model Corpus Sparsity: Active Dimensions: 126.5, Sparsity Ratio: 0.9959
"""
# Print the results
print(f"Primary metric: {reranking_evaluator.primary_metric}")
# => Primary metric: ms_marco_dev_small_ndcg@10
print(f"Primary metric value: {results[reranking_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.6506
