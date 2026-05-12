import logging

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load a model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Load the NFcorpus IR dataset (https://huggingface.co/datasets/BeIR/nfcorpus, https://huggingface.co/datasets/BeIR/nfcorpus-qrels)
corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
relevant_docs_data = load_dataset("BeIR/nfcorpus-qrels", split="test")

# For this dataset, we want to concatenate the title and texts for the corpus
corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])

# Convert the datasets to dictionaries
corpus = dict(zip(corpus["_id"], corpus["text"]))  # Our corpus (cid => document)
queries = dict(zip(queries["_id"], queries["text"]))  # Our queries (qid => question)
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the SparseInformationRetrievalEvaluator computes different IR metrics.
ir_evaluator = SparseInformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="BeIR-nfcorpus-subset-test",
    show_progress_bar=True,
    batch_size=16,
)

# Run evaluation
results = ir_evaluator(model)
"""
Queries: 323
Corpus: 3269

Score-Function: dot
Accuracy@1: 49.23%
Accuracy@3: 63.47%
Accuracy@5: 66.56%
Accuracy@10: 71.83%
Precision@1: 49.23%
Precision@3: 39.63%
Precision@5: 33.13%
Precision@10: 25.23%
Recall@1: 6.08%
Recall@3: 11.60%
Recall@5: 13.47%
Recall@10: 17.01%
MRR@10: 0.5706
NDCG@10: 0.3530
MAP@100: 0.1778
Model Query Sparsity: Active Dimensions: 40.0, Sparsity Ratio: 0.9987
Model Corpus Sparsity: Active Dimensions: 206.2, Sparsity Ratio: 0.9932
Average FLOPS: 4.7
"""
# Print the results
print(f"Primary metric: {ir_evaluator.primary_metric}")
# => Primary metric: BeIR-nfcorpus-subset-test_dot_ndcg@10
print(f"Primary metric value: {results[ir_evaluator.primary_metric]:.4f}")
# => Primary metric value: 0.Primary metric value: 0.3530
