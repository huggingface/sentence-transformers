import logging
import os

import torch
from datasets import load_dataset

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.models import Transformer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator import ReciprocalRankFusionEvaluator
from sentence_transformers.sparse_encoder.models import SpladePooling

# Configure logging
logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directories
os.makedirs("runs", exist_ok=True)

###########################
# 1. Load the NanoNFCorpus IR dataset (https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus)
###########################
logger.info("=" * 80)
logger.info("STEP 1: LOADING DATASET")
logger.info("=" * 80)

dataset_path = "zeta-alpha-ai/NanoNFCorpus"
logger.info(f"Loading dataset: {dataset_path}")
corpus = load_dataset(dataset_path, "corpus", split="train")
queries = load_dataset(dataset_path, "queries", split="train")
qrels = load_dataset(dataset_path, "qrels", split="train")

# Process the dataset
logger.info("Processing dataset into required format")
corpus_dict = {sample["_id"]: sample["text"] for sample in corpus if len(sample["text"]) > 0}
queries_dict = {sample["_id"]: sample["text"] for sample in queries if len(sample["text"]) > 0}
qrels_dict = {}
for sample in qrels:
    if sample["query-id"] not in qrels_dict:
        qrels_dict[sample["query-id"]] = set()
    qrels_dict[sample["query-id"]].add(sample["corpus-id"])

# Convert HuggingFace Dataset to a lookup dict by _id
corpus_lookup = {item["_id"]: item for item in corpus}
logger.info(f"Dataset loaded. Corpus size: {len(corpus_dict)}, Queries: {len(queries_dict)}")

#########################
# 2. Sparse Retrieval
#########################
logger.info("=" * 80)
logger.info("STEP 2: EVALUATING SPARSE RETRIEVAL")
logger.info("=" * 80)

sparse_encoder_model_name = "ibm-granite/granite-embedding-30m-sparse"
logger.info(f"Loading sparse encoder model: {sparse_encoder_model_name}")
sparse_encoder = SparseEncoder(
    modules=[Transformer(sparse_encoder_model_name, transformer_task="fill-mask"), SpladePooling("max")]
)
sparse_encoder_similarity_fn_name = sparse_encoder.similarity_fn_name

# Create output directory
os.makedirs(f"runs/{sparse_encoder_model_name}/result", exist_ok=True)

logger.info("Running sparse retrieval evaluation on NanoNFCorpus dataset")
evaluator = SparseInformationRetrievalEvaluator(
    queries=queries_dict,
    corpus=corpus_dict,
    relevant_docs=qrels_dict,
    show_progress_bar=True,
    batch_size=12,
    write_predictions=True,
)
sparse_results = evaluator(
    sparse_encoder, output_path=f"runs/{sparse_encoder_model_name}/result", write_predictions=True
)
logger.info(
    f"Sparse retrieval evaluation complete. NDCG@10: {sparse_results.get(f'{sparse_encoder_similarity_fn_name}_ndcg@10'):.4f}"
)

del sparse_encoder
torch.cuda.empty_cache()
logger.info("Freed sparse encoder resources")

#########################
# 3. Dense Retrieval
#########################
logger.info("=" * 80)
logger.info("STEP 2: EVALUATING DENSE RETRIEVAL")
logger.info("=" * 80)

bi_encoder_model_name = "multi-qa-MiniLM-L6-cos-v1"
logger.info(f"Loading dense encoder model: {bi_encoder_model_name}")
bi_encoder = SentenceTransformer(bi_encoder_model_name)
bi_encoder_similarity_fn_name = bi_encoder.similarity_fn_name

# Create output directory
os.makedirs(f"runs/{bi_encoder_model_name}/result", exist_ok=True)

logger.info("Running dense retrieval evaluation on NanoNFCorpus dataset")
evaluator = InformationRetrievalEvaluator(
    queries=queries_dict,
    corpus=corpus_dict,
    relevant_docs=qrels_dict,
    show_progress_bar=True,
    batch_size=12,
    write_predictions=True,
)
dense_results = evaluator(bi_encoder, output_path=f"runs/{bi_encoder_model_name}/result")
logger.info(
    f"Dense retrieval evaluation complete. NDCG@10: {dense_results.get(f'{bi_encoder_similarity_fn_name}_ndcg@10'):.4f}"
)

del bi_encoder
torch.cuda.empty_cache()
logger.info("Freed dense encoder resources")


#########################
# 4. Reranking Sparse Results
#########################
logger.info("=" * 80)
logger.info("STEP 4: RERANKING SPARSE RETRIEVAL RESULTS")
logger.info("=" * 80)

# Load cross-encoder for reranking
cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
logger.info(f"Loading cross-encoder model for reranking: {cross_encoder_model_name}")
cross_encoder = CrossEncoder(cross_encoder_model_name)

# Load sparse prediction results
logger.info("Loading sparse retrieval results for reranking")
sparse_pred_data = load_dataset(
    "json",
    data_files=f"runs/{sparse_encoder_model_name}/result/Information-Retrieval_evaluation_predictions_{sparse_encoder_similarity_fn_name}.jsonl",
)["train"]

# Create samples for sparse reranking
logger.info("Preparing sparse results for reranking")
sparse_samples = [
    {
        "query_id": sample["query_id"],
        "query": sample["query"],
        "positive": [corpus_lookup[doc_id]["text"] for doc_id in qrels_dict[sample["query_id"]]],
        "documents": [corpus_lookup[dict_["corpus_id"]]["text"] for dict_ in sample["results"]],
    }
    for sample in sparse_pred_data
]

# Initialize the evaluator for sparse reranking
sparse_reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=sparse_samples,
    show_progress_bar=True,
)
os.makedirs(f"runs/{sparse_encoder_model_name}/result/rerank_{cross_encoder_model_name}", exist_ok=True)

# Run evaluation
logger.info("Running reranking on sparse retrieval results")
sparse_reranking_results = sparse_reranking_evaluator(
    cross_encoder, output_path=f"runs/{sparse_encoder_model_name}/result/rerank_{cross_encoder_model_name}"
)
logger.info(f"Sparse reranking complete. NDCG@10: {sparse_reranking_results.get('ndcg@10'):.4f}")

#########################
# 5. Reranking Dense Results
#########################
logger.info("=" * 80)
logger.info("STEP 5: RERANKING DENSE RETRIEVAL RESULTS")
logger.info("=" * 80)

# Load dense prediction results
logger.info("Loading dense retrieval results for reranking")
dense_pred_data = load_dataset(
    "json",
    data_files=f"runs/{bi_encoder_model_name}/result/Information-Retrieval_evaluation_predictions_{bi_encoder_similarity_fn_name}.jsonl",
)["train"]

# Create samples for dense reranking
logger.info("Preparing dense results for reranking")
dense_samples = [
    {
        "query_id": sample["query_id"],
        "query": sample["query"],
        "positive": [corpus_lookup[doc_id]["text"] for doc_id in qrels_dict[sample["query_id"]]],
        "documents": [corpus_lookup[dict_["corpus_id"]]["text"] for dict_ in sample["results"]],
    }
    for sample in dense_pred_data
]

# Initialize the evaluator for dense reranking
dense_reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=dense_samples,
    show_progress_bar=True,
)
os.makedirs(f"runs/{bi_encoder_model_name}/result/rerank_{cross_encoder_model_name}", exist_ok=True)

# Run evaluation
logger.info("Running reranking on dense retrieval results")
dense_reranking_results = dense_reranking_evaluator(
    cross_encoder, output_path=f"runs/{bi_encoder_model_name}/result/rerank_{cross_encoder_model_name}"
)
logger.info(f"Dense reranking complete. NDCG@10: {dense_reranking_results.get('ndcg@10'):.4f}")

#########################
# 6. Hybrid Search with RRF
#########################
logger.info("=" * 80)
logger.info("STEP 6: HYBRID SEARCH WITH RECIPROCAL RANK FUSION")
logger.info("=" * 80)

# Initialize the RRF evaluator
rrf_output_path = f"runs/hybrid_search/rrf_{sparse_encoder_model_name}_{bi_encoder_model_name}"
os.makedirs(rrf_output_path, exist_ok=True)

# Create RRF evaluator for hybrid search
logger.info("Setting up Reciprocal Rank Fusion for hybrid search")
rrf_evaluator = ReciprocalRankFusionEvaluator(
    dense_samples=dense_samples,
    sparse_samples=sparse_samples,
    at_k=10,
    rrf_k=60,  # Default RRF constant
    show_progress_bar=True,
    write_predictions=True,
)

# Run evaluation
logger.info("Running Reciprocal Rank Fusion evaluation")
rrf_results = rrf_evaluator(output_path=rrf_output_path)
logger.info(f"Hybrid search with RRF complete. NDCG@10: {rrf_results.get('ndcg@10'):.4f}")

#########################
# 7. Reranking Hybrid Results
#########################
logger.info("=" * 80)
logger.info("STEP 7: RERANKING HYBRID SEARCH RESULTS")
logger.info("=" * 80)

# Load the RRF fusion results for reranking
logger.info("Loading fusion results for reranking")
fused_pred_data = load_dataset(
    "json",
    data_files=f"{rrf_output_path}/ReciprocalRankFusion_evaluation_predictions.jsonl",
)["train"]

# Initialize the reranking evaluator for the fused results
logger.info("Setting up reranking for hybrid search results")
fusion_reranking_evaluator = CrossEncoderRerankingEvaluator(
    samples=fused_pred_data,
    show_progress_bar=True,
)

# Run reranking on the fused results
fusion_reranking_path = f"{rrf_output_path}/rerank_{cross_encoder_model_name}"
os.makedirs(fusion_reranking_path, exist_ok=True)

logger.info("Running reranking on hybrid search results")
fusion_reranking_results = fusion_reranking_evaluator(cross_encoder, output_path=fusion_reranking_path)
logger.info(f"Hybrid results reranking complete. NDCG@10: {fusion_reranking_results.get('ndcg@10'):.4f}")

#########################
# 8. Results Summary
#########################
logger.info("=" * 80)
logger.info("FINAL EVALUATION SUMMARY")
logger.info("=" * 80)

# Get sparse retrieval metrics
sparse_ndcg = sparse_results.get(f"{sparse_encoder_similarity_fn_name}_ndcg@10")
sparse_mrr = sparse_results.get(f"{sparse_encoder_similarity_fn_name}_mrr@10")
sparse_map = sparse_reranking_results.get("base_map")

# Get dense retrieval metrics
dense_ndcg = dense_results.get(f"{bi_encoder_similarity_fn_name}_ndcg@10")
dense_mrr = dense_results.get(f"{bi_encoder_similarity_fn_name}_mrr@10")
dense_map = dense_reranking_results.get("base_map")

# Get sparse reranking metrics
sparse_rerank_ndcg = sparse_reranking_results.get("ndcg@10")
sparse_rerank_mrr = sparse_reranking_results.get("mrr@10")
sparse_rerank_map = sparse_reranking_results.get("map")

# Get dense reranking metrics
dense_rerank_ndcg = dense_reranking_results.get("ndcg@10")
dense_rerank_mrr = dense_reranking_results.get("mrr@10")
dense_rerank_map = dense_reranking_results.get("map")

# Get fusion metrics
rrf_ndcg = rrf_results.get("ndcg@10")
rrf_mrr = rrf_results.get("mrr@10")
rrf_map = rrf_results.get("map")

# Get fusion reranking metrics
rrf_rerank_ndcg = fusion_reranking_results.get("ndcg@10")
rrf_rerank_mrr = fusion_reranking_results.get("mrr@10")
rrf_rerank_map = fusion_reranking_results.get("map")

# Print the metrics
print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)
print(f"{'METHOD':<30} {'NDCG@10':>10} {'MRR@10':>10} {'MAP':>10}")
print("-" * 80)
print(f"{'Sparse Retrieval':<30} {sparse_ndcg * 100:>10.2f} {sparse_mrr * 100:>10.2f} {sparse_map * 100:>10.2f}")
print(f"{'Dense Retrieval':<30} {dense_ndcg * 100:>10.2f} {dense_mrr * 100:>10.2f} {dense_map * 100:>10.2f}")
print(
    f"{'Sparse + Reranking':<30} {sparse_rerank_ndcg * 100:>10.2f} {sparse_rerank_mrr * 100:>10.2f} {sparse_rerank_map * 100:>10.2f}"
)
print(
    f"{'Dense + Reranking':<30} {dense_rerank_ndcg * 100:>10.2f} {dense_rerank_mrr * 100:>10.2f} {dense_rerank_map * 100:>10.2f}"
)
print(f"{'Hybrid RRF':<30} {rrf_ndcg * 100:>10.2f} {rrf_mrr * 100:>10.2f} {rrf_map * 100:>10.2f}")
print(
    f"{'Hybrid RRF + Reranking':<30} {rrf_rerank_ndcg * 100:>10.2f} {rrf_rerank_mrr * 100:>10.2f} {rrf_rerank_map * 100:>10.2f}"
)
print("=" * 80)

# Also log the summary
logger.info("Evaluation complete. Results summary has been printed.")
