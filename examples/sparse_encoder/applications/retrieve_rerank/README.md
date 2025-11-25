# Retrieve & Re-Rank

In [Semantic Search](../semantic_search/README.md) we have shown how to use SparseEncoder to compute embeddings for queries, sentences, and paragraphs and how to use this for semantic search. For complex search tasks, for example question answering retrieval, the search can significantly be improved by using **Retrieve & Re-Rank**. Note that a detailed explanation with dense embeddings produced by Bi-Encoder is accessible [here](../../../sentence_transformer/applications/retrieve_rerank/README.md).

## Overview

The Retrieve & Re-Rank approach consists of two stages:

1. **Retrieval Stage**: Use fast but less accurate methods (SparseEncoder/bi-encoders) to retrieve a larger set of potentially relevant documents
1. **Re-Ranking Stage**: Use more sophisticated but slower models (cross-encoders) to re-rank the retrieved documents for better precision

This approach combines the efficiency of first-stage retrieval with the accuracy of second-stage re-ranking.

## Interactive Demo: Simple Wikipedia Search

**File**: [retrieve_rerank_simple_wikipedia.ipynb](retrieve_rerank_simple_wikipedia.ipynb) [ [Colab Version](https://colab.research.google.com/github/huggingface/sentence-transformers/blob/main/examples/sparse_encoder/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]

This Jupyter notebook provides an interactive demonstration of retrieve & re-rank over [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) as corpus. The example allows you to:

- Input queries or questions
- Compare different retrieval methods:
  - **BM25** (lexical/keyword search)
  - **Sparse Encoder** [ibm-granite/granite-embedding-30m-sparse](https://huggingface.co/ibm-granite/granite-embedding-30m-sparse)
  - **Dense Encoder** [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
- And re-ranking results using a CrossEncoder [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

## Comprehensive Evaluation: Hybrid Search Pipeline

**File**: [hybrid_search.py](hybrid_search.py)

This script provides a complete evaluation pipeline comparing different retrieval and re-ranking approaches on a given dataset (here in our example [NanoNFCorpus](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus)). It includes:

1. **Sparse Retrieval** using [ibm-granite/granite-embedding-30m-sparse](https://huggingface.co/ibm-granite/granite-embedding-30m-sparse)
1. **Dense Retrieval** using [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1)
1. **Re-ranking** both sparse and dense results with [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
1. **Hybrid Search** using Reciprocal Rank Fusion [ReciprocalRankFusionEvaluator](../../../../docs/package_reference/sparse_encoder/evaluation.md#reciprocalrankfusionevaluator)
1. **Hybrid Re-ranking** applying cross-encoder to fused results

**Output**: The script generates comprehensive metrics and saves results in the `runs/` directory.

### Evaluation Results

Example results from running the hybrid search evaluation on [NanoNFCorpus](https://huggingface.co/datasets/zeta-alpha-ai/NanoNFCorpus):

```
================================================================================
EVALUATION SUMMARY
================================================================================
METHOD                            NDCG@10     MRR@10        MAP
--------------------------------------------------------------------------------
Sparse Retrieval                    32.10      47.27      28.29
Dense Retrieval                     27.35      41.59      22.79
Sparse + Reranking                  37.35      57.19      32.12
Dense + Reranking                   37.56      58.27      31.93
Hybrid RRF                          32.62      49.63      22.51
Hybrid RRF + Reranking              36.16      55.77      26.99
================================================================================
```

**Key Observations**:

- Re-ranking consistently improves performance across all retrieval methods
- Sparse retrieval seems to already give strong first results
- Both sparse and dense re-ranking achieve similar high performance
- Hybrid approaches provide balanced results

## Pre-trained Models

### Sparse Encoder (Retrieval)

The SparseEncoder produces embeddings independently for your paragraphs and for your search queries. You can use it like this:

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

docs = [
    "My first paragraph. That contains information",
    "Python is a programming language.",
]
document_embeddings = model.encode_document(docs)

query = "What is Python?"
query_embedding = model.encode_query(query)
```

For pre-trained Sparse Encoder models, see: [Pretrained Sparse-Encoders](../../../../docs/sparse_encoder/pretrained_models.md).

### Cross-Encoders (Re-Ranker)

For pre-trained Cross Encoder models, see: [MS MARCO Cross-Encoders](../../../../docs/cross_encoder/pretrained_models.md#ms-marco)
