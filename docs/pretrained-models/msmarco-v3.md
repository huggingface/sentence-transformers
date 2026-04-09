# MSMARCO Models 
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data consists of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 
## Usage
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v3")

query_embedding = model.encode("How big is London")
passage_embedding = model.encode("London has 9,787,426 inhabitants at the 2011 census")

print("Similarity:", util.cos_sim(query_embedding, passage_embedding))
```


For more details on the usage, see [Applications - Information Retrieval](../../examples/sentence_transformer/applications/retrieve_rerank/README.md)


## Performance
Performance is evaluated on [TREC-DL 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/), which is a query-passage retrieval task where multiple queries have been annotated as with their relevance with respect to the given query.  Further, we evaluate on the [MS Marco Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 

As baseline we show the results for lexical search with BM25 using Elasticsearch.

| Approach       | NDCG@10 (TREC DL 19 Reranking) | MRR@10 (MS Marco Dev) |  Queries (GPU / CPU) | Docs (GPU / CPU)
| ------------- |:-------------: | :---: | :---: | :---: |
| **Models tuned for cosine-similarity** | |
| [sentence-transformers/msmarco-MiniLM-L6-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-v3) | 67.46 | 32.27 | 18,000 / 750 | 2,800 / 180
| [sentence-transformers/msmarco-MiniLM-L12-v3](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L12-v3) | 65.14 | 32.75 | 11,000 / 400 | 1,500 / 90
| [sentence-transformers/msmarco-distilbert-base-v3](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3) | 69.02 | 33.13 | 7,000 / 350 | 1,100 / 70
| [sentence-transformers/msmarco-distilbert-base-v4](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4) | **70.24** | **33.79**| 7,000 / 350 | 1,100 / 70
| [sentence-transformers/msmarco-roberta-base-v3](https://huggingface.co/sentence-transformers/msmarco-roberta-base-v3) | 69.08 | 33.01 | 4,000 / 170 | 540 / 30
| **Models tuned for dot-product** | |
| [sentence-transformers/msmarco-distilbert-base-dot-prod-v3](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-dot-prod-v3) | 68.42 | 33.04 | 7,000 / 350 | 1100 / 70
| [sentence-transformers/msmarco-roberta-base-ance-firstp](https://huggingface.co/sentence-transformers/msmarco-roberta-base-ance-firstp) | 67.84 | 33.01 | 4,000 / 170 | 540 / 30
| [sentence-transformers/msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | **71.04** | **34.43** | 7,000 / 350 | 1100 / 70
| **Previous approaches** |  |  |
| BM25 (Elasticsearch)   | 45.46 | 17.29  |
| [sentence-transformers/msmarco-distilroberta-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilroberta-base-v2) | 65.65 |  28.55    |  
| [sentence-transformers/msmarco-roberta-base-v2](https://huggingface.co/sentence-transformers/msmarco-roberta-base-v2) | 67.18 | 29.17 | 
| [sentence-transformers/msmarco-distilbert-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2) | 68.35 | 30.77 |

**Notes:**
- We provide two type of models, one tuned for **cosine-similarity**, the other for **dot-product**. Make sure to use the right method to compute the similarity between query and passages.
- Models tuned for **cosine-similarity** will prefer the retrieval of shorter passages, while models for **dot-product** will prefer the retrieval of longer passages. Depending on your task, you might prefer the one or the other type of model.
- **sentence-transformers/msmarco-roberta-base-ance-firstp** is the MSMARCO Dev Passage Retrieval ANCE(FirstP) 600K model from [ANCE](https://github.com/microsoft/ANCE). This model should be used with dot-product instead of cosine similarity.
- **sentence-transformers/msmarco-distilbert-base-tas-b** uses the model from [sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco). See the linked documentation / paper for more details.
- Encoding speeds are per second and were measured on a V100 GPU and an 8 core Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz


## Changes in v3
The models from v2 have been used for find for all training queries similar passages. An [MS MARCO Cross-Encoder](ce-msmarco.md) based on the electra-base-model has been then used to classify if these retrieved passages answer the question.

If they received a low score by the cross-encoder, we saved them as hard negatives: They got a high score from the bi-encoder, but a low-score from the (better) cross-encoder.

We then trained the v2 models with these new hard negatives.

## Version History 

- [Version 2](msmarco-v2.md)
- [Version 1](msmarco-v1.md)
