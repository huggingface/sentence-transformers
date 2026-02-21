# MSMARCO Models 
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data consists of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 
## Usage
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("msmarco-distilbert-dot-v5")

query_embedding = model.encode("How big is London")
passage_embedding = model.encode([
    "London has 9,787,426 inhabitants at the 2011 census",
    "London is known for its financial district",
])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
```


For more details on the usage, see [Applications - Information Retrieval](../../examples/sentence_transformer/applications/retrieve_rerank/README.md)


## Performance
Performance is evaluated on [TREC-DL 2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019) and [TREC-DL 2020](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020), which are a query-passage retrieval task where multiple queries have been annotated as with their relevance with respect to the given query.  Further, we evaluate on the [MS Marco Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 


| Approach       | MRR@10 (MS Marco Dev) | NDCG@10 (TREC DL 19 Reranking) | NDCG@10 (TREC DL 20 Reranking) |   Queries (GPU / CPU) | Docs (GPU / CPU)
| ------------- | :-------------: | :-------------: | :---: | :---: | :---: |
| **Models tuned with normalized embeddings** | |
| [msmarco-MiniLM-L6-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5) | 32.27 | 67.46 | 64.73 | 18,000 / 750 | 2,800 / 180
| [msmarco-MiniLM-L12-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L12-cos-v5) | 32.75 | 65.14 | 67.48 | 11,000 / 400 | 1,500 / 90
| [msmarco-distilbert-cos-v5](https://huggingface.co/sentence-transformers/msmarco-distilbert-cos-v5) | 33.79 | 70.24 | 66.24  | 7,000 / 350 | 1,100 / 70
| [multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) | | 65.55 | 64.66 | 18,000 / 750 | 2,800 / 180 
| [multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1) | | 67.59 | 66.46 | 7,000 / 350 | 1,100 / 70
| [multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) | | 67.78 |	69.87 | 4,000 / 170 |  540 / 30
| **Models tuned for dot-product** | |
| [msmarco-distilbert-base-tas-b](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | 34.43 | 71.04 | 69.78  | 7,000 / 350 | 1100 / 70
| [msmarco-distilbert-dot-v5](https://huggingface.co/sentence-transformers/msmarco-distilbert-dot-v5) | 37.25 | 70.14 | 71.08 | 7,000 / 350 | 1100 / 70
| [msmarco-bert-base-dot-v5](https://huggingface.co/sentence-transformers/msmarco-bert-base-dot-v5) | 38.08 | 70.51	| 73.45 | 4,000 / 170 |  540 / 30
| [multi-qa-MiniLM-L6-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1) | | 66.70 | 65.98 | 18,000 / 750 | 2,800 / 180 
| [multi-qa-distilbert-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-dot-v1) | | 68.05 | 70.49 | 7,000 / 350 | 1,100 / 70
| [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) | | 70.66 |	71.18 | 4,000 / 170 |  540 / 30


**Notes:**
- We provide two type of models: One that produces **normalized embedding** and can be used with dot-product, cosine-similarity or euclidean distance (all three scoring function will produce the same results). The models tuned for **dot-product** will produce embeddings of different lengths and must be used with dot-product to find close items in a vector space.
- Models with normalized embeddings will prefer the retrieval of shorter passages, while models tuned for **dot-product** will prefer the retrieval of longer passages. Depending on your task, you might prefer the one or the other type of model.
- Encoding speeds are per second and were measured on a V100 GPU and an 8 core Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz


## Changes in v5
- Models with normalized embeddings were added: These are the v3 cosine-similarity models, but with an additional normalize layer on-top.
- New models trained with MarginMSE loss trained: msmarco-distilbert-dot-v5 and msmarco-bert-base-dot-v5

## Changes in v4
- Just one new model was trained with better hard negatives, leading to a small improvement compared to v3

## Changes in v3
The models from v2 have been used for find for all training queries similar passages. An [MS MARCO Cross-Encoder](ce-msmarco.md) based on the electra-base-model has been then used to classify if these retrieved passages answer the question.

If they received a low score by the cross-encoder, we saved them as hard negatives: They got a high score from the bi-encoder, but a low-score from the (better) cross-encoder.

We then trained the v2 models with these new hard negatives.

## Version History 

- [Version 3](msmarco-v3.md)
- [Version 2](msmarco-v2.md)
- [Version 1](msmarco-v1.md)
