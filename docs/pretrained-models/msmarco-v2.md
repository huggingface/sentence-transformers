# MSMARCO Models (Version 2)
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data consists of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 
## Usage
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/msmarco-distilroberta-base-v2")

query_embedding = model.encode("How big is London")
passage_embedding = model.encode("London has 9,787,426 inhabitants at the 2011 census")

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
```


For more details on the usage, see [Applications - Information Retrieval](../../examples/sentence_transformer/applications/retrieve_rerank/README.md)


## Performance
Performance is evaluated on [TREC-DL 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/), which is a query-passage retrieval task where multiple queries have been annotated as with their relevance with respect to the given query.  Further, we evaluate on the [MS Marco Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset. 

As baseline we show the results for lexical search with BM25 using Elasticsearch.

| Approach       | NDCG@10 (TREC DL 19 Reranking) | MRR@10 (MS Marco Dev) |  
| ------------- |:-------------: | :---: |
| BM25 (Elasticsearch)   | 45.46 | 17.29  |
| [sentence-transformers/msmarco-distilroberta-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilroberta-base-v2) | 65.65 |  28.55    |  
| [sentence-transformers/msmarco-roberta-base-v2](https://huggingface.co/sentence-transformers/msmarco-roberta-base-v2) | 67.18 | 29.17 | 
| [sentence-transformers/msmarco-distilbert-base-v2](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v2) | 68.35 | 30.77 |



## Version History 

- [Version 1](msmarco-v1.md)
