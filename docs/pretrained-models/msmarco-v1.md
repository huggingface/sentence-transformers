# MSMARCO Models
[MS MARCO](https://microsoft.github.io/msmarco/) is a large scale information retrieval corpus that was created based on real user search queries using Bing search engine. The provided models can be used for semantic search, i.e., given keywords / a search phrase / a question, the model will find passages that are relevant for the search query.

The training data consists of over 500k examples, while the complete  corpus consist of over 8.8 Million passages.
 


## Version History 

### v1
Version 1 models were trained on the training set of MS Marco Passage retrieval task. The models were trained using in-batch negative sampling via the MultipleNegativesRankingLoss with a scaling factor of 20 and a batch size of 128.

They can be used like this:
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/distilroberta-base-msmarco-v1")

query_embedding = model.encode("[QRY] " + "How big is London")
passage_embedding = model.encode("[DOC] " + "London has 9,787,426 inhabitants at the 2011 census")

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
```

**Models**:
- **distilroberta-base-msmarco-v1** - Performance MSMARCO dev dataset (queries.dev.small.tsv) MRR@10: 23.28