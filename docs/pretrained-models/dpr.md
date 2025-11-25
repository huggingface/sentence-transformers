# DPR-Models
In [Dense Passage Retrieval  for Open-Domain Question Answering](https://huggingface.co/papers/2004.04906)  Karpukhin et al. trained models based on [Google's Natural Questions dataset](https://ai.google.com/research/NaturalQuestions):
- **facebook-dpr-ctx_encoder-single-nq-base** 
- **facebook-dpr-question_encoder-single-nq-base**

They also trained models on the combination of Natural Questions, TriviaQA, WebQuestions, and CuratedTREC.
- **facebook-dpr-ctx_encoder-multiset-base** 
- **facebook-dpr-question_encoder-multiset-base**


There is one model to encode passages and one model to encode question / queries.

## Usage

To encode paragraphs, you need to provide a title (e.g. the Wikipedia article title) and the text passage. These must be separated with a `[SEP]` token.  For encoding paragraphs, we use the **ctx_encoder**.

Queries are encoded with **question_encoder**:
```python
from sentence_transformers import SentenceTransformer, util

passage_encoder = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")

passages = [
    "London [SEP] London is the capital and largest city of England and the United Kingdom.",
    "Paris [SEP] Paris is the capital and most populous city of France.",
    "Berlin [SEP] Berlin is the capital and largest city of Germany by both area and population.",
]

passage_embeddings = passage_encoder.encode(passages)

query_encoder = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
query = "What is the capital of England?"
query_embedding = query_encoder.encode(query)

# Important: You must use dot-product, not cosine_similarity
scores = util.dot_score(query_embedding, passage_embeddings)
print("Scores:", scores)
```

**Important note:** When you use these models, you have to use them with dot-product (e.g. as implemented in `util.dot_score`) and not with cosine similarity.
