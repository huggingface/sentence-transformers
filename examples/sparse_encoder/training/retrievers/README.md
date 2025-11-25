# Information Retrieval

```{eval-rst}
Sparse retriever models are often SPLADE models that map queries and documents to high-dimensional sparse vectors. Given a query, relevant documents are retrieved by computing the dot-product (or cosine similarity) between the query's sparse vector and the sparse vectors of all documents in a collection. This process is often made highly efficient using inverted indexes and algorithms to speed up the inference.

Many sparse retriever models are trained on MS MARCO, and you can find an example of this training methodologies here: 

- `Sparse Encoder > Training Examples > MS MARCO <../ms_marco/README.html>`_
```

However, you will likely achieve the best results by training on your specific dataset. This page will outline example training scripts that you can adapt for your own data, focusing on sparse retrieval.

Example scripts could be:

- **[train_splade_gooaq.py](train_splade_gooaq.py)**:

  ```{eval-rst}
  This example uses :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` (which internally uses :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`) on (query, positive_passage) pair data mined from a dataset like `GooAQ <https://huggingface.co/datasets/sentence-transformers/gooaq>`_. The goal is to train a SPLADE models where the query and its positive_passage have high similarity, and are dissimilar to other passages in the batch (in-batch negatives).

  The model would be evaluated on its retrieval performance on datasets like `MS MARCO <https://huggingface.co/datasets/sentence-transformers/NanoMSMARCO-bm25>`_, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, or `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ using appropriate retrieval metrics (e.g., nDCG@k, MRR@k) via an evaluator like :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`.
  ```

- **[train_splade_nq.py](train_splade_nq.py)**:

  ```{eval-rst}
  This example also uses :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` (similarly utilizing :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`) and trains on the `NQ (natural questions) <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_ dataset. It showcases an alternative configuration or approach for training SPLADE models on question-answering data for sparse retrieval.
  ```

- **[train_csr_nq.py](train_csr_nq.py)**:

  ```{eval-rst}
  This example uses :class:`~sentence_transformers.sparse_encoder.losses.CSRLoss` (which internally uses :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`) for sparse retrievers. It trains on data from datasets like `NQ (natural questions) <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_. The script demonstrates how to train a sparse model with a SparseAutoEncoder head on top of a SentenceTransformer model for retrieval tasks.
  ```

## SparseMultipleNegativesRankingLoss (MNRL)

```{eval-rst}
The :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` is a very common and effective loss for training  sparse models for retrieval. It takes (query, positive_document) pairs. For each query in a batch, its corresponding positive_document is considered a positive, and all other documents (from other pairs in the batch) are considered negatives (in-batch negatives). The loss aims to maximize the similarity score (e.g., dot-product) between the query and its positive_document, while minimizing the similarity scores between the query and all negative_documents.

For sparse models, the output embeddings are sparse, and the similarity is typically a dot product. It's crucial to have a large enough batch size to provide a sufficient number of informative negatives.
```

## Inference & Evaluation

Once a sparse retriever is trained, you would typically encode your entire document corpus into sparse vectors and store them in an efficient index (e.g., an inverted index).

Given a new query:

1. Encode the query into its sparse vector using the trained sparse retriever.
1. Use this query vector to search the indexed document vectors to find the top-k most similar documents (highest dot-product scores).

An example of how inference might look (conceptual):

```python
from sentence_transformers import SparseEncoder, util

# 1. Load my trained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 2. Encode a corpus of texts using the SparseEncoder model
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 3. Encode the user queries using the same SparseEncoder model
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]
query_embeddings = model.encode(queries, convert_to_tensor=True)

# 4. Use the similarity function to compute the similarity scores between the query and corpus embeddings
top_k = min(5, len(corpus))  # Find at most 5 sentences of the corpus for each query sentence
results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=model.similarity)

# 5. Sort the results and print the top 5 most similar sentences for each query
for query_id, query in enumerate(queries):
    pointwise_scores = model.intersection(query_embeddings[query_id], corpus_embeddings)

    print(f"Query: {query}")
    for res in results[query_id]:
        corpus_id, score = res.values()
        sentence = corpus[corpus_id]

        pointwise_score = model.decode(pointwise_scores[corpus_id], top_k=10)

        token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in pointwise_score])

        print(f"Score: {score:.4f} - Sentence: {sentence} - Top influential tokens: {token_scores}")
    print("")

"""
Query: A man is eating pasta.
Score: 21.0064 - Sentence: A man is eating food. - Top influential tokens: ("man", 5.48), ("eating", 3.83), ("eat", 3.15), ("men", 3.12), ("food", 1.78), ("male", 0.87), ("person", 0.62), ("a", 0.39), ("hunger", 0.28), ("meat", 0.27)
Score: 18.2966 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("man", 4.85), ("eating", 3.49), ("eat", 3.02), ("men", 2.74), ("male", 0.68), ("food", 0.66), ("person", 0.58), ("a", 0.51), ("meat", 0.36), ("culture", 0.27)
Score: 10.1537 - Sentence: A man is riding a horse. - Top influential tokens: ("man", 4.85), ("men", 3.11), ("male", 0.68), ("a", 0.60), ("person", 0.59), ("animal", 0.21), ("adam", 0.04), ("sex", 0.03), ("god", 0.02), ("who", 0.01)
Score: 6.5993 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("man", 3.31), ("men", 1.58), ("a", 0.51), ("male", 0.41), ("person", 0.34), ("on", 0.17), ("animal", 0.16), ("wearing", 0.04), ("god", 0.04), ("culture", 0.02)
Score: 5.2185 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("men", 2.60), ("man", 2.51), ("a", 0.09), ("murder", 0.01), ("said", 0.00)

Query: Someone in a gorilla costume is playing a set of drums.
Score: 16.4688 - Sentence: A monkey is playing drums. - Top influential tokens: ("drums", 4.38), ("drum", 2.27), ("play", 2.16), ("playing", 1.77), ("drummer", 0.80), ("dance", 0.63), ("monkey", 0.55), ("music", 0.48), ("a", 0.40), ("sound", 0.39)
Score: 8.6239 - Sentence: A woman is playing violin. - Top influential tokens: ("play", 2.12), ("playing", 1.79), ("person", 0.67), ("dance", 0.58), ("music", 0.55), ("instrument", 0.52), ("guitar", 0.39), ("a", 0.35), ("wearing", 0.32), ("player", 0.21)
Score: 2.7615 - Sentence: A man is riding a horse. - Top influential tokens: ("person", 0.91), ("a", 0.49), ("man", 0.45), ("animal", 0.37), ("sport", 0.32), ("savage", 0.10), ("billy", 0.06), ("dance", 0.02), ("god", 0.01), ("hunting", 0.01)
Score: 2.4471 - Sentence: A man is eating a piece of bread. - Top influential tokens: ("person", 0.90), ("man", 0.45), ("a", 0.42), ("someone", 0.29), ("animal", 0.08), ("god", 0.07), ("ritual", 0.07), ("culture", 0.07), ("something", 0.05), ("who", 0.03)
Score: 2.3295 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("person", 0.53), ("a", 0.42), ("man", 0.31), ("sport", 0.27), ("animal", 0.27), ("savage", 0.09), ("character", 0.09), ("wearing", 0.07), ("symbol", 0.07), ("hunting", 0.05)

Query: A cheetah chases prey on across a field.
Score: 16.3185 - Sentence: A cheetah is running behind its prey. - Top influential tokens: ("che", 3.80), ("##eta", 3.72), ("prey", 2.77), ("hunting", 0.75), ("behavior", 0.70), ("##h", 0.62), ("movement", 0.45), ("animal", 0.33), ("predator", 0.30), ("chasing", 0.29)
Score: 1.9917 - Sentence: A monkey is playing drums. - Top influential tokens: ("animal", 0.43), ("a", 0.41), ("behavior", 0.28), ("movement", 0.18), ("bird", 0.17), ("dance", 0.16), ("species", 0.07), ("dog", 0.06), ("game", 0.05), ("they", 0.05)
Score: 1.4335 - Sentence: A man is riding a white horse on an enclosed ground. - Top influential tokens: ("a", 0.43), ("animal", 0.35), ("hunting", 0.21), ("movement", 0.17), ("breed", 0.12), ("sport", 0.08), ("bird", 0.04), ("dog", 0.02)
Score: 1.4071 - Sentence: A man is riding a horse. - Top influential tokens: ("a", 0.51), ("animal", 0.48), ("movement", 0.27), ("sport", 0.10), ("hunting", 0.04), ("dance", 0.01)
Score: 1.3531 - Sentence: Two men pushed carts through the woods. - Top influential tokens: ("hunting", 0.49), ("cross", 0.41), ("move", 0.22), ("escape", 0.08), ("a", 0.07), ("across", 0.05), ("obstacle", 0.01), ("deer", 0.01), ("they", 0.01)
"""
```

```{eval-rst}
Evaluation is typically done using standard information retrieval metrics like nDCG@k, MRR@k, Recall@k, and Precision@k on benchmark datasets. The :class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator` can be used for this purpose.
```
