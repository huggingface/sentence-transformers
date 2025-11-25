# Semantic Search

Semantic search seeks to improve search accuracy by understanding the semantic meaning of the search query and the corpus to search over. Semantic search can also perform well given synonyms, abbreviations, and misspellings, unlike keyword search engines that can only find documents based on lexical matches.

## Background

The idea behind semantic search is to embed all entries in your corpus, whether they be sentences, paragraphs, or documents, into a vector space. At search time, the query is embedded into the same vector space and the closest embeddings from your corpus are found. These entries should have a high semantic similarity with the query.

![SemanticSearch](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/SemanticSearch.png)

## Symmetric vs. Asymmetric Semantic Search

A **critical distinction** for your setup is *symmetric* vs. *asymmetric semantic search*:

- For **symmetric semantic search** your query and the entries in your corpus are of about the same length and have the same amount of content. An example would be searching for similar questions: Your query could for example be *"How to learn Python online?"* and you want to find an entry like *"How to learn Python on the web?"*. For symmetric tasks, you could potentially flip the query and the entries in your corpus.
  - Related training example: [Quora Duplicate Questions](../../training/quora_duplicate_questions/README.md).
  - Suitable models: [Pre-Trained Sentence Embedding Models](../../../../docs/sentence_transformer/pretrained_models.md)
- For **asymmetric semantic search**, you usually have a **short query** (like a question or some keywords) and you want to find a longer paragraph answering the query. An example would be a query like *"What is Python"* and you want to find the paragraph *"Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy ..."*. For asymmetric tasks, flipping the query and the entries in your corpus usually does not make sense.
  - Related training example: [MS MARCO](../../training/ms_marco/README.md)
  - Suitable models: [Pre-Trained MS MARCO Models](../../../../docs/pretrained-models/msmarco-v5.md)

It is critical **that you choose the right model** for your type of task.

```{eval-rst}
.. tip::

    For asymmetric semantic search, you are recommended to use :meth:`SentenceTransformer.encode_query <sentence_transformers.SentenceTransformer.encode_query>` to encode your queries and :meth:`SentenceTransformer.encode_document <sentence_transformers.SentenceTransformer.encode_document>` to encode your corpus. 
    
    The more general :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` method differs in two ways from :meth:`SentenceTransformer.encode_query <sentence_transformers.SentenceTransformer.encode_query>` and :meth:`SentenceTransformer.encode_document <sentence_transformers.SentenceTransformer.encode_document>`:

    1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" or "document" prompt, if specified in the model's ``prompts`` dictionary.
    2. It sets the ``task`` to "document". If the model has a :class:`~sentence_transformers.models.Router` module, it will use the "query" or "document" task type to route the input through the appropriate submodules.

    Note that :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` is the most general method and can be used for any task, including Information Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three methods will return identical embeddings.

```

## Manual Implementation

```{eval-rst}
For small corpora (up to about 1 million entries), we can perform semantic search with a manual implementation by computing the embeddings for the corpus with :meth:`SentenceTransformer.encode_document <sentence_transformers.SentenceTransformer.encode_document>` as well as for our query with :meth:`SentenceTransformer.encode_query <sentence_transformers.SentenceTransformer.encode_query>`, and then calculating the `semantic textual similarity <../../../../docs/sentence_transformer/usage/semantic_textual_similarity.html>`_ using :func:`SentenceTransformer.similarity <sentence_transformers.SentenceTransformer.similarity>`.
```

For a simple example, see [semantic_search.py](semantic_search.py):

```{eval-rst}

.. sidebar:: Output

   .. code-block:: text

        Query: How do artificial neural networks work?
        Top 5 most similar sentences in corpus:
        (Score: 0.5926) Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.
        (Score: 0.5288) Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.
        (Score: 0.4647) Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
        (Score: 0.1381) Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.
        (Score: 0.0912) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.

        Query: What technology is used for modern space exploration?
        Top 5 most similar sentences in corpus:
        (Score: 0.3754) Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.
        (Score: 0.3669) SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.
        (Score: 0.3452) The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.
        (Score: 0.2625) Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.
        (Score: 0.2275) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.

        Query: How can we address climate change challenges?
        Top 5 most similar sentences in corpus:
        (Score: 0.3760) Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.
        (Score: 0.3144) Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.
        (Score: 0.2948) Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.
        (Score: 0.0420) Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
        (Score: 0.0411) Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.

.. literalinclude:: semantic_search.py
```

## Optimized Implementation

```{eval-rst}
Instead of implementing semantic search by yourself, you can use the :func:`util.semantic_search <sentence_transformers.util.semantic_search>` function.
```

The function accepts the following parameters:

```{eval-rst}
.. autofunction:: sentence_transformers.util.semantic_search
```

By default, up to 100 queries are processed in parallel. Further, the corpus is chunked into set of up to 500k entries. You can increase `query_chunk_size` and `corpus_chunk_size`, which leads to increased speed for large corpora, but also increases the memory requirement.

## Speed Optimization

```{eval-rst}
To get the optimal speed for the :func:`util.semantic_search <sentence_transformers.util.semantic_search>` method, it is advisable to have the ``query_embeddings`` as well as the ``corpus_embeddings`` on the same GPU-device. This significantly boost the performance. Further, we can normalize the corpus embeddings so that each corpus embeddings is of length 1. In that case, we can use dot-product for computing scores.

.. code-block:: python

    corpus_embeddings = corpus_embeddings.to("cuda")
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = query_embeddings.to("cuda")
    query_embeddings = util.normalize_embeddings(query_embeddings)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)
```

## Elasticsearch

[Elasticsearch](https://www.elastic.co/elasticsearch/) has the possibility to [index dense vectors](https://www.elastic.co/what-is/vector-search) and to use them for document scoring. We can easily index embedding vectors, store other data alongside our vectors and, most importantly, efficiently retrieve relevant entries using [approximate nearest neighbor search](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0) (HNSW, see also below) on the embeddings.

For further details, see [semantic_search_quora_elasticsearch.py](semantic_search_quora_elasticsearch.py).

## OpenSearch

[OpenSearch](https://opensearch.org/) is a community-driven, open-source search engine that supports vector search capabilities. It allows you to index dense vectors and perform efficient similarity search using approximate nearest neighbor algorithms. OpenSearch can be used to implement both traditional keyword-based search (BM25) and semantic search, making it possible to compare and combine both approaches.

For an example implementation, see [semantic_search_nq_opensearch.py](semantic_search_nq_opensearch.py), which shows how to use OpenSearch with the Natural Questions dataset, demonstrating both semantic search and BM25 search capabilities.

## Approximate Nearest Neighbor

```{eval-rst}
Searching a large corpus with millions of embeddings can be time-consuming if exact nearest neighbor search is used (like it is used by :func:`util.semantic_search <sentence_transformers.util.semantic_search>`).
```

In that case, Approximate Nearest Neighbor (ANN) can be helpful. Here, the data is partitioned into smaller fractions of similar embeddings. This index can be searched efficiently and the embeddings with the highest similarity (the nearest neighbors) can be retrieved within milliseconds, even if you have millions of vectors. However, the results are not necessarily exact. It is possible that some vectors with high similarity will be missed.

For all ANN methods, there are usually one or more parameters to tune that determine the recall-speed trade-off. If you want the highest speed, you have a high chance of missing hits. If you want high recall, the search speed decreases.

Three popular libraries for approximate nearest neighbor are [Annoy](https://github.com/spotify/annoy), [FAISS](https://github.com/facebookresearch/faiss), and [hnswlib](https://github.com/nmslib/hnswlib/).

Examples:

- [semantic_search_quora_hnswlib.py](semantic_search_quora_hnswlib.py)
- [semantic_search_quora_annoy.py](semantic_search_quora_annoy.py)
- [semantic_search_quora_faiss.py](semantic_search_quora_faiss.py)

## Retrieve & Re-Rank

For complex semantic search scenarios, a two-stage retrieve & re-rank pipeline is advisable:
![InformationRetrieval](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/InformationRetrieval.png)

For further details, see [Retrieve & Re-rank](../retrieve_rerank/README.md).

## Examples

We list a handful of common use cases:

### Similar Questions Retrieval

[semantic_search_quora_pytorch.py](semantic_search_quora_pytorch.py) [ [Colab version](https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing) ] shows an example based on the [Quora duplicate questions](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) dataset. The user can enter a question, and the code retrieves the most similar questions from the dataset using `util.semantic_search`. As model, we use [distilbert-multilingual-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking), which was trained to identify similar questions and supports 50+ languages. Hence, the user can input the question in any of the 50+ languages. This is a **symmetric search task**, as the search queries have the same length and content as the questions in the corpus.

### Similar Publication Retrieval

[semantic_search_publications.py](semantic_search_publications.py) [ [Colab version](https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06?usp=sharing) ] shows an example how to find similar scientific publications. As corpus, we use all publications that have been presented at the EMNLP 2016 - 2018 conferences. As search query, we input the title and abstract of more recent publications and find related publications from our corpus. We use the [SPECTER](https://huggingface.co/sentence-transformers/allenai-specter) model. This is a **symmetric search task**, as the paper in the corpus consists of title & abstract and we search for title & abstract.

### Question & Answer Retrieval

[semantic_search_wikipedia_qa.py](semantic_search_wikipedia_qa.py) [ [Colab Version](https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing) ]: This example uses a model that was trained on the [Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions). It consists of about 100k real Google search queries, together with an annotated passage from Wikipedia that provides the answer. It is an example of an **asymmetric search task**. As corpus, we use the smaller [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page) so that it fits easily into memory.

[retrieve_rerank_simple_wikipedia.ipynb](../retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) [ [Colab Version](https://colab.research.google.com/github/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb) ]: This script uses the [Retrieve & Re-rank](../retrieve_rerank/README.md) strategy and is an example for an **asymmetric search task**. We split all Wikipedia articles into paragraphs and encode them with a bi-encoder. If a new query / question is entered, it is encoded by the same bi-encoder and the paragraphs with the highest cosine-similarity are retrieved. Next, the retrieved candidates are scored by a Cross-Encoder re-ranker and the 5 passages with the highest score from the Cross-Encoder are presented to the user. We use models that were trained on the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset, a dataset with about 500k real queries from Bing search.
