# Semantic Search

Semantic search refers to search techniques that go beyond traditional keyword-based search. Instead of relying solely on exact matches of keywords, semantic search aims to understand the meaning and context of the query and the documents being searched. This allows for more relevant and accurate search results, even when the exact keywords may not match.

Sparse embeddings are a type of representation where most of the values are zero, and only a small number of dimensions contain non-zero (a.k.a. active) values. This is in contrast to dense embeddings, where all dimensions typically have non-zero values. Traditional sparse embedding solutions are often lexically based, meaning they rely on exact matches of terms or phrases. However, modern sparse encoders like SPLADE and other sparse encoder models can generate embeddings that capture semantic meaning while still being sparse.

These embeddings can allow for extremely efficient semantic search, as long as the search solution takes good advantage of the fact that the large majority of sparse embedding dimensions are 0. This page shows an example demonstrating how to perform semantic search manually, but also how to integrate a SparseEncoder model with popular vector databases/search systems.

If you aren't familiar with Semantic Search, see the [Sentence Transformers > Semantic Search](../../../sentence_transformer/applications/semantic-search/README.md) for a broader explanation using dense embedding models.

## Manual Search

Manually performing semantic search with sparse encoders is straightforward, and only consists of a few steps:

1. **Load a SparseEncoder model**: Load a pretrained sparse encoder model from the Hugging Face Hub or your local directory.
1. **Encode the corpus**: Use the model to encode a set of documents (the corpus) into sparse embeddings.
1. **Encode the queries**: Encode the user queries into sparse embeddings using the same model.
1. **Compute similarity**: Calculate the similarity between the query embeddings and the corpus embeddings using a suitable similarity function (e.g., cosine similarity, dot product).
1. **Retrieve results**: Sort the results based on similarity scores and return the most relevant documents.
1. **Analyze results**: Optionally, analyze the results to understand which tokens contributed most to the similarity scores.

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. :meth:`SparseEncoder.encode_query <sentence_transformers.sparse_encoder.SparseEncoder.encode_query>`
   #. :meth:`SparseEncoder.encode_document <sentence_transformers.sparse_encoder.SparseEncoder.encode_document>`
   #. :meth:`util.semantic_search <sentence_transformers.util.semantic_search>`
   #. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>`
   #. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_

::

    from sentence_transformers import SparseEncoder, util

    # 1. Load a pretrained SparseEncoder model
    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 2. Encode a corpus of texts using the SparseEncoder model
    corpus = [
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
        "Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.",
        "The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.",
        "SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.",
        "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.",
        "Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.",
        "Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.",
    ]

    # Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True)

    # 3. Encode the user queries using the same SparseEncoder model
    queries = [
        "How do artificial neural networks work?",
        "What technology is used for modern space exploration?",
        "How can we address climate change challenges?",
    ]
    query_embeddings = model.encode_query(queries, convert_to_tensor=True)

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
```

<details><summary>Toggle To See Results</summary>

```python
"""
Query: How do artificial neural networks work?
Score: 16.9053 - Sentence: Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. - Top influential tokens: ("neural", 5.71), ("networks", 3.24), ("network", 2.93), ("brain", 2.10), ("computer", 0.50), ("##uron", 0.32), ("artificial", 0.27), ("technology", 0.27), ("communication", 0.27), ("connection", 0.21)
Score: 13.6119 - Sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. - Top influential tokens: ("artificial", 3.71), ("neural", 3.15), ("networks", 1.78), ("brain", 1.22), ("network", 1.12), ("ai", 1.07), ("machine", 0.39), ("robot", 0.20), ("technology", 0.20), ("algorithm", 0.18)
Score: 2.7373 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("machine", 0.78), ("computer", 0.50), ("technology", 0.32), ("artificial", 0.22), ("robot", 0.21), ("ai", 0.20), ("process", 0.16), ("theory", 0.11), ("technique", 0.11), ("fuzzy", 0.06)
Score: 2.1430 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("technology", 0.42), ("function", 0.41), ("mechanism", 0.21), ("sensor", 0.21), ("device", 0.18), ("process", 0.18), ("generator", 0.13), ("detection", 0.10), ("technique", 0.10), ("tracking", 0.05)
Score: 2.0195 - Sentence: Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments. - Top influential tokens: ("robot", 0.67), ("function", 0.34), ("technology", 0.29), ("device", 0.23), ("experiment", 0.20), ("machine", 0.10), ("artificial", 0.08), ("design", 0.04), ("useful", 0.03), ("they", 0.02)

Query: What technology is used for modern space exploration?
Score: 10.4748 - Sentence: SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond. - Top influential tokens: ("space", 4.40), ("technology", 1.15), ("nasa", 1.06), ("mars", 0.63), ("exploration", 0.52), ("spacecraft", 0.44), ("robot", 0.32), ("rocket", 0.28), ("astronomy", 0.27), ("travel", 0.26)
Score: 9.3818 - Sentence: The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy. - Top influential tokens: ("space", 3.89), ("nasa", 1.09), ("astronomy", 0.93), ("discovery", 0.48), ("instrument", 0.47), ("technology", 0.35), ("device", 0.26), ("spacecraft", 0.25), ("invented", 0.22), ("equipment", 0.22)
Score: 8.5147 - Sentence: Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments. - Top influential tokens: ("technology", 1.39), ("mars", 0.79), ("exploration", 0.78), ("robot", 0.67), ("used", 0.66), ("nasa", 0.52), ("spacecraft", 0.44), ("device", 0.39), ("explore", 0.38), ("travel", 0.25)
Score: 7.6993 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("technology", 1.99), ("tech", 1.76), ("technologies", 1.74), ("equipment", 0.32), ("device", 0.31), ("technological", 0.28), ("mining", 0.22), ("sensor", 0.19), ("tool", 0.18), ("software", 0.11)
Score: 2.5526 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("technology", 1.52), ("machine", 0.27), ("robot", 0.21), ("computer", 0.18), ("engineering", 0.12), ("technique", 0.11), ("science", 0.05), ("technological", 0.05), ("techniques", 0.02), ("innovation", 0.01)

Query: How can we address climate change challenges?
Score: 9.5587 - Sentence: Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities. - Top influential tokens: ("climate", 3.21), ("warming", 2.87), ("weather", 1.58), ("change", 0.46), ("global", 0.41), ("environmental", 0.39), ("storm", 0.19), ("pollution", 0.15), ("environment", 0.11), ("adaptation", 0.08)
Score: 1.3191 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("warming", 0.39), ("pollution", 0.34), ("environmental", 0.15), ("goal", 0.12), ("strategy", 0.07), ("monitoring", 0.07), ("protection", 0.06), ("greenhouse", 0.05), ("safety", 0.02), ("escape", 0.01)
Score: 1.0774 - Sentence: Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time. - Top influential tokens: ("conservation", 0.39), ("sustainability", 0.18), ("environmental", 0.18), ("sustainable", 0.13), ("agriculture", 0.13), ("alternative", 0.07), ("recycling", 0.00)
Score: 0.2401 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("strategy", 0.10), ("success", 0.06), ("foster", 0.04), ("engineering", 0.03), ("innovation", 0.00), ("research", 0.00)
Score: 0.1516 - Sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. - Top influential tokens: ("strategy", 0.09), ("foster", 0.04), ("research", 0.01), ("approach", 0.01), ("engineering", 0.01)
"""
```

</details>
<br>

## Vector Database Search

Alternatively, some vector databases and search engines can be used to perform semantic search with sparse encoders. These systems are designed to efficiently handle large-scale vector data and provide fast retrieval of relevant documents. They can leverage the sparsity of the embeddings to optimize storage and search operations.

The overall structure is similar to the manual search, but the vector database handles the indexing and retrieval of documents. The steps are approximately as follows:

1. **Encode the corpus**: Load your data and encode the documents using a pretrained sparse encoder.
1. **Indexing**: The documents and their sparse embeddings are indexed in the vector database.
1. **Encode the query**: User queries are encoded with the same sparse encoder.
1. **Retrieval**: The vector database performs a similarity search to find the most relevant documents.
1. **Results**: Search results are returned with their similarity scores and document content.

The advantages of Sparse Vectors for search are:

- **Efficiency**: Sparse vectors (where most values are zero) can be stored and searched more efficiently than dense vectors.
- **Interpretability**: Non-zero dimensions in sparse embeddings often correspond to specific tokens, allowing you to understand which tokens contributed to the similarity score.
- **Exact Matching**: Sparse vectors can preserve exact term matching signals that might be lost in dense embeddings.

## Qdrant Integration

This example demonstrates how to set up Qdrant for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [semantic_search_qdrant.py](semantic_search_qdrant.py) or below:

### Prerequisites:

- Qdrant running locally (or accessible), see the [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/) for more details.
- The Qdrant Python client must be installed:
  ```bash
  pip install qdrant-client
  ```

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. :meth:`SparseEncoder.encode_query <sentence_transformers.sparse_encoder.SparseEncoder.encode_query>`
   #. :meth:`SparseEncoder.encode_document <sentence_transformers.sparse_encoder.SparseEncoder.encode_document>`
   #. :meth:`semantic_search_qdrant <sentence_transformers.sparse_encoder.search_engines.semantic_search_qdrant>`
   #. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_
   #. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

:: 

    import time

    from datasets import load_dataset
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 4. Encode the corpus
    corpus_embeddings = sparse_model.encode_document(
        corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True
    )

    # Initially, we don't have a qdrant index yet
    corpus_index = None
    while True:
        # 5. Encode the queries using the full precision
        start_time = time.time()
        query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
        print(f"Encoding time: {time.time() - start_time:.6f} seconds")

        # 6. Perform semantic search using qdrant
        results, search_time, corpus_index = semantic_search_qdrant(
            query_embeddings,
            corpus_index=corpus_index,
            corpus_embeddings=corpus_embeddings if corpus_index is None else None,
            top_k=5,
            output_index=True,
        )

        # 7. Output the results
        print(f"Search time: {search_time:.6f} seconds")
        for query, result in zip(queries, results):
            print(f"Query: {query}")
            for entry in result:
                print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
            print("")

        # 8. Prompt for more queries
        queries = [input("Please enter a question: ")]
```

## OpenSearch Integration

This example demonstrates how to set up OpenSearch for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [semantic_search_opensearch.py](semantic_search_opensearch.py) or below:

### Prerequisites:

- OpenSearch running locally (or accessible), see [OpenSearch locally](https://docs.opensearch.org/docs/latest/getting-started/quickstart/) for more details.
- Further, the OpenSearch Python client must be installed: https://docs.opensearch.org/docs/latest/clients/python-low-level/, e.g.:
  ```bash
    pip install opensearch-py
  ```
- This script was created for `opensearch` v2.15.0+.

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. :meth:`SparseEncoder.encode_query <sentence_transformers.sparse_encoder.SparseEncoder.encode_query>`
   #. :meth:`SparseEncoder.encode_document <sentence_transformers.sparse_encoder.SparseEncoder.encode_document>`
   #. :meth:`semantic_search_opensearch <sentence_transformers.sparse_encoder.search_engines.semantic_search_opensearch>`
   #. `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill <https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill>`_
   #. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

::
  
    import time

    from datasets import load_dataset

    from sentence_transformers import SparseEncoder
    from sentence_transformers.models import Router
    from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_opensearch

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]
    print(f"Finish loading data. Corpus size: {len(corpus)}")

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
    doc_encoder = MLMTransformer(model_id)
    router = Router.for_query_document(
        query_modules=[
            SparseStaticEmbedding.from_json(
                model_id,
                tokenizer=doc_encoder.tokenizer,
                frozen=True,
            ),
        ],
        document_modules=[
            doc_encoder,
            SpladePooling("max", activation_function="log1p_relu"),
        ],
    )

    sparse_model = SparseEncoder(modules=[router], similarity_fn_name="dot")

    print("Start encoding corpus...")
    start_time = time.time()
    # 4. Encode the corpus
    corpus_embeddings = sparse_model.encode_document(
        corpus, convert_to_sparse_tensor=True, batch_size=32, show_progress_bar=True
    )
    corpus_embeddings_decoded = sparse_model.decode(corpus_embeddings)
    print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")

    corpus_index = None
    while True:
        # 5. Encode the queries using inference-free mode
        start_time = time.time()
        query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
        query_embeddings_decoded = sparse_model.decode(query_embeddings)
        print(f"Query encoding time: {time.time() - start_time:.6f} seconds")

        # 6. Perform semantic search using OpenSearch
        results, search_time, corpus_index = semantic_search_opensearch(
            query_embeddings_decoded,
            corpus_embeddings_decoded=corpus_embeddings_decoded if corpus_index is None else None,
            corpus_index=corpus_index,
            top_k=5,
            output_index=True,
        )

        # 7. Output the results
        print(f"Search time: {search_time:.6f} seconds")
        for query, result in zip(queries, results):
            print(f"Query: {query}")
            for entry in result:
                print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
            print("")

        # 8. Prompt for more queries
        queries = [input("Please enter a question: ")]
```

## Elasticsearch Integration

This example demonstrates how to set up Elasticsearch for sparse vector search by showing how to efficiently encode and index documents with sparse encoders, formulating search queries with sparse vectors, and providing an interactive query interface. See [semantic_search_elasticsearch.py](semantic_search_elasticsearch.py) or below:

### Prerequisites:

- Elasticsearch running locally (or accessible), see [Elasticsearch locally](https://www.elastic.co/guide/en/elasticsearch/reference/current/run-elasticsearch-locally.html) for more details.
- The Elasticsearch Python client must be installed:
  ```bash
  pip install elasticsearch
  ```

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. :meth:`SparseEncoder.encode_query <sentence_transformers.sparse_encoder.SparseEncoder.encode_query>`
   #. :meth:`SparseEncoder.encode_document <sentence_transformers.sparse_encoder.SparseEncoder.encode_document>`
   #. :meth:`semantic_search_elasticsearch <sentence_transformers.sparse_encoder.search_engines.semantic_search_elasticsearch>`
   #. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_
   #. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

::

    import time

    from datasets import load_dataset

    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_elasticsearch

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 4. Encode the corpus
    print("Start encoding corpus...")
    start_time = time.time()
    corpus_embeddings = sparse_model.encode_document(
        corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True
    )
    corpus_embeddings_decoded = sparse_model.decode(corpus_embeddings)
    print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")

    corpus_index = None
    while True:
        # 5. Encode the queries using the full precision
        start_time = time.time()
        query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
        query_embeddings_decoded = sparse_model.decode(query_embeddings)
        print(f"Encoding time: {time.time() - start_time:.6f} seconds")

        # 6. Perform semantic search using Elasticsearch
        results, search_time, corpus_index = semantic_search_elasticsearch(
            query_embeddings_decoded,
            corpus_embeddings_decoded=corpus_embeddings_decoded if corpus_index is None else None,
            corpus_index=corpus_index,
            top_k=5,
            output_index=True,
        )

        # 7. Output the results
        print(f"Search time: {search_time:.6f} seconds")
        for query, result in zip(queries, results):
            print(f"Query: {query}")
            for entry in result:
                print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
            print("")

        # 8. Prompt for more queries
        queries = [input("Please enter a question: ")]
```

## Seismic Integration

This example demonstrates how to use [Seismic](https://github.com/TusKANNy/seismic) for extremely performant sparse vector search. It does not require running a separate client, but instead performs search directly in memory. The Seismic library was introduced in [Bruch et al. (2024)](https://huggingface.co/papers/2404.18812), where it's shown to outperform the common inverted file (IVF) approach by an order of magnitude. For more information on building your Seismic Index you can look at the [Seismic Guidelines](https://github.com/TusKANNy/seismic/blob/main/docs/Guidelines.md). See [semantic_search_seismic.py](semantic_search_seismic.py) or below:

### Prerequisites:

- The Seismic Python package must be installed:
  ```bash
  pip install pyseismic-lsr
  ```

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. :meth:`SparseEncoder.encode_query <sentence_transformers.sparse_encoder.SparseEncoder.encode_query>`
   #. :meth:`SparseEncoder.encode_document <sentence_transformers.sparse_encoder.SparseEncoder.encode_document>`
   #. :meth:`semantic_search_seismic <sentence_transformers.sparse_encoder.search_engines.semantic_search_seismic>`
   #. `naver/splade-cocondenser-ensembledistil <https://huggingface.co/naver/splade-cocondenser-ensembledistil>`_
   #. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

::

    import time

    from datasets import load_dataset

    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.search_engines import semantic_search_seismic

    # 1. Load the natural-questions dataset with 100K answers
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    num_docs = 10_000
    corpus = dataset["answer"][:num_docs]

    # 2. Come up with some queries
    queries = dataset["query"][:2]

    # 3. Load the model
    sparse_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # 4. Encode the corpus
    print("Start encoding corpus...")
    start_time = time.time()
    corpus_embeddings = sparse_model.encode_document(
        corpus, convert_to_sparse_tensor=True, batch_size=16, show_progress_bar=True
    )
    corpus_embeddings_decoded = sparse_model.decode(corpus_embeddings)
    print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")

    corpus_index = None
    while True:
        # 5. Encode the queries using the full precision
        start_time = time.time()
        query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
        query_embeddings_decoded = sparse_model.decode(query_embeddings)
        print(f"Encoding time: {time.time() - start_time:.6f} seconds")

        # 6. Perform semantic search using Seismic
        results, search_time, corpus_index = semantic_search_seismic(
            query_embeddings_decoded,
            corpus_embeddings_decoded=corpus_embeddings_decoded if corpus_index is None else None,
            corpus_index=corpus_index,
            top_k=5,
            output_index=True,
        )

        # 7. Output the results
        print(f"Search time: {search_time:.6f} seconds")
        for query, result in zip(queries, results):
            print(f"Query: {query}")
            for entry in result:
                print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
            print("")

        # 8. Prompt for more queries
        queries = [input("Please enter a question: ")]
```

## SPLADE-index Integration

This example demonstrates how to use [splade-index](https://github.com/rasyosef/splade-index) for very fast sparse vector search powered by SciPy sparse matrices, built on top of the excellent [bm25s](https://github.com/xhluca/bm25s), a fast BM25 implementation. It does not require running a separate client, but instead performs search directly in memory. See [semantic_search_splade_index.py](semantic_search_splade_index.py) or below:

### Prerequisites:

- The SPLADE-index Python package must be installed:
  ```bash
  pip install splade-index
  ```

```{eval-rst}

.. sidebar:: Documentation

   #. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   #. `rasyosef/splade-tiny <https://huggingface.co/rasyosef/splade-tiny>`_
   #. `rasyosef/splade-mini <https://huggingface.co/rasyosef/splade-mini>`_
   #. `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

::
   
   import time
   
   from datasets import load_dataset
   from splade_index import SPLADE
   
   from sentence_transformers import SparseEncoder
   
   # 1. Load the natural-questions dataset with 100K answers
   dataset = load_dataset("sentence-transformers/natural-questions", split="train")
   num_docs = 10_000
   corpus = dataset["answer"][:num_docs]
   
   # 2. Come up with some queries
   queries = dataset["query"][:2]
   
   # 3. Load the model
   sparse_model = SparseEncoder("rasyosef/splade-tiny")
   
   # 4. Encode the corpus & create the index
   print("Start encoding corpus and creating index...")
   start_time = time.time()
   corpus_index = SPLADE()
   corpus_index.index(model=sparse_model, documents=corpus, batch_size=16, show_progress=True)
   print(f"Encoded corpus and created index in {time.time() - start_time:.6f} seconds")
   
   while True:
       # 5. Encode the queries using the full precision
       start_time = time.time()
       all_doc_ids, all_documents, all_scores = corpus_index.retrieve(queries, k=5)
       print(f"Encoding & Search time: {time.time() - start_time:.6f} seconds")
   
       # 7. Output the results
       for query, doc_ids, documents, scores in zip(queries, all_doc_ids, all_documents, all_scores):
           print(f"Query: {query}")
           for doc_id, document, score in zip(doc_ids, documents, scores):
               print(f"(Score: {score:.4f}) {document}, corpus_id: {doc_id}")
           print("")
   
       # 8. Prompt for more queries
       queries = [input("Please enter a question: ")]
```
