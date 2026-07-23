Usage
=====

Characteristics of Multi-Vector Encoder (a.k.a. ColBERT or late-interaction) models:

1. Calculates a **sequence of token-level vectors** per input, rather than a single vector for the whole text.
2. Queries and documents are scored with the **MaxSim operator**: for each query token, take the maximum similarity to any document token, then sum across query tokens.
3. Preserves **token-level matching information** that single-vector models discard, typically yielding **stronger retrieval** at the cost of a **larger index footprint**.
4. State of the art for **visual document retrieval** (ColPali-style), where text queries match page images directly, skipping OCR entirely.

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Multi-Vector Encoder models:

.. sidebar:: Documentation

   1. :class:`MultiVectorEncoder <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder>`
   2. :meth:`MultiVectorEncoder.encode_query <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query>`
   3. :meth:`MultiVectorEncoder.encode_document <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document>`
   4. :meth:`MultiVectorEncoder.similarity <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity>`
   5. :meth:`MultiVectorEncoder.similarity_pairwise <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity_pairwise>`

::

   from sentence_transformers import MultiVectorEncoder

   # 1. Load a pretrained MultiVectorEncoder model
   model = MultiVectorEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")

   queries = ["What is the capital of France?"]
   documents = [
       "Paris is the capital of France.",
       "Berlin is the capital of Germany.",
   ]

   # 2. Encode queries and documents. Each embedding is a 2D array of
   # shape (num_tokens, embedding_dim), variable-length per input.
   query_embeddings = model.encode_query(queries)
   document_embeddings = model.encode_document(documents)

   # 3. Compute the MaxSim similarity matrix
   scores = model.similarity(query_embeddings, document_embeddings)
   print(scores)
   # tensor([[10.6578, 10.4499]])

Use :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query` and
:meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document` for retrieval. These set the
right prefix token (``[Q]`` / ``[D]``), max length, and apply any document-side skiplist configured on the model (empty by
default, though legacy ColBERT / PyLate checkpoints pre-seed it with punctuation tokens). When query expansion is enabled,
queries additionally pad to the fixed ``query_expansion["length"]``.

Multi-vector models can be loaded from any of the following sources, transparently::

    from sentence_transformers import MultiVectorEncoder

    # Checkpoints in the native Sentence Transformers format, e.g. models trained with this
    # library. PyLate builds on the same schema, so its checkpoints load identically:
    model = MultiVectorEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")
    model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
    model = MultiVectorEncoder("lightonai/Reason-ModernColBERT")

    # Stanford-NLP ColBERT format (auto-detected via the `HF_ColBERT` architecture marker;
    # the inline projection weight and special tokens are read from artifact.metadata)
    model = MultiVectorEncoder("colbert-ir/colbertv2.0")
    model = MultiVectorEncoder("answerdotai/answerai-colbert-small-v1")

    # transformers-native late-interaction retrievers (`*ForRetrieval` architectures, e.g.
    # ColPali / ColQwen2 / ColModernVBert): auto-detected; the projection and normalisation
    # live inside the model, and queries / image documents are formatted by the processor
    model = MultiVectorEncoder("vidore/colqwen2-v1.0-hf")

    # Bare transformer: a fresh random projection is appended; training required
    model = MultiVectorEncoder("answerdotai/ModernBERT-base")

For scoring, ``model.similarity`` returns the full all-pairs MaxSim score matrix, and ``model.similarity_pairwise``
returns matched-pair scores::

    scores = model.similarity(query_embeddings, document_embeddings)
    print(scores.shape)  # (1, 2): one query × two documents

    pairwise = model.similarity_pairwise([query_embeddings[0], query_embeddings[0]], document_embeddings)
    print(pairwise.shape)  # (2,)

You can also call the standalone scoring functions directly::

    from sentence_transformers.util import maxsim, maxsim_pairwise

    scores = maxsim(query_embeddings, document_embeddings)
    pairwise = maxsim_pairwise([query_embeddings[0], query_embeddings[0]], document_embeddings)

Late interaction is also the state of the art for visual document retrieval: text queries against page images, skipping OCR
entirely. Image documents encode the same way as text, passed as URLs, local paths, or PIL images::

    model = MultiVectorEncoder("vidore/colqwen2-v1.0-hf")

    queries = ["Total outlay is maximum in which year?"]
    images = [
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc1.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc2.jpg",
    ]

    query_embeddings = model.encode_query(queries)
    document_embeddings = model.encode_document(images)

    scores = model.similarity(query_embeddings, document_embeddings)

..
    TODO: Re-enable this section once PyLate supports sentence-transformers >= 6.0: it currently
    pins an older version, so the two libraries cannot be co-installed and this snippet cannot run.

    Retrieval at scale (indexing)
    -----------------------------

    Sentence Transformers does not ship a late-interaction index, but multi-vector indexes are
    model-independent: they store whatever ``encode_document`` produced. The two libraries co-exist,
    so you can encode with a Sentence Transformers ``MultiVectorEncoder`` and index / retrieve with
    `PyLate <https://github.com/lightonai/pylate>`_'s PLAID::

        from pylate import indexes, retrieve

        from sentence_transformers import MultiVectorEncoder

        model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

        document_ids = ["doc1", "doc2"]
        documents = [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
        ]

        index = indexes.PLAID(index_folder="indexes", index_name="my-corpus", override=True)
        index.add_documents(
            documents_ids=document_ids,
            documents_embeddings=model.encode_document(documents),
        )

        retriever = retrieve.ColBERT(index=index)
        queries = ["What is the capital of France?"]
        results = retriever.retrieve(queries_embeddings=model.encode_query(queries), k=10)

.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   custom_models
