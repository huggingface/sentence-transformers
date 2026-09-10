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
   2. :meth:`MultiVectorEncoder.encode <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode>`
   3. :meth:`MultiVectorEncoder.encode_query <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query>`
   4. :meth:`MultiVectorEncoder.encode_document <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document>`
   5. :meth:`MultiVectorEncoder.similarity <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity>`
   6. :meth:`MultiVectorEncoder.similarity_pairwise <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.similarity_pairwise>`

::

   from sentence_transformers import MultiVectorEncoder

   # 1. Load a pretrained MultiVectorEncoder model
   model = MultiVectorEncoder("lightonai/LateOn")

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
   # tensor([[9.1129, 8.8769]])

Some Multi-Vector Encoder models support inputs beyond text, most notably page images for visual document retrieval. You can check which modalities a model supports using the :attr:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.modalities` property and the :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.supports` method. The :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode` method accepts different input formats depending on the modality:

.. tip::

   Multimodal models require additional dependencies. Install them with e.g. ``pip install -U "sentence-transformers[image]"`` for image support. See `Installation <../../installation.html>`_ for all options.

- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. ``{"text": ..., "image": ...}``. The keys must be ``"text"``, ``"image"``, ``"audio"``, or ``"video"``, although released late-interaction checkpoints only accept text and images.
- **Chat messages**: a list of dicts with ``"role"`` and ``"content"`` keys for multimodal models that use an uncommon chat template to combine text and non-text inputs.

The following example loads a ColQwen2.5 model and scores text queries against page images directly, skipping OCR entirely:

.. sidebar:: Modality Support

   .. code-block:: python

      from sentence_transformers import MultiVectorEncoder

      model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

      # List all supported modalities
      print(model.modalities)
      # ['text', 'image']

      # Check for a specific modality
      print(model.supports("image"))
      # True
      print(model.supports("audio"))
      # False

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder

   # 1. Load a model that supports both text and images
   model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")

   queries = [
       "What is the variable represented on the y-axis of the graph?",
       "Total outlay is maximum in which year?",
   ]
   # 2. Image documents are passed as URLs, local paths, or PIL images
   images = [
       "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
       "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
       "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
       "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
   ]

   # 3. Image documents encode the same way as text ones
   query_embeddings = model.encode_query(queries)
   document_embeddings = model.encode_document(images)

   # 4. Compute cross-modal MaxSim scores
   scores = model.similarity(query_embeddings, document_embeddings)
   print(scores)
   # tensor([[13.8672, 12.3115, 12.1670, 11.0293],
   #         [ 7.2012, 14.7207,  6.9414,  6.9746]])

Use :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_query` and :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.encode_document` for retrieval. These set the right prefix token (``[Q]`` / ``[D]``), max length, and apply any document-side skiplist configured on the model (empty by default, though legacy ColBERT / PyLate checkpoints pre-seed it with punctuation tokens). When query expansion is enabled, queries additionally expand to the configured length.

The prefix tokens are stored as the model's ``"query"`` and ``"document"`` prompts, so you can inspect exactly what each method prepends:

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder

   model = MultiVectorEncoder("lightonai/mLateOn")

   print(model.prompts)
   # {'query': '[Q] ', 'document': '[D] '}

   query_embeddings = model.encode_query(["What is the capital of France?"])
   document_embeddings = model.encode_document([
       "Paris is the capital of France.",
       "Berlin is the capital of Germany.",
   ])

   scores = model.similarity(query_embeddings, document_embeddings)
   print(scores)
   # tensor([[9.6578, 9.5248]])

The prefixes differ per checkpoint. Models in the original ColBERT format reuse reserved vocabulary entries instead, e.g. ``answerdotai/answerai-colbert-small-v1`` has ``{'query': '[unused0] ', 'document': '[unused1] '}``.

For scoring, ``model.similarity`` returns the full all-pairs MaxSim score matrix, and ``model.similarity_pairwise`` returns matched-pair scores. Both follow the model's ``similarity_fn_name``, which is either ``"maxsim"`` (the default) or ``"meanmaxsim"`` (MaxSim divided by the query's token count):

.. code-block:: python

   scores = model.similarity(query_embeddings, document_embeddings)
   print(scores.shape)  # torch.Size([1, 2]), one query against two documents

   pairwise = model.similarity_pairwise([query_embeddings[0], query_embeddings[0]], document_embeddings)
   print(pairwise.shape)  # torch.Size([2])

Multi-vector models can be loaded from any of the following sources, transparently:

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder

   # Checkpoints in the native Sentence Transformers format, e.g. models trained with this
   # library. PyLate builds on the same schema, so its checkpoints load identically:
   model = MultiVectorEncoder("lightonai/LateOn")
   model = MultiVectorEncoder("lightonai/mLateOn")
   model = MultiVectorEncoder("LiquidAI/LFM2-ColBERT-350M")

   # Some native checkpoints ship custom architecture code and need trust_remote_code
   model = MultiVectorEncoder("perplexity-ai/pplx-embed-v1-late-0.6b", trust_remote_code=True)

   # Stanford-NLP ColBERT format, auto-detected via the `HF_ColBERT` architecture marker.
   # The inline projection weight and special tokens are read from artifact.metadata
   model = MultiVectorEncoder("colbert-ir/colbertv2.0")
   model = MultiVectorEncoder("answerdotai/answerai-colbert-small-v1")

   # transformers-native late-interaction retrievers (`*ForRetrieval` architectures, e.g.
   # ColPali / ColQwen2 / ColModernVBert) are auto-detected. The projection and normalization
   # live inside the model, and queries / image documents are formatted by the processor
   model = MultiVectorEncoder("vidore/colqwen2-v1.0-hf")

   # Bare transformer: a fresh random projection is appended, so training is required
   model = MultiVectorEncoder("answerdotai/ModernBERT-base")

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

        model = MultiVectorEncoder("lightonai/LateOn")

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

   ../../../examples/multi_vector_encoder/applications/README
   ../../../examples/multi_vector_encoder/evaluation/README
   custom_models
   efficiency
