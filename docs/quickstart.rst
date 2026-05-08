Quickstart
==========

Sentence Transformer
--------------------

Characteristics of Sentence Transformer (a.k.a bi-encoder) models:

1. Calculates a **fixed-size vector representation (embedding)** given **texts, images, audio, or video**.
2. Embedding calculation is often **efficient**, embedding similarity calculation is **very fast**.
3. Applicable for a **wide range of tasks**, such as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.
4. Often used as a **first step in a two-step retrieval process**, where a Cross-Encoder (a.k.a. reranker) model is used to re-rank the top-k results from the bi-encoder.

Once you have `installed <installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>`
   4. :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>`
   5. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

   **Other useful methods and links:**

   - :meth:`SentenceTransformer.similarity_pairwise <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity_pairwise>`
   - `SentenceTransformer > Usage <./sentence_transformer/usage/usage.html>`_
   - `SentenceTransformer > Usage > Speeding up Inference <./sentence_transformer/usage/efficiency.html>`_
   - `SentenceTransformer > Pretrained Models <./sentence_transformer/pretrained_models.html>`_
   - `SentenceTransformer > Training Overview <./sentence_transformer/training_overview.html>`_
   - `SentenceTransformer > Dataset Overview <./sentence_transformer/dataset_overview.html>`_
   - `SentenceTransformer > Loss Overview <./sentence_transformer/loss_overview.html>`_
   - `SentenceTransformer > Training Examples <./sentence_transformer/training/examples.html>`_

.. tab:: Text

   .. code-block:: python

      from sentence_transformers import SentenceTransformer

      # 1. Load a pretrained Sentence Transformer model
      model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

      # The sentences to encode
      sentences = [
          "The weather is lovely today.",
          "It's so sunny outside!",
          "He drove to the stadium.",
      ]

      # 2. Calculate embeddings by calling model.encode()
      embeddings = model.encode(sentences)
      print(embeddings.shape)
      # [3, 384]

      # 3. Calculate the embedding similarities
      similarities = model.similarity(embeddings, embeddings)
      print(similarities)
      # tensor([[1.0000, 0.6660, 0.1046],
      #         [0.6660, 1.0000, 0.1411],
      #         [0.1046, 0.1411, 1.0000]])

.. tab:: Multimodal

   .. tip::

      Multimodal models require additional dependencies. Install them with e.g. ``pip install -U "sentence-transformers[image]"`` for image support. See `Installation <installation.html>`_ for all options.

   .. code-block:: python

      from sentence_transformers import SentenceTransformer

      # 1. Load a model that supports both text and images
      model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

      # 2. Encode images from URLs
      img_embeddings = model.encode([
          "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
          "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
      ])

      # 3. Encode text queries (one matching + one hard negative per image)
      text_embeddings = model.encode([
          "A green car parked in front of a yellow building",
          "A red car driving on a highway",
          "A bee on a pink flower",
          "A wasp on a wooden table",
      ])

      # 4. Compute cross-modal similarities
      similarities = model.similarity(text_embeddings, img_embeddings)
      print(similarities)
      # tensor([[0.5115, 0.1078],
      #         [0.1999, 0.1108],
      #         [0.1255, 0.6749],
      #         [0.1283, 0.2704]])

With ``SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")`` we pick which `Sentence Transformer model <https://huggingface.co/models?library=sentence-transformers>`_ we load. In this example, we load `sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_, which is a MiniLM model finetuned on a large dataset of over 1 billion training pairs. Using :meth:`SentenceTransformer.similarity() <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`, we compute the similarity between all pairs of sentences. As expected, the similarity between semantically related inputs is higher than between unrelated ones. Multimodal models like `Qwen/Qwen3-VL-Embedding-2B <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>`_ can also encode images, audio, or video into the same embedding space.

Finetuning Sentence Transformer models is easy and requires only a few lines of code. For more information, see the `Training Overview <./sentence_transformer/training_overview.html>`__ section.

.. tip::

    Read `Sentence Transformer > Usage > Speeding up Inference <sentence_transformer/usage/efficiency.html>`_ for tips on how to speed up inference of models by up to 2x-3x.

Cross Encoder
-------------

Characteristics of Cross Encoder (a.k.a reranker) models:

1. Calculates a **similarity score** given **pairs of inputs** (typically text, but also images or other modalities).
2. Generally provides **superior performance** compared to a Sentence Transformer (a.k.a. bi-encoder) model.
3. Often **slower** than a Sentence Transformer model, as it requires computation for each pair rather than each text.
4. Due to the previous 2 characteristics, Cross Encoders are often used to **re-rank the top-k results** from a Sentence Transformer model.

The usage for Cross Encoder (a.k.a. reranker) models is similar to Sentence Transformers:

.. sidebar:: Documentation

   1. :class:`CrossEncoder <sentence_transformers.CrossEncoder>`
   2. :meth:`CrossEncoder.rank <sentence_transformers.CrossEncoder.rank>`
   3. :meth:`CrossEncoder.predict <sentence_transformers.CrossEncoder.predict>`

   **Other useful methods and links:**

   - `CrossEncoder > Usage <./cross_encoder/usage/usage.html>`_
   - `CrossEncoder > Pretrained Models <./cross_encoder/pretrained_models.html>`_
   - `CrossEncoder > Training Overview <./cross_encoder/training_overview.html>`_
   - `CrossEncoder > Dataset Overview <./cross_encoder/dataset_overview.html>`_
   - `CrossEncoder > Loss Overview <./cross_encoder/loss_overview.html>`_
   - `CrossEncoder > Training Examples <./cross_encoder/training/examples.html>`_

.. tab:: Text

   .. code-block:: python

      from sentence_transformers import CrossEncoder

      # 1. Load a pretrained CrossEncoder model
      model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

      # We want to compute the similarity between the query sentence...
      query = "A man is eating pasta."

      # ... and all sentences in the corpus
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

      # 2. We rank all sentences in the corpus for the query
      ranks = model.rank(query, corpus)

      # Print the scores
      print("Query: ", query)
      for rank in ranks:
          print(f"{rank['score']:.2f}\t{corpus[rank['corpus_id']]}")
      """
      Query:  A man is eating pasta.
      0.67    A man is eating food.
      0.34    A man is eating a piece of bread.
      0.08    A man is riding a horse.
      0.07    A man is riding a white horse on an enclosed ground.
      0.01    The girl is carrying a baby.
      0.01    Two men pushed carts through the woods.
      0.01    A monkey is playing drums.
      0.01    A woman is playing violin.
      0.01    A cheetah is running behind its prey.
      """

      # 3. Alternatively, you can also manually compute the score between two sentences
      import numpy as np

      sentence_combinations = [[query, sentence] for sentence in corpus]
      scores = model.predict(sentence_combinations)

      # Sort the scores in decreasing order to get the corpus indices
      ranked_indices = np.argsort(scores)[::-1]
      print("Scores:", scores)
      print("Indices:", ranked_indices)
      """
      Scores: [0.6732372, 0.34102544, 0.00542465, 0.07569341, 0.00525378, 0.00536814, 0.06676237, 0.00534825, 0.00516717]
      Indices: [0 1 3 6 2 5 7 4 8]
      """

.. tab:: Multimodal

   .. code-block:: python

      from sentence_transformers import CrossEncoder

      model = CrossEncoder("Qwen/Qwen3-VL-Reranker-2B")

      query = "A green car parked in front of a yellow building"
      documents = [
          # Image documents (URL or local file path)
          "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
          "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
          # Text document
          "A vintage Volkswagen Beetle painted in bright green sits in a driveway.",
          # Combined text + image document
          {
              "text": "A car in a European city",
              "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
          },
      ]

      rankings = model.rank(query, documents)
      for rank in rankings:
          print(f"{rank['score']:.4f}\t(document {rank['corpus_id']})")
      """
      0.9375  (document 0)
      0.5000  (document 3)
      -1.2500 (document 2)
      -2.4375 (document 1)
      """

With ``CrossEncoder("cross-encoder/stsb-distilroberta-base")`` we pick which `CrossEncoder model <./cross_encoder/pretrained_models.html>`_ we load. CrossEncoder models can also work with multimodal inputs: `Qwen/Qwen3-VL-Reranker-2B <https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B>`_ can rank images and text by relevance to a query.

Finetuning CrossEncoder models is easy and requires only a few lines of code. For more information, see the `Training Overview <./cross_encoder/training_overview.html>`__ section.

.. tip::

    Read `CrossEncoder > Usage > Speeding up Inference <cross_encoder/usage/efficiency.html>`_ for tips on how to speed up inference of models by up to 2x-3x.

Sparse Encoder
--------------

Characteristics of Sparse Encoder models:

1. Calculates **sparse vector representations** where most dimensions are zero.
2. Provides **efficiency benefits** for large-scale retrieval systems due to the sparse nature of embeddings.
3. Often **more interpretable** than dense embeddings, with non-zero dimensions corresponding to specific tokens.
4. **Complementary to dense embeddings**, enabling hybrid search systems that combine the strengths of both approaches.

The usage for Sparse Encoder models follows a similar pattern to Sentence Transformers:

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.SparseEncoder.similarity>`
   4. :meth:`SparseEncoder.sparsity <sentence_transformers.SparseEncoder.sparsity>`

   **Other useful methods and links:**

   - `SparseEncoder > Usage <./sparse_encoder/usage/usage.html>`_
   - `SparseEncoder > Pretrained Models <./sparse_encoder/pretrained_models.html>`_
   - `SparseEncoder > Training Overview <./sparse_encoder/training_overview.html>`_
   - `SparseEncoder > Loss Overview <./sparse_encoder/loss_overview.html>`_
   - `Sparse Encoder > Vector Database Integration <../examples/sparse_encoder/applications/semantic_search/README.html#vector-database-search>`_

::

   from sentence_transformers import SparseEncoder

   # 1. Load a pretrained SparseEncoder model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # The sentences to encode
   sentences = [
       "The weather is lovely today.",
       "It's so sunny outside!",
       "He drove to the stadium.",
   ]

   # 2. Calculate sparse embeddings by calling model.encode()
   embeddings = model.encode(sentences)
   print(embeddings.shape)
   # [3, 30522] - sparse representation with vocabulary size dimensions

   # 3. Calculate the embedding similarities (using dot product by default)
   similarities = model.similarity(embeddings, embeddings)
   print(similarities)
   # tensor([[   35.629,     9.154,     0.098],
   #         [    9.154,    27.478,     0.019],
   #         [    0.098,     0.019,    29.553]])

   # 4. Check sparsity statistics
   stats = SparseEncoder.sparsity(embeddings)
   print(f"Sparsity: {stats['sparsity_ratio']:.2%}")  # Typically >99% zeros
   print(f"Avg non-zero dimensions per embedding: {stats['active_dims']:.2f}")

With ``SparseEncoder("naver/splade-cocondenser-ensembledistil")`` we load a pretrained SPLADE model that generates sparse embeddings. SPLADE (SParse Lexical AnD Expansion) models use MLM prediction mechanisms to create sparse representations that are particularly effective for information retrieval tasks.

Finetuning Sparse Encoder models is easy and requires only a few lines of code. For more information, see the `Training Overview <./sparse_encoder/training_overview.html>`__ section.

.. tip::

    Read `Sparse Encoder > Usage > Speeding up Inference <sparse_encoder/usage/efficiency.html>`_ for tips on how to speed up inference of models by up to 2x-3x.

Next Steps
----------

Consider reading one of the following sections next:

* `Sentence Transformers > Usage <./sentence_transformer/usage/usage.html>`_
* `Sentence Transformers > Pretrained Models <./sentence_transformer/pretrained_models.html>`_
* `Sentence Transformers > Training Overview <./sentence_transformer/training_overview.html>`_
* `Sentence Transformers > Training Examples > Multilingual Models <../examples/sentence_transformer/training/multilingual/README.html>`_
* `Sentence Transformers > Training Examples > Multimodal Models <../examples/sentence_transformer/training/multimodal/README.html>`_
* `Cross Encoder > Usage <./cross_encoder/usage/usage.html>`_
* `Cross Encoder > Pretrained Models <./cross_encoder/pretrained_models.html>`_
* `Cross Encoder > Training Examples > Multimodal Models <../examples/cross_encoder/training/multimodal/README.html>`_
* `Sparse Encoder > Usage <./sparse_encoder/usage/usage.html>`_
* `Sparse Encoder > Pretrained Models <./sparse_encoder/pretrained_models.html>`_
* `Sparse Encoder > Vector Database Integration <../examples/sparse_encoder/applications/semantic_search/README.html#vector-database-search>`_

