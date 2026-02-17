Usage
=====

Characteristics of Sparse Encoder models:

1. Calculates **sparse vector representations** where most dimensions are zero
2. Provides **efficiency benefits** for large-scale retrieval systems due to the sparse nature of embeddings
3. Often **more interpretable** than dense embeddings, with non-zero dimensions corresponding to specific tokens
4. **Complementary to dense embeddings**, enabling hybrid search systems that combine the strengths of both approaches

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Sparse Encoder models:

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.model.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.model.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.model.SparseEncoder.similarity>`
   4. :meth:`SparseEncoder.sparsity <sentence_transformers.sparse_encoder.model.SparseEncoder.sparsity>`

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


.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   ../../../examples/sparse_encoder/applications/computing_embeddings/README
   ../../../examples/sparse_encoder/applications/semantic_textual_similarity/README
   ../../../examples/sparse_encoder/applications/semantic_search/README
   ../../../examples/sparse_encoder/applications/retrieve_rerank/README
   ../../../examples/sparse_encoder/evaluation/README
   efficiency

