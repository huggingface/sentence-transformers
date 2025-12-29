Computing Sparse Embeddings
===========================

Once you have `installed <../../../../docs/installation.html>`_ Sentence Transformers, you can easily use Sparse Encoder models:

.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.model.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.model.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.model.SparseEncoder.similarity>`

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

.. note::
   Even though we talk about sentence embeddings, you can use Sparse Encoder for shorter phrases as well as for longer texts with multiple sentences. See :ref:`sparse-input-sequence-length` for notes on embeddings for longer texts.


Initializing a Sparse Encoder Model
-----------------------------------

The first step is to load a pretrained Sparse Encoder model. You can use any of the models from the `Pretrained Models <../../../../docs/sparse_encoder/pretrained_models.html>`_ or a local model. See also :class:`~sentence_transformers.sparse_encoder.model.SparseEncoder` for information on parameters.

::

   from sentence_transformers import SparseEncoder

   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
   # Alternatively, you can pass a path to a local model directory:
   model = SparseEncoder("output/models/sparse-distilbert-nq-finetuned")

The model will automatically be placed on the most performant available device, e.g. ``cuda`` or ``mps`` if available. You can also specify the device explicitly:

::

   model = SparseEncoder("naver/splade-cocondenser-ensembledistil", device="cuda")

Calculating Embeddings
----------------------

The method to calculate embeddings is :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.model.SparseEncoder.encode>`.

.. _sparse-input-sequence-length:

Input Sequence Length
---------------------

For transformer models like BERT, RoBERTa, DistilBERT etc., the runtime and memory requirement grows quadratic with the input length. This limits transformers to inputs of certain lengths. A common value for BERT-based models are 512 tokens, which corresponds to about 300-400 words (for English).

Each model has a maximum sequence length under ``model.max_seq_length``, which is the maximal number of tokens that can be processed. Longer texts will be truncated to the first ``model.max_seq_length`` tokens::

    from sentence_transformers import SparseEncoder

    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 256

    # Change the length to 200
    model.max_seq_length = 200

    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 200

.. note::

   You cannot increase the length higher than what is maximally supported by the respective transformer model. Also note that if a model was trained on short texts, the representations for long texts might not be that good.

Controlling Sparsity
--------------------

For sparse models, you can control the maximum number of active dimensions (non-zero values) in the output embeddings using the ``max_active_dims`` parameter. This is particularly useful for reducing memory usage and storage requirements and controlling the trade-off between accuracy and retrieval latency.

You can specify ``max_active_dims`` either when initializing the model or during encoding:

::

   from sentence_transformers import SparseEncoder

   # Initialize the SPLADE model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # Embed a list of sentences
   sentences = [
      "This framework generates embeddings for each input sentence",
      "Sentences are passed as a list of string.",
      "The quick brown fox jumps over the lazy dog.",
   ]

   # Generate embeddings
   embeddings = model.encode(sentences)

   # Print embedding dimensionality and sparsity
   print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")

   stats = model.sparsity(embeddings)
   print(f"Embedding sparsity: {stats}")
   print(f"Average non-zero dimensions: {stats['active_dims']:.2f}")
   print(f"Sparsity percentage: {stats['sparsity_ratio']:.2%}")
   """
   Embedding dim: 30522
   Embedding sparsity: {'active_dims': 56.333335876464844, 'sparsity_ratio': 0.9981543366792325}
   Average non-zero dimensions: 56.33
   Sparsity percentage: 99.82%
   """
   
   # Example of using max_active_dims during encoding to limit the active dimensions
   embeddings_limited = model.encode(sentences, max_active_dims=32)
   stats_limited = model.sparsity(embeddings_limited)
   print(f"Limited embedding sparsity: {stats_limited}")
   print(f"Average non-zero dimensions: {stats_limited['active_dims']:.2f}")
   print(f"Sparsity percentage: {stats_limited['sparsity_ratio']:.2%}")
   """
   Limited embedding sparsity: {'active_dims': 32.0, 'sparsity_ratio': 0.9989515759124565}
   Average non-zero dimensions: 32.00
   Sparsity percentage: 99.90%
   """

When you set ``max_active_dims``, the model will keep only the top-K dimensions with the highest values and set all other values to zero. This ensures your embeddings maintain a controlled level of sparsity while preserving the most important semantic information.

.. note::

   Setting a very low ``max_active_dims`` value may reduce the quality of search results. The optimal value depends on your specific use case and dataset.


One of the key benefits of controlling sparsity with ``max_active_dims`` is reduced memory usage. Here's an example showing the memory savings:

::

   def get_sparse_embedding_memory_size(tensor):
       # For sparse tensors, only count non-zero elements
       return (tensor._values().element_size() * tensor._values().nelement() + 
              tensor._indices().element_size() * tensor._indices().nelement())

   print(f"Original embeddings memory: {get_sparse_embedding_memory_size(embeddings) / 1024:.2f} KB")
   print(f"Embeddings with max_active_dims=32 memory: {get_sparse_embedding_memory_size(embeddings_limited) / 1024:.2f} KB")
   """
   Original embeddings memory: 3.32 KB
   Embeddings with max_active_dims=32 memory: 1.88 KB
   """

As shown in the example, limiting active dimensions to 32 reduced memory usage by approximately 43%. This efficiency becomes even more significant when working with large document collections but need to be put in balance with the possible loss of quality of the embeddings representations. Note that each of the `Evaluator classes <../../../../docs/package_reference/sparse_encoder/evaluation.html>`_ has a ``max_active_dims`` parameter that can be set to control the number of active dimensions during evaluation, so you can easily compare the performance of different settings.

Interpretability with SPLADE Models
-----------------------------------

When using SPLADE models, a key advantage is interpretability. You can easily visualize which tokens contribute most to the embedding, providing insights into what the model considers important in the text:

::

   from sentence_transformers import SparseEncoder

   # Initialize the SPLADE model
   model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

   # Embed a list of sentences
   sentences = [
      "This framework generates embeddings for each input sentence",
      "Sentences are passed as a list of string.",
      "The quick brown fox jumps over the lazy dog.",
   ]

   # Generate embeddings
   embeddings = model.encode(sentences)

   # Visualize top tokens for each text
   top_k = 10

   token_weights = model.decode(embeddings, top_k=top_k)

   print(f"\nTop tokens {top_k} for each text:")
   # The result is a list of sentence embeddings as numpy arrays
   for i, sentence in enumerate(sentences):
      token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in token_weights[i]])
      print(f"{i}: {sentence} -> Top tokens:  {token_scores}")

   """
   Top tokens 10 for each text:
      0: This framework generates embeddings for each input sentence -> Top tokens:  ("framework", 2.19), ("##bed", 2.12), ("input", 1.99), ("each", 1.60), ("em", 1.58), ("sentence", 1.49), ("generate", 1.42), ("##ding", 1.33), ("sentences", 1.10), ("create", 0.93)
      1: Sentences are passed as a list of string. -> Top tokens:  ("string", 2.72), ("pass", 2.24), ("sentences", 2.15), ("passed", 2.07), ("sentence", 1.90), ("strings", 1.86), ("list", 1.84), ("lists", 1.49), ("as", 1.18), ("passing", 0.73)
      2: The quick brown fox jumps over the lazy dog. -> Top tokens:  ("lazy", 2.18), ("fox", 1.67), ("brown", 1.56), ("over", 1.52), ("dog", 1.50), ("quick", 1.49), ("jump", 1.39), ("dogs", 1.25), ("foxes", 0.99), ("jumping", 0.84)
   """
   
This interpretability helps in understanding why certain documents match or don't match in search applications, and provides transparency into the model's behavior.

Multi-Process / Multi-GPU Encoding
----------------------------------

You can encode input texts with more than one GPU (or with multiple processes on a CPU machine). It tends to help significantly with large datasets, but the overhead of starting multiple processes can be significant for smaller datasets.
 
You can use :meth:`SparseEncoder.encode() <sentence_transformers.sparse_encoder.model.SparseEncoder.encode>` (or :meth:`SparseEncoder.encode_query() <sentence_transformers.sparse_encoder.model.SparseEncoder.encode_query>` or :meth:`SparseEncoder.encode_document() <sentence_transformers.sparse_encoder.model.SparseEncoder.encode_document>`) with either:

- The ``device`` parameter, which can be set to e.g. ``"cuda:0"`` or ``"cpu"`` for single-process computations, but also a list of devices for multi-process or multi-gpu computations, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``::

        from sentence_transformers import SparseEncoder

        def main():
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
            # Encode with multiple GPUs
            embeddings = model.encode(
                inputs,
                device=["cuda:0", "cuda:1"]  # or ["cpu", "cpu", "cpu", "cpu"]
            )

        if __name__ == "__main__":
            main()

- The ``pool`` parameter can be provided, after calling :meth:`SparseEncoder.start_multi_process_pool() <sentence_transformers.sparse_encoder.model.SparseEncoder.start_multi_process_pool>` with a list of devices, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``. The benefit of this is that the pool can be reused for multiple calls to :meth:`SparseEncoder.encode() <sentence_transformers.sparse_encoder.model.SparseEncoder.encode>`, which is considerably more efficient than starting a new pool for each call::

        from sentence_transformers import SparseEncoder

        def main():
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
            # Start a multi-process pool with multiple GPUs
            pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
            # Encode with multiple GPUs
            embeddings = model.encode(inputs, pool=pool)
            # Don't forget to stop the pool after usage
            model.stop_multi_process_pool(pool)
        
        if __name__ == "__main__":
            main()

Additionally, you can use the ``chunk_size`` parameter to control the size of the chunks sent to each process. This differs from the ``batch_size`` parameter. For example, with a ``chunk_size=1000`` and a ``batch_size=32``, the input texts will be split into chunks of 1000 texts, and each chunk will be sent to a process and embedded in batches of 32 texts at a time. This can help with memory management and performance, especially for large datasets.
