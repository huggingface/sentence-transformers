Computing Embeddings
====================

Once you have `installed <../../../../docs/installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.SentenceTransformer.similarity>`

::

   from sentence_transformers import SentenceTransformer

   # 1. Load a pretrained Sentence Transformer model
   model = SentenceTransformer("all-MiniLM-L6-v2")

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

.. note::
   Even though we talk about sentence embeddings, you can use Sentence Transformers for shorter phrases as well as for longer texts with multiple sentences. See :ref:`input-sequence-length` for notes on embeddings for longer texts.


Initializing a Sentence Transformer Model
-----------------------------------------

The first step is to load a pretrained Sentence Transformer model. You can use any of the models from the `Pretrained Models <../../../../docs/sentence_transformer/pretrained_models.html>`_ or a local model. See also :class:`~sentence_transformers.SentenceTransformer` for information on parameters.

::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("all-mpnet-base-v2")
   # Alternatively, you can pass a path to a local model directory:
   model = SentenceTransformer("output/models/mpnet-base-finetuned-all-nli")

The model will automatically be placed on the most performant available device, e.g. ``cuda`` or ``mps`` if available. You can also specify the device explicitly:

::

   model = SentenceTransformer("all-mpnet-base-v2", device="cuda")

Calculating Embeddings
----------------------

The method to calculate embeddings is :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`.


Prompt Templates
----------------

Some models require using specific text *prompts* to achieve optimal performance. For example, with `intfloat/multilingual-e5-large <https://huggingface.co/intfloat/multilingual-e5-large>`_ you should prefix all queries with ``"query: "`` and all passages with ``"passage: "``. Another example is `BAAI/bge-large-en-v1.5 <https://huggingface.co/BAAI/bge-large-en-v1.5>`_, which performs best for retrieval when the input texts are prefixed with ``"Represent this sentence for searching relevant passages: "``. 

Sentence Transformer models can be initialized with ``prompts`` and ``default_prompt_name`` parameters:

- ``prompts`` is an optional argument that accepts a dictionary of prompts with prompt names to prompt texts. The prompt will be prepended to the input text during inference. For example::

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        prompts={
            "classification": "Classify the following text: ",
            "retrieval": "Retrieve semantically similar text: ",
            "clustering": "Identify the topic or theme based on the text: ",
        },
    )
    # or
    model.prompts = {
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    }

- ``default_prompt_name`` is an optional argument that determines the default prompt to be used. It has to correspond with a prompt name from ``prompts``. If ``None``, then no prompt is used by default. For example::

    model = SentenceTransformer(
        "intfloat/multilingual-e5-large",
        prompts={
            "classification": "Classify the following text: ",
            "retrieval": "Retrieve semantically similar text: ",
            "clustering": "Identify the topic or theme based on the text: ",
        },
        default_prompt_name="retrieval",
    )
    # or
    model.default_prompt_name="retrieval"

Both of these parameters can also be specified in the ``config_sentence_transformers.json`` file of a saved model. That way, you won't have to specify these options manually when loading. When you save a Sentence Transformer model, these options will be automatically saved as well.

During inference, prompts can be applied in a few different ways. All of these scenarios result in identical texts being embedded:

1. Explicitly using the ``prompt`` option in ``SentenceTransformer.encode``::

    embeddings = model.encode("How to bake a strawberry cake", prompt="Retrieve semantically similar text: ")

2. Explicitly using the ``prompt_name`` option in ``SentenceTransformer.encode`` by relying on the prompts loaded from a) initialization or b) the model config::

    embeddings = model.encode("How to bake a strawberry cake", prompt_name="retrieval")

3. If ``prompt`` nor ``prompt_name`` are specified in ``SentenceTransformer.encode``, then the prompt specified by ``default_prompt_name`` will be applied. If it is ``None``, then no prompt will be applied::

    embeddings = model.encode("How to bake a strawberry cake")

.. _input-sequence-length:
Input Sequence Length
---------------------

For transformer models like BERT, RoBERTa, DistilBERT etc., the runtime and memory requirement grows quadratic with the input length. This limits transformers to inputs of certain lengths. A common value for BERT-based models are 512 tokens, which corresponds to about 300-400 words (for English).

Each model has a maximum sequence length under ``model.max_seq_length``, which is the maximal number of tokens that can be processed. Longer texts will be truncated to the first ``model.max_seq_length`` tokens::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 256

    # Change the length to 200
    model.max_seq_length = 200

    print("Max Sequence Length:", model.max_seq_length)
    # => Max Sequence Length: 200

.. note::

   You cannot increase the length higher than what is maximally supported by the respective transformer model. Also note that if a model was trained on short texts, the representations for long texts might not be that good.

Multi-Process / Multi-GPU Encoding
----------------------------------

You can encode input texts with more than one GPU (or with multiple processes on a CPU machine). It tends to help significantly with large datasets, but the overhead of starting multiple processes can be significant for smaller datasets.
For an example, see: `computing_embeddings_multi_gpu.py <https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/computing-embeddings/computing_embeddings_multi_gpu.py>`_.
 
You can use :meth:`SentenceTransformer.encode() <sentence_transformers.SentenceTransformer.encode>` (or :meth:`SentenceTransformer.encode_query() <sentence_transformers.SentenceTransformer.encode_query>` or :meth:`SentenceTransformer.encode_document() <sentence_transformers.SentenceTransformer.encode_document>`) with either:

- The ``device`` parameter, which can be set to e.g. ``"cuda:0"`` or ``"cpu"`` for single-process computations, but also a list of devices for multi-process or multi-gpu computations, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``::

        from sentence_transformers import SentenceTransformer

        def main():
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Encode with multiple GPUs
            embeddings = model.encode(
                inputs,
                device=["cuda:0", "cuda:1"]  # or ["cpu", "cpu", "cpu", "cpu"]
            )

        if __name__ == "__main__":
            main()

- The ``pool`` parameter can be provided, after calling :meth:`SentenceTransformer.start_multi_process_pool() <sentence_transformers.SentenceTransformer.start_multi_process_pool>` with a list of devices, e.g. ``["cuda:0", "cuda:1"]`` or ``["cpu", "cpu", "cpu", "cpu"]``. The benefit of this is that the pool can be reused for multiple calls to :meth:`SentenceTransformer.encode() <sentence_transformers.SentenceTransformer.encode>`, which is considerably more efficient than starting a new pool for each call::

        from sentence_transformers import SentenceTransformer

        def main():
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Start a multi-process pool with multiple GPUs
            pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
            # Encode with multiple GPUs
            embeddings = model.encode(inputs, pool=pool)
            # Don't forget to stop the pool after usage
            model.stop_multi_process_pool(pool)
        
        if __name__ == "__main__":
            main()

Additionally, you can use the ``chunk_size`` parameter to control the size of the chunks sent to each process. This differs from the ``batch_size`` parameter. For example, with a ``chunk_size=1000`` and a ``batch_size=32``, the input texts will be split into chunks of 1000 texts, and each chunk will be sent to a process and embedded in batches of 32 texts at a time. This can help with memory management and performance, especially for large datasets.
