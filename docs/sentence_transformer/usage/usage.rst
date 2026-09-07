
Usage
=====

Characteristics of Sentence Transformer (a.k.a bi-encoder) models:

1. Calculates a **fixed-size vector representation (embedding)** given **texts, images, audio, video, or combinations thereof** (depending on the model).
2. Embedding calculation is often **efficient**, embedding similarity calculation is **very fast**.
3. Applicable for a **wide range of tasks**, such as semantic textual similarity, semantic search, clustering, classification, paraphrase mining, and more.
4. Often used as a **first step in a two-step retrieval process**, where a Cross-Encoder (a.k.a. reranker) model is used to re-rank the top-k results from the bi-encoder.

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Sentence Transformer models:

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.encode_query <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query>`
   4. :meth:`SentenceTransformer.encode_document <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document>`
   5. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

::

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

Some Sentence Transformer models support inputs beyond text, such as images, audio, or video. You can check which modalities a model supports using the :attr:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.modalities` property and the :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.supports` method. The :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode` method accepts different input formats depending on the modality:

.. tip::

   Multimodal models require additional dependencies. Install them with e.g. ``pip install -U "sentence-transformers[image]"`` for image support. See `Installation <../../installation.html>`_ for all options.

- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Audio**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"sampling_rate"`` keys, or (if ``torchcodec`` installed) :class:`torchcodec.AudioDecoder <torchcodec.decoders.AudioDecoder>` instances.
- **Video**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"video_metadata"`` keys, or (if ``torchcodec`` installed) :class:`torchcodec.VideoDecoder <torchcodec.decoders.VideoDecoder>` instances.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. ``{"text": ..., "audio": ...}``. The keys must be ``"text"``, ``"image"``, ``"audio"``, or ``"video"``.
- **Chat messages**: a list of dicts with ``"role"`` and ``"content"`` keys for multimodal models that use an uncommon chat template to combine text and non-text inputs.

The following example loads a multimodal model and computes similarities between text and image embeddings:

.. sidebar:: Modality Support

   .. code-block:: python

      from sentence_transformers import SentenceTransformer
   
      model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")
   
      # List all supported modalities
      print(model.modalities)
      # ['text', 'image', 'video', 'message']
   
      # Check for a specific modality
      print(model.supports("image"))
      # True
      print(model.supports("audio"))
      # False

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

For retrieval tasks, :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query` and :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document` are the recommended methods. Many embedding models use different prompts or instructions for queries vs. documents, and these methods handle that automatically:

- :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_query` uses the model's ``"query"`` prompt (if available) and sets ``task="query"``.
- :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode_document` uses the first available prompt from ``"document"``, ``"passage"``, or ``"corpus"``, and sets ``task="document"``.

These methods accept all the same input types as :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode` (text, images, URLs, multimodal dicts, etc.) and pass through all the same parameters. For models without specialized query/document prompts, they behave identically to :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode`.

.. code-block:: python

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

   # Encode text queries with the query prompt
   query_embeddings = model.encode_query([
       "Find me a photo of a vehicle parked near a building",
       "Show me an image of a pollinating insect",
   ])

   # Encode document screenshots with the document prompt
   doc_embeddings = model.encode_document([
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
   ])

   # Compute similarities
   similarities = model.similarity(query_embeddings, doc_embeddings)
   print(similarities)
   # tensor([[0.3907, 0.1490],
   #         [0.1235, 0.4872]])

Encoding pre-sampled videos
---------------------------

If you have already decoded and sampled a video, pass its frames and metadata together as ``{"array": frames, "video_metadata": metadata}``. This lets each video in a batch carry its own timing information. For example, Qwen3-VL uses the original frame indices and frame rate to determine timestamps for the sampled frames.

The example below assumes that you have extracted three frames from a 15-frame video recorded at 15 FPS, and two frames from a 10-frame video recorded at 10 FPS. The filenames identify the original, zero-based frame indices:

.. code-block:: python

   from PIL import Image
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")

   videos = [
       {
           "array": [Image.open(f"clip_a/frame_{index}.jpg") for index in [0, 5, 10]],
           "video_metadata": {
               "fps": 15,
               "total_num_frames": 15,
               "frames_indices": [0, 5, 10],
           },
       },
       {
           "array": [Image.open(f"clip_b/frame_{index}.jpg") for index in [0, 5]],
           "video_metadata": {
               "fps": 10,
               "total_num_frames": 10,
               "frames_indices": [0, 5],
           },
       },
   ]

   embeddings = model.encode_document(
       videos,
       processing_kwargs={"video": {"do_sample_frames": False}},
   )

``array`` can also be a pre-decoded NumPy array or PyTorch tensor with shape ``(num_sampled_frames, C, H, W)``. The metadata describes the source video:

- ``fps``: the original video's frame rate, before sampling.
- ``total_num_frames``: the total number of frames in the original video, not the number passed in ``array``.
- ``frames_indices``: the original, zero-based indices of the supplied frames, in the same order as ``array``.

Keep per-video metadata in each input dict. Options that apply to the entire call, such as ``do_sample_frames`` and resize settings, belong in ``processing_kwargs["video"]``. Setting ``do_sample_frames=False`` prevents the processor from sampling your frames again. To include text alongside a video, nest the same wrapper: ``{"text": "A description", "video": videos[0]}``; ``video_metadata`` is not a top-level modality key.

If you pass video file paths instead of pre-sampled frames, the model's video processor can decode and sample the videos and populate the metadata for you. See `Installation <../../installation.html>`_ for the video dependencies.

.. toctree::
   :maxdepth: 1
   :caption: Tasks and Advanced Usage

   ../../../examples/sentence_transformer/applications/computing-embeddings/README
   semantic_textual_similarity
   ../../../examples/sentence_transformer/applications/semantic-search/README
   ../../../examples/sentence_transformer/applications/retrieve_rerank/README
   ../../../examples/sentence_transformer/applications/clustering/README
   ../../../examples/sentence_transformer/applications/paraphrase-mining/README
   ../../../examples/sentence_transformer/applications/parallel-sentence-mining/README
   ../../../examples/sentence_transformer/applications/image-search/README
   ../../../examples/sentence_transformer/applications/embedding-quantization/README
   custom_models
   mteb_evaluation
   efficiency
