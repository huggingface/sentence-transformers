Input Formats
=============

This page describes input formats for :class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`, :class:`~sentence_transformers.cross_encoder.model.CrossEncoder`, :class:`~sentence_transformers.sparse_encoder.model.SparseEncoder`, and :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models. Supported modalities depend on the loaded model.

Use :attr:`model.modalities <sentence_transformers.base.model.BaseModel.modalities>` to list supported modalities and :meth:`model.supports("image") <sentence_transformers.base.model.BaseModel.supports>` to check a particular modality. The input module and processor determine which representations of that modality are accepted.

For embedding models, ``encode()``, ``encode_query()``, and ``encode_document()`` accept the same input formats for the modalities supported by the model.

.. tip::

   Install the dependencies for the modalities you use, for example ``pip install -U "sentence-transformers[image]"``. See `Installation <installation.html>`_ for audio and video dependencies.

Input representations
---------------------

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Input
     - Accepted forms
   * - :ref:`Text <input-formats-text>`
     - Strings
   * - :ref:`Images <input-formats-images>`
     - PIL images, paths, URLs, arrays
   * - :ref:`Audio <input-formats-audio>`
     - Paths, URLs, arrays, metadata wrappers, decoders
   * - :ref:`Video <input-formats-video>`
     - Paths, URLs, frame arrays, metadata wrappers, decoders
   * - :ref:`Chat messages <input-formats-messages>`
     - Lists of message dictionaries
   * - :ref:`Combined modalities <input-formats-combined>`
     - Dictionaries

.. _input-formats-text:

Text
----

Pass text as strings:

.. code-block:: python

   embeddings = model.encode([
       "The weather is lovely today.",
       "It's so sunny outside!",
   ])

For retrieval, use ``encode_query()`` and ``encode_document()`` to apply the model's query and document settings:

.. code-block:: python

   query_embedding = model.encode_query("What is the weather like?")
   document_embedding = model.encode_document("It is sunny today.")

Multimodal models can recognize media paths and URLs by their extensions. To explicitly treat a string as text, wrap it in a ``"text"`` dict:

.. code-block:: python

   embedding = model.encode({"text": "photo.jpg"})

.. _input-formats-images:

Images
------

Pass local image paths as strings, or use image URLs:

.. code-block:: python

   document_embeddings = model.encode_document([
       "photo.jpg",
       "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
   ])

For images already in memory, use PIL images or NumPy/PyTorch arrays. Arrays can have shape ``(H, W, C)`` or ``(C, H, W)``. The processor handles resizing and normalization:

.. code-block:: python

   import numpy as np
   import torch
   from PIL import Image

   image = Image.open("photo.jpg").convert("RGB")
   query_embedding = model.encode_query(image)

   pixels = np.array(image)
   embedding = model.encode(pixels)

   pixels_tensor = torch.from_numpy(pixels).permute(2, 0, 1)
   embedding = model.encode(pixels_tensor)

A local file must exist for automatic path detection. If a URL lacks a recognizable extension, specify its modality explicitly with ``model.encode({"image": image_url})``.

.. _input-formats-audio:

Audio
-----

Pass local audio paths or URLs when the model's processor supports decoding them:

.. code-block:: python

   document_embeddings = model.encode_document([
       "speech.wav",
       "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
   ])

You can also decode audio yourself, which is required for processors that expect waveforms. Use the sampling rate expected by the processor. This example loads and resamples audio to 16 kHz:

.. code-block:: python

   from transformers.audio_utils import load_audio

   waveform = load_audio("speech.wav", sampling_rate=16000)
   audio = {"array": waveform, "sampling_rate": 16000}
   embedding = model.encode(audio)

A mono waveform has shape ``(num_samples,)``. Both NumPy arrays and PyTorch tensors can be passed with ``"array"`` and ``"sampling_rate"`` this way. Alternatively, pass the waveform directly and set the sampling rate through ``processing_kwargs``:

.. code-block:: python

   query_embedding = model.encode_query(
       waveform,
       processing_kwargs={"audio": {"sampling_rate": 16000}},
   )

In embedding calls, ``sampling_rate`` describes the supplied waveform and does not resample it. Audio wrappers in the same batch must have the same sampling rate.

You can also pass a :class:`torchcodec.AudioDecoder <torchcodec.decoders.AudioDecoder>`, optionally configured to decode at the required sampling rate:

.. code-block:: python

   from torchcodec.decoders import AudioDecoder

   audio = AudioDecoder("speech.wav", sample_rate=16000)
   embedding = model.encode(audio)

.. _input-formats-video:

Video
-----

Pass local video paths or URLs, such as this clip from `example-documents <https://huggingface.co/datasets/sentence-transformers/example-documents>`_. The processor handles decoding, sampling, and timing metadata:

.. code-block:: python

   embeddings = model.encode([
       "clip_a.mp4",
       "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/mapo_tofu.mp4",
   ])

For pre-sampled frames, attach metadata to each video and disable further sampling. ``frames_a`` and ``frames_b`` below contain three and two sampled frames, respectively. They can be NumPy/PyTorch arrays with shape ``(num_frames, C, H, W)`` or ``(num_frames, H, W, C)``. Alternatively, use a list or tuple of PIL images, NumPy frame arrays, or PyTorch frame tensors.

.. code-block:: python

   videos = [
       {
           "array": frames_a,
           "video_metadata": {
               "fps": 15, "total_num_frames": 15, "frames_indices": [0, 5, 10],
           },
       },
       {
           "array": frames_b,
           "video_metadata": {
               "fps": 10, "total_num_frames": 10, "frames_indices": [0, 5],
           },
       },
   ]
   embeddings = model.encode_document(
       videos, processing_kwargs={"video": {"do_sample_frames": False}},
   )

``fps`` and ``total_num_frames`` describe the original video, before sampling. ``frames_indices`` contains the original, zero-based indices of the supplied frames, in array order. Models such as Qwen3-VL use the indices and frame rate to determine timestamps.

When timing metadata is unavailable, set ``"video_metadata"`` to ``None`` to use processor defaults. These defaults cannot recover the original frame rate or sampled frame indices.

.. code-block:: python

   embeddings = model.encode_document(
       [{"array": frames_a, "video_metadata": None}, {"array": frames_b, "video_metadata": None}],
       processing_kwargs={"video": {"do_sample_frames": False}},
   )

Both examples above produce two embeddings, one per video. To combine both videos into one embedding, reuse the ``videos`` list from the first example, which already contains each video's frames and metadata, under the ``"video"`` key. This requires a model with a structured chat template that supports multiple videos:

.. code-block:: python

   embedding = model.encode(
       {"text": "Compare these clips", "video": videos},
       processing_kwargs={"video": {"do_sample_frames": False}},
   )

A list or tuple under ``"video"`` represents one video's frames when every item is recognized as an image, such as a PIL image, a 3D NumPy/PyTorch frame, or an image path/URL. Otherwise, it represents separate videos, each of which can have its own metadata wrapper. Use the ``{"array": frames, "video_metadata": metadata}`` wrapper to attach timing metadata or explicitly identify frames whose paths/URLs cannot be recognized as images.

You can also pass :class:`torchcodec.VideoDecoder <torchcodec.decoders.VideoDecoder>` objects:

.. code-block:: python

   from torchcodec.decoders import VideoDecoder

   video = VideoDecoder("clip_a.mp4")
   query_embedding = model.encode_query(video)

.. _input-formats-messages:

Chat messages
-------------

For embedding models that support the ``"message"`` modality, chat messages allow multiple content items and turns in one input. Most chat templates use one of two formats, differing in how ``"content"`` is represented.

**Flat content**, commonly used by text-only models, puts the text directly in ``"content"``:

.. code-block:: python

   messages = [
       {"role": "user", "content": "A bright, spacious room with large windows."},
   ]
   embedding = model.encode(messages)

**Structured content**, commonly used by multimodal models, puts a list of typed items in ``"content"``. This can, for example, combine text with multiple images:

.. code-block:: python

   messages = [
       {
           "role": "user",
           "content": [
               {"type": "text", "text": "Two views of the same room"},
               {"type": "image", "image": image_a},
               {"type": "image", "image": image_b},
           ],
       },
   ]
   embedding = model.encode(messages)

Use the format supported by the model's chat template. In both examples, ``messages`` is one conversation and produces one embedding.

For Cross Encoder models with chat templates, each query/document pair passed to ``predict()`` is converted into messages with ``"query"`` and ``"document"`` roles before applying the template. ``rank()`` constructs these pairs from its query and documents.

.. _input-formats-combined:

Combining modalities
--------------------

Use a dict with the keys ``"text"``, ``"image"``, ``"audio"``, or ``"video"`` to combine content into one input. Check that the model supports the combination using a tuple of the modalities:

.. code-block:: python

   print(model.supports(("image", "text")))
   # True

   embeddings = model.encode([
       {"text": "A cat on a sofa", "image": image_a},
       {"text": "A dog in a park", "image": image_b},
   ])

This produces two embeddings, each representing text and an image together. Passing the text and image as separate list entries instead produces separate embeddings, if the model supports mixed-modality batches:

.. code-block:: python

   embeddings = model.encode(["A cat on a sofa", image_a])

For a model with a structured chat template, a modality value can also be a list or tuple of items. They stay together in one input. For example, this produces one embedding from four descriptions and two images:

.. code-block:: python

   embedding = model.encode({
       "text": ["19 inch monitor", "Black finish", "HDMI input", "Adjustable stand"],
       "image": [image_a, image_b],
   })

Each text and image becomes a separate content item in the same message. Content follows each input's dict key order and the order within each collection, so this example places the four texts before the two images. The model and processor must support multiple media items in one input.

For video collections and frame metadata, see :ref:`input-formats-video`. Text-only inputs containing two strings retain the pair behavior described in :ref:`input-formats-batches`.

.. note::

   Mixed-modality batches require support for each input modality and for ``"message"``, which is used to process the different input types together. For a batch containing separate text and image inputs:

   .. code-block:: python

      can_mix = all(model.supports(modality) for modality in ("text", "image", "message"))

   Support for combining text and images within one input does not necessarily imply support for mixed-modality batches. See :meth:`model.supports() <sentence_transformers.base.model.BaseModel.supports>` for how modality combinations are checked.

Processing options
------------------

For models using the :class:`~sentence_transformers.base.modules.Transformer` input module, use ``processing_kwargs`` for per-call processing options, grouped by modality. This differs from ``processor_kwargs``, which configures processor initialization when loading the model:

.. code-block:: python

   embeddings = model.encode(
       ["A sentence", "Another sentence"],
       processing_kwargs={"text": {"max_length": 128, "truncation": True}},
   )

These options apply throughout the call. Pass each video's metadata alongside its frames as ``{"array": frames, "video_metadata": metadata}``, as in the video example, so the metadata stays with the correct video when inputs are sorted and batched. See :meth:`Transformer.preprocess <sentence_transformers.base.modules.Transformer.preprocess>` for the supported option groups.

.. _input-formats-batches:

Single inputs, batches, and pairs
---------------------------------

For embedding models, ``encode()``, ``encode_query()``, and ``encode_document()`` accept a single input or a list of inputs. For example, using images:

.. code-block:: python

   embedding = model.encode(image_a)
   embeddings = model.encode([image_a, image_b])

A text-only input containing a list or tuple of two strings is interpreted as a query/document pair. For example, ``model.encode({"text": ["query", "document"]})`` embeds one text pair, while ``model.encode(["query", "document"])`` embeds two separate texts.

For Cross Encoder models, ``predict()`` scores pairs and ``rank()`` scores a query against a list of documents. Here, ``reranker`` is a Cross Encoder that supports text and images:

.. code-block:: python

   scores = reranker.predict([("A cat", image_a), ("A dog", image_b)])
   rankings = reranker.rank("A cat", [image_a, image_b])

For CrossEncoder models with chat templates, the query message precedes the document message. Within each message, content follows the user's dict key order and the order within each collection.

Cross Encoder models can score audio or video inputs when the checkpoint supports those modalities. However, pairs do not currently support the ``{"array": ..., "sampling_rate": ...}`` or ``{"array": ..., "video_metadata": ...}`` forms, or TorchCodec decoder objects. This limitation applies to ``predict()``, ``rank()``, and Cross Encoder training. These forms are supported in embedding calls and dataset columns for embedding-model training.

In training datasets, each cell in an input column represents one model input. To obtain one embedding from multiple modalities, put a multimodal dict in that cell, such as ``{"text": caption, "image": image}``. Keep separate columns for the inputs required by your loss function, such as an anchor and a positive. See `Dataset Overview <sentence_transformer/dataset_overview.html>`_.
