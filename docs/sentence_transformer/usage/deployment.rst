Deployment, Cold Starts and Offline Loading
============================================

.. seealso::
   This page is about the time and failure modes *before the first request is served*: loading the
   model in a fresh container. For making inference itself faster once a model is loaded, see
   :doc:`Speeding up Inference <efficiency>`.

When a Sentence Transformer (or Cross Encoder) model runs on a scale-to-zero platform — for
example Cloud Run, a Lambda-style container, or Kubernetes pods scaled by KEDA — an instance
that has been idle can be stopped entirely. The next request then has to wait for a *cold start*
before it can be answered. This page explains what a cold start is made of, how to make model
loading deterministic in restricted or offline environments, and how to warm a model up before
traffic is routed to it.

The same guidance applies to the other model types (:class:`CrossEncoder <sentence_transformers.cross_encoder.model.CrossEncoder>`,
:class:`SparseEncoder <sentence_transformers.sparse_encoder.model.SparseEncoder>` and
:class:`MultiVectorEncoder <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder>`):
they accept the same ``local_files_only`` and ``cache_folder`` arguments when loading.

What a Cold Start Is Made Of
----------------------------

A cold start is not one cost, it is at least three, paid in sequence:

1. **Container / instance start** — the platform schedules and starts your container or pod,
   before any of your Python code runs.
2. **Model load** — importing Sentence Transformers and constructing the model, e.g.
   ``SentenceTransformer(...)``, which reads the weights (and tokenizer / processor files) into
   memory and moves them to the device.
3. **First inference** — the first ``encode()`` (or ``predict()`` / ``rank()``) call, which can be
   slower than later calls because of lazy initialisation in the underlying libraries.

Measure the segments separately instead of timing only the end-to-end request. An aggregate
number hides which lever to pull: if instance start dominates, a smaller or faster-loading model
will not fix your cold starts. A minimal way to split model load from first inference is:

.. code-block:: python

   import logging
   import time

   from sentence_transformers import SentenceTransformer

   logger = logging.getLogger(__name__)

   start = time.perf_counter()
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
   logger.info("Model load: %.2fs", time.perf_counter() - start)

   start = time.perf_counter()
   model.encode(["warmup"])
   logger.info("First inference: %.2fs", time.perf_counter() - start)

Container start has to be measured from the platform's own logs or metrics, as it happens before
this code runs.

Making Model Loading Deterministic and Offline
----------------------------------------------

By default, loading a model by name may contact the Hugging Face Hub at runtime to check for —
or download — files. In a restricted or offline environment that contact fails; in a normal
environment it adds a variable amount of time, which makes cold starts non-deterministic. The
remedy is to download the model *when you build your container image*, and to load only local
files at runtime:

1. **Pre-download the model at image build time**, into a cache folder inside the image:

   .. code-block:: dockerfile

      FROM python:3.12-slim

      RUN pip install -U sentence-transformers

      # Download the model weights while building, not while serving:
      ENV SENTENCE_TRANSFORMERS_HOME=/models
      RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

      # Fail fast at runtime instead of trying to reach the Hub:
      ENV HF_HUB_OFFLINE=1

      COPY app.py .
      CMD ["python", "app.py"]

   Alternatively, save a model to a folder with :meth:`model.save_pretrained() <sentence_transformers.sentence_transformer.model.SentenceTransformer.save_pretrained>`
   (or download it by any other means), copy that folder into the image, and load it from its
   local path.

2. **Load with ``local_files_only=True``** so a missing file is an immediate, clear error instead
   of a runtime download attempt:

   .. code-block:: python

      from sentence_transformers import SentenceTransformer

      # Only reads files that are already on disk (in the cache folder
      # or a local path); never downloads from the Hub.
      model = SentenceTransformer(
          "sentence-transformers/all-MiniLM-L6-v2",
          local_files_only=True,
      )

   ``cache_folder`` (or the ``SENTENCE_TRANSFORMERS_HOME`` environment variable) controls where
   models are cached; point it at the folder you populated at build time if that is not the
   default cache location.

3. **Or set ``HF_HUB_OFFLINE=1``** in the runtime environment. This tells ``huggingface_hub``
   (which Sentence Transformers uses to resolve model files) not to contact the Hub at all.
   ``local_files_only=True`` and ``HF_HUB_OFFLINE=1`` can be combined: the first is explicit in
   your code, the second also covers Hub lookups made by dependencies.

Frame this as *removing a variable network dependency* — it matters for reliability and for
restricted environments. As the measurement below shows, it should not be expected to make cold
starts meaningfully faster on its own.

Warmup and Readiness
--------------------

Loading the model is not quite the same as being ready to serve: the first inference call can
still pay lazy-initialisation costs. Run a dummy inference once at startup, as in the timing
example above:

.. code-block:: python

   model.encode(["warmup"])

Order startup so that warmup happens *before* your instance reports ready — for example, load
and warm up the model before starting the HTTP server, or keep the platform's readiness probe
failing until loading and warmup have finished. Otherwise the platform can route a real request
to an instance that is still loading, and that request pays the full cold-start cost (or times
out).

What Did Not Help: a Measurement
--------------------------------

Local-only loading alone does not necessarily improve end-to-end cold latency. In the deployment
that motivated this page (a FastAPI service with an embedding model and a cross-encoder reranker
on a scale-to-zero container; 5 paired cold measurements after 15 minutes idle, identical image
digest and resources), the results were:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - Baseline
     - Local-only loading
   * - Model load (median)
     - 24.24 s
     - 23.09 s (−4.7%)
   * - End-to-end cold start (median)
     - 37.68 s
     - 37.78 s (no improvement)

Splitting the cold start into segments showed why: model loading was not the dominant cost in
that setup — instance startup and transport were, at roughly 27 s of the total. Measuring the
segments first is what made that visible. See `the original report <https://github.com/huggingface/sentence-transformers/issues/3923>`_
for the full method.

Trade-offs
----------

- **Image size vs. cold start:** baking a model into the image makes the image larger, which can
  itself slow down container start. Whether that is a net win depends on the platform and the
  model size, which is another reason to measure the segments separately.
- **Minimum instances vs. latency and cost:** keeping at least one instance always running
  removes cold starts for the requests it serves, at the cost of paying for idle capacity.
  Scale-to-zero trades that cost for first-request latency.
- **Model choice:** a smaller model loads faster and makes a smaller image. See
  :doc:`Speeding up Inference <efficiency>` for backends and quantization options that also
  reduce model size.
