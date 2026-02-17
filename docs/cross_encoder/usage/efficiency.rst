
Speeding up Inference
=====================

Sentence Transformers supports 3 backends for performing inference with Cross Encoder models, each with its own optimizations for speeding up inference:


.. raw:: html

    <div class="components">
        <a href="#pytorch" class="box">
            <div class="header">PyTorch</div>
            The default backend for Cross Encoders.
        </a>
        <a href="#onnx" class="box">
            <div class="header">ONNX</div>
            Flexible and efficient model accelerator.
        </a>
        <a href="#openvino" class="box">
            <div class="header">OpenVINO</div>
            Optimization of models, mainly for Intel Hardware.
        </a>
        <a href="#benchmarks" class="box">
            <div class="header">Benchmarks</div>
            Benchmarks for the different backends.
        </a>
        <a href="#user-interface" class="box">
            <div class="header">User Interface</div>
            GUI to export, optimize, and quantize models.
        </a>
    </div>
    <br>

PyTorch
-------

The PyTorch backend is the default backend for Cross Encoders. If you don't specify a device, it will use the strongest available option across "cuda", "mps", and "cpu". Its default usage looks like this:

.. code-block:: python

   from sentence_transformers import CrossEncoder
   
   model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

   query = "Which planet is known as the Red Planet?"
   passages = [
      "Venus is often called Earth's twin because of its similar size and proximity.",
      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
   ]

   scores = model.predict([(query, passage) for passage in passages])
   print(scores)

If you're using a GPU, then you can use the following options to speed up your inference:

.. tab:: float16 (fp16)

   Float32 (fp32, full precision) is the default floating-point format in ``torch``, whereas float16 (fp16, half precision) is a reduced-precision floating-point format that can speed up inference on GPUs at a minimal loss of model accuracy. To use it, you can specify the ``torch_dtype`` during initialization or call :meth:`model.half() <torch.Tensor.half>` on the initialized model:

   .. code-block:: python

      from sentence_transformers import CrossEncoder
      
      model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float16"})
      # or: model.half()

      query = "Which planet is known as the Red Planet?"
      passages = [
         "Venus is often called Earth's twin because of its similar size and proximity.",
         "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
         "Jupiter, the largest planet in our solar system, has a prominent red spot.",
         "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
      ]

      scores = model.predict([(query, passage) for passage in passages])
      print(scores)

.. tab:: bfloat16 (bf16)

   Bfloat16 (bf16) is similar to fp16, but preserves more of the original accuracy of fp32. To use it, you can specify the ``torch_dtype`` during initialization or call :meth:`model.bfloat16() <torch.Tensor.bfloat16>` on the initialized model:

   .. code-block:: python

      from sentence_transformers import CrossEncoder
      
      model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", model_kwargs={"torch_dtype": "bfloat16"})
      # or: model.bfloat16()

      query = "Which planet is known as the Red Planet?"
      passages = [
         "Venus is often called Earth's twin because of its similar size and proximity.",
         "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
         "Jupiter, the largest planet in our solar system, has a prominent red spot.",
         "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
      ]

      scores = model.predict([(query, passage) for passage in passages])
      print(scores)

ONNX
----

.. include:: backend_export_sidebar.rst

ONNX can be used to speed up inference by converting the model to ONNX format and using ONNX Runtime to run the model. To use the ONNX backend, you must install Sentence Transformers with the ``onnx`` or ``onnx-gpu`` extra for CPU or GPU acceleration, respectively:

.. code-block:: bash

   pip install sentence-transformers[onnx-gpu]
   # or
   pip install sentence-transformers[onnx]

To convert a model to ONNX format, you can use the following code:

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="onnx")
   
   query = "Which planet is known as the Red Planet?"
   passages = [
      "Venus is often called Earth's twin because of its similar size and proximity.",
      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
   ]

   scores = model.predict([(query, passage) for passage in passages])
   print(scores)

If the model path or repository already contains a model in ONNX format, Sentence Transformers will automatically use it. Otherwise, it will convert the model to the ONNX format. 

.. note::

   If you wish to use the ONNX model outside of Sentence Transformers, you might need to apply your chosen activation function (e.g. Sigmoid) to get identical results as the Cross Encoder in Sentence Transformers.

All keyword arguments passed via ``model_kwargs`` will be passed on to :meth:`ORTModelForSequenceClassification.from_pretrained <optimum.onnxruntime.ORTModel.from_pretrained>`. Some notable arguments include:

* ``provider``: ONNX Runtime provider to use for loading the model, e.g. ``"CPUExecutionProvider"`` . See https://onnxruntime.ai/docs/execution-providers/ for possible providers. If not specified, the strongest provider (E.g. ``"CUDAExecutionProvider"``) will be used.
* ``file_name``: The name of the ONNX file to load. If not specified, will default to ``"model.onnx"`` or otherwise ``"onnx/model.onnx"``. This argument is useful for specifying optimized or quantized models.
* ``export``: A boolean flag specifying whether the model will be exported. If not provided, ``export`` will be set to ``True`` if the model repository or directory does not already contain an ONNX model.

.. tip::

   It's heavily recommended to save the exported model to prevent having to re-export it every time you run your code. You can do this by calling :meth:`model.save_pretrained() <sentence_transformers.cross_encoder.model.CrossEncoder.save_pretrained>` if your model was local:

   .. code-block:: python

      model = CrossEncoder("path/to/my/model", backend="onnx")
      model.save_pretrained("path/to/my/model")
   
   or with :meth:`model.push_to_hub() <sentence_transformers.cross_encoder.model.CrossEncoder.push_to_hub>` if your model was from the Hugging Face Hub:

   .. code-block:: python

      model = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", backend="onnx")
      model.push_to_hub("Alibaba-NLP/gte-reranker-modernbert-base", create_pr=True)

Optimizing ONNX Models
^^^^^^^^^^^^^^^^^^^^^^

.. include:: backend_export_sidebar.rst

ONNX models can be optimized using `Optimum <https://huggingface.co/docs/optimum/index>`_, allowing for speedups on CPUs and GPUs alike. To do this, you can use the :func:`~sentence_transformers.backend.export_optimized_onnx_model` function, which saves the optimized in a directory or model repository that you specify. It expects:

- ``model``: a Sentence Transformer, Sparse Encoder, or Cross Encoder model loaded with the ONNX backend.
- ``optimization_config``: ``"O1"``, ``"O2"``, ``"O3"``, or ``"O4"`` representing optimization levels from :class:`~optimum.onnxruntime.AutoOptimizationConfig`, or an :class:`~optimum.onnxruntime.OptimizationConfig` instance.
- ``model_name_or_path``: a path to save the optimized model file, or the repository name if you want to push it to the Hugging Face Hub.
- ``push_to_hub``: (Optional) a boolean to push the optimized model to the Hugging Face Hub.
- ``create_pr``: (Optional) a boolean to create a pull request when pushing to the Hugging Face Hub. Useful when you don't have write access to the repository.
- ``file_suffix``: (Optional) a string to append to the model name when saving it. If not specified, the optimization level name string will be used, or just ``"optimized"`` if the optimization config was not just a string optimization level.

See this example for exporting a model with :doc:`optimization level 3 <optimum-onnx:onnxruntime/usage_guides/optimization>` (basic and extended general optimizations, transformers-specific fusions, fast Gelu approximation):

.. tab:: Hugging Face Hub Model

   Only optimize once::

      from sentence_transformers import CrossEncoder, export_optimized_onnx_model

      model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="onnx")
      export_optimized_onnx_model(
          model=model,
          optimization_config="O3",
          model_name_or_path="cross-encoder/ms-marco-MiniLM-L6-v2",
          push_to_hub=True,
          create_pr=True,
      )

   Before the pull request gets merged::

      from sentence_transformers import CrossEncoder

      pull_request_nr = 2 # TODO: Update this to the number of your pull request
      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_O3.onnx"},
          revision=f"refs/pr/{pull_request_nr}"
      )
   
   Once the pull request gets merged::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_O3.onnx"},
      )

.. tab:: Local Model

   Only optimize once::

      from sentence_transformers import CrossEncoder, export_optimized_onnx_model

      model = CrossEncoder("path/to/my/mpnet-legal-finetuned", backend="onnx")
      export_optimized_onnx_model(
          model=model, optimization_config="O3", model_name_or_path="path/to/my/mpnet-legal-finetuned"
      )

   After optimizing::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "path/to/my/mpnet-legal-finetuned",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_O3.onnx"},
      )

Quantizing ONNX Models
^^^^^^^^^^^^^^^^^^^^^^

.. include:: backend_export_sidebar.rst

ONNX models can be quantized to int8 precision using `Optimum <https://huggingface.co/docs/optimum/index>`_, allowing for faster inference on CPUs. To do this, you can use the :func:`~sentence_transformers.backend.export_dynamic_quantized_onnx_model` function, which saves the quantized in a directory or model repository that you specify. Dynamic quantization, unlike static quantization, does not require a calibration dataset. It expects:

- ``model``: a Sentence Transformer, Sparse Encoder, or Cross Encoder model loaded with the ONNX backend.
- ``quantization_config``: ``"arm64"``, ``"avx2"``, ``"avx512"``, or ``"avx512_vnni"`` representing quantization configurations from :class:`~optimum.onnxruntime.AutoQuantizationConfig`, or an :class:`~optimum.onnxruntime.QuantizationConfig` instance.
- ``model_name_or_path``: a path to save the quantized model file, or the repository name if you want to push it to the Hugging Face Hub.
- ``push_to_hub``: (Optional) a boolean to push the quantized model to the Hugging Face Hub.
- ``create_pr``: (Optional) a boolean to create a pull request when pushing to the Hugging Face Hub. Useful when you don't have write access to the repository.
- ``file_suffix``: (Optional) a string to append to the model name when saving it. If not specified, ``"qint8_quantized"`` will be used.

On my CPU, each of the default quantization configurations (``"arm64"``, ``"avx2"``, ``"avx512"``, ``"avx512_vnni"``) resulted in roughly equivalent speedups.

See this example for quantizing a model to ``int8`` with :doc:`avx512_vnni <optimum-onnx:onnxruntime/usage_guides/quantization>`:

.. tab:: Hugging Face Hub Model

   Only quantize once::

      from sentence_transformers import CrossEncoder, export_dynamic_quantized_onnx_model

      model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="onnx")
      export_dynamic_quantized_onnx_model(
          model=model,
          quantization_config="avx512_vnni",
          model_name_or_path="sentence-transformers/cross-encoder/ms-marco-MiniLM-L6-v2",
          push_to_hub=True,
          create_pr=True,
      )

   Before the pull request gets merged::

      from sentence_transformers import CrossEncoder

      pull_request_nr = 2 # TODO: Update this to the number of your pull request
      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"},
          revision=f"refs/pr/{pull_request_nr}",
      )
   
   Once the pull request gets merged::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"},
      )

.. tab:: Local Model

   Only quantize once::

      from sentence_transformers import CrossEncoder, export_dynamic_quantized_onnx_model

      model = CrossEncoder("path/to/my/mpnet-legal-finetuned", backend="onnx")
      export_dynamic_quantized_onnx_model(
          model=model, quantization_config="avx512_vnni", model_name_or_path="path/to/my/mpnet-legal-finetuned"
      )

   After quantizing::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "path/to/my/mpnet-legal-finetuned",
          backend="onnx",
          model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"},
      )

OpenVINO
--------

.. include:: backend_export_sidebar.rst

OpenVINO allows for accelerated inference on CPUs by exporting the model to the OpenVINO format. To use the OpenVINO backend, you must install Sentence Transformers with the ``openvino`` extra:

.. code-block:: bash

   pip install sentence-transformers[openvino]

To convert a model to OpenVINO format, you can use the following code:

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="openvino")

   query = "Which planet is known as the Red Planet?"
   passages = [
      "Venus is often called Earth's twin because of its similar size and proximity.",
      "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
      "Jupiter, the largest planet in our solar system, has a prominent red spot.",
      "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
   ]

   scores = model.predict([(query, passage) for passage in passages])
   print(scores)

If the model path or repository already contains a model in OpenVINO format, Sentence Transformers will automatically use it. Otherwise, it will convert the model to the OpenVINO format.

.. note::

   If you wish to use the OpenVINO model outside of Sentence Transformers, you might need to apply your chosen activation function (e.g. Sigmoid) to get identical results as the Cross Encoder in Sentence Transformers.

.. raw:: html

   All keyword arguments passed via <code>model_kwargs</code> will be passed on to <a href="https://huggingface.co/docs/optimum/intel/openvino/reference#optimum.intel.openvino.modeling_base.OVBaseModel.from_pretrained"><code style="color: #404040; font-weight: 700;">OVBaseModel.from_pretrained()</code></a>. Some notable arguments include:

* ``file_name``: The name of the ONNX file to load. If not specified, will default to ``"openvino_model.xml"`` or otherwise ``"openvino/openvino_model.xml"``. This argument is useful for specifying optimized or quantized models.
* ``export``: A boolean flag specifying whether the model will be exported. If not provided, ``export`` will be set to ``True`` if the model repository or directory does not already contain an OpenVINO model.

.. tip::

   It's heavily recommended to save the exported model to prevent having to re-export it every time you run your code. You can do this by calling :meth:`model.save_pretrained() <sentence_transformers.cross_encoder.model.CrossEncoder.save_pretrained>` if your model was local:

   .. code-block:: python

      model = CrossEncoder("path/to/my/model", backend="openvino")
      model.save_pretrained("path/to/my/model")
   
   or with :meth:`model.push_to_hub() <sentence_transformers.cross_encoder.model.CrossEncoder.push_to_hub>` if your model was from the Hugging Face Hub:

   .. code-block:: python

      model = CrossEncoder("Alibaba-NLP/gte-reranker-modernbert-base", backend="openvino")
      model.push_to_hub("Alibaba-NLP/gte-reranker-modernbert-base", create_pr=True)

Quantizing OpenVINO Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: backend_export_sidebar.rst

OpenVINO models can be quantized to int8 precision using `Optimum Intel <https://huggingface.co/docs/optimum/main/en/intel/index>`_ to speed up inference.
To do this, you can use the :func:`~sentence_transformers.backend.export_static_quantized_openvino_model` function,
which saves the quantized model in a directory or model repository that you specify.
Post-Training Static Quantization expects:

- ``model``: a Sentence Transformer, Sparse Encoder, or Cross Encoder model loaded with the OpenVINO backend.
- ``quantization_config``: (Optional) The quantization configuration. This parameter accepts either:
  ``None`` for the default 8-bit quantization, a dictionary representing quantization configurations, or
  an :class:`~optimum.intel.OVQuantizationConfig` instance.
- ``model_name_or_path``: a path to save the quantized model file, or the repository name if you want to push it to the Hugging Face Hub.
- ``dataset_name``: (Optional) The name of the dataset to load for calibration. If not specified, defaults to ``sst2`` subset from the ``glue`` dataset.
- ``dataset_config_name``: (Optional) The specific configuration of the dataset to load.
- ``dataset_split``: (Optional) The split of the dataset to load (e.g., 'train', 'test').
- ``column_name``: (Optional) The column name in the dataset to use for calibration.
- ``push_to_hub``: (Optional) a boolean to push the quantized model to the Hugging Face Hub.
- ``create_pr``: (Optional) a boolean to create a pull request when pushing to the Hugging Face Hub. Useful when you don't have write access to the repository.
- ``file_suffix``: (Optional) a string to append to the model name when saving it. If not specified, ``"qint8_quantized"`` will be used.

See this example for quantizing a model to ``int8`` with `static quantization <https://huggingface.co/docs/optimum/main/en/intel/openvino/optimization#static-quantization>`_:

.. tab:: Hugging Face Hub Model

   Only quantize once::

      from sentence_transformers import CrossEncoder, export_static_quantized_openvino_model

      model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", backend="openvino")
      export_static_quantized_openvino_model(
          model=model,
          quantization_config=None,
          model_name_or_path="cross-encoder/ms-marco-MiniLM-L6-v2",
          push_to_hub=True,
          create_pr=True,
      )

   Before the pull request gets merged::

      from sentence_transformers import CrossEncoder

      pull_request_nr = 2 # TODO: Update this to the number of your pull request
      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="openvino",
          model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
          revision=f"refs/pr/{pull_request_nr}"
      )

   Once the pull request gets merged::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "cross-encoder/ms-marco-MiniLM-L6-v2",
          backend="openvino",
          model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
      )

.. tab:: Local Model

   Only quantize once::

      from sentence_transformers import CrossEncoder, export_static_quantized_openvino_model
      from optimum.intel import OVQuantizationConfig

      model = CrossEncoder("path/to/my/mpnet-legal-finetuned", backend="openvino")
      quantization_config = OVQuantizationConfig()
      export_static_quantized_openvino_model(
          model=model, quantization_config=quantization_config, model_name_or_path="path/to/my/mpnet-legal-finetuned"
      )

   After quantizing::

      from sentence_transformers import CrossEncoder

      model = CrossEncoder(
          "path/to/my/mpnet-legal-finetuned",
          backend="openvino",
          model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
      )

Benchmarks
----------

The following images show the benchmark results for the different backends on GPUs and CPUs. The results are averaged across 4 models of various sizes, 3 datasets, and numerous batch sizes.

.. raw:: html

   <details>
      <summary>Expand the benchmark details</summary>

   <br>
   Speedup ratio:
   <ul>
      <li>
         <b>Hardware: </b>RTX 3090 GPU, i7-17300K CPU
      </li>
      <li>
         <b>Datasets: </b> 2000 samples for GPU tests, 1000 samples for CPU tests.
         <ul>
            <li>
               <a href="https://huggingface.co/datasets/sentence-transformers/stsb">sentence-transformers/stsb</a>: <code>sentence1</code> and <code>sentence2</code> columns as pairs, with 38.94 ± 13.97 and 38.96 ± 14.05 characters on average, respectively.
            </li>
            <li>
               <a href="https://huggingface.co/datasets/sentence-transformers/natural-questions">sentence-transformers/natural-questions</a>: <code>query</code> and <code>answer</code> columns as pairs, with 46.99 ± 10.98 and 619.63 ± 345.30 characters on average, respectively.
            </li>
            <li>
               <a href="https://huggingface.co/datasets/stanfordnlp/imdb">stanfordnlp/imdb</a>: Two variants used from the <code>text</code> column: first 100 characters (100.00 ± 0.00 characters) and each sample repeated 4 times (16804.25 ± 10178.26 characters).
            </li>
         </ul>
      </li>
      <li>
         <b>Models: </b>
         <ul>
            <li>
               <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2">cross-encoder/ms-marco-MiniLM-L6-v2</a>: 22.7M parameters; batch sizes of 16, 32, 64, 128 and 256.
            </li>
            <li>
               <a href="https://huggingface.co/BAAI/bge-reranker-base">BAAI/bge-reranker-base</a>: 278M parameters; batch sizes of 16, 32, 64, and 128.
            </li>
            <li>
               <a href="https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1">mixedbread-ai/mxbai-rerank-large-v1</a>: 435M parameters; batch sizes of 8, 16, 32, and 64. Also 128 and 256 for GPU tests.
            </li>
            <li>
               <a href="https://huggingface.co/BAAI/bge-reranker-v2-m3">BAAI/bge-reranker-v2-m3</a>: 568M parameters; batch sizes of 2, 4. Also 8, 16, and 32 for GPU tests.
            </li>
         </ul>
      </li>
   </ul>
   Performance ratio: The same models and hardware was used. We compare the performance against the performance of PyTorch with fp32, i.e. the default backend and precision.
   <ul>
      <li>
         <b>Evaluation: </b>
         <ul>
            <li>
               <b>Information Retrieval: </b>NDCG@10 based on cosine similarity on the MS MARCO and NQ subsets from the <a href="https://huggingface.co/collections/zeta-alpha-ai/nanobeir-66e1a0af21dfd93e620cd9f6">NanoBEIR</a> collection of datasets, computed via the CrossEncoderNanoBEIREvaluator.
            </li>
         </ul>
      </li>
   </ul>

   <ul>
      <li>
         <b>Backends: </b>
         <ul>
            <li>
               <code>torch-fp32</code>: PyTorch with float32 precision (default).
            </li>
            <li>
               <code>torch-fp16</code>: PyTorch with float16 precision, via <code>model_kwargs={"torch_dtype": "float16"}</code>.
            </li>
            <li>
               <code>torch-bf16</code>: PyTorch with bfloat16 precision, via <code>model_kwargs={"torch_dtype": "bfloat16"}</code>.
            </li>
            <li>
               <code>onnx</code>: ONNX with float32 precision, via <code>backend="onnx"</code>.
            </li>
            <li>
               <code>onnx-O1</code>: ONNX with float32 precision and O1 optimization, via <code>export_optimized_onnx_model(..., optimization_config="O1", ...)</code> and <code>backend="onnx"</code>.
            </li>
            <li>
               <code>onnx-O2</code>: ONNX with float32 precision and O2 optimization, via <code>export_optimized_onnx_model(..., optimization_config="O2", ...)</code> and <code>backend="onnx"</code>.
            </li>
            <li>
               <code>onnx-O3</code>: ONNX with float32 precision and O3 optimization, via <code>export_optimized_onnx_model(..., optimization_config="O3", ...)</code> and <code>backend="onnx"</code>.
            </li>
            <li>
               <code>onnx-O4</code>: ONNX with float16 precision and O4 optimization, via <code>export_optimized_onnx_model(..., optimization_config="O4", ...)</code> and <code>backend="onnx"</code>.
            </li>
            <li>
               <code>onnx-qint8</code>: ONNX quantized to int8 with "avx512_vnni", via <code>export_dynamic_quantized_onnx_model(..., quantization_config="avx512_vnni", ...)</code> and <code>backend="onnx"</code>. The different quantization configurations resulted in roughly equivalent speedups.
            </li>
            <li>
               <code>openvino</code>: OpenVINO, via <code>backend="openvino"</code>.
            </li>
            <li>
               <code>openvino-qint8</code>: OpenVINO quantized to int8 via <code>export_static_quantized_openvino_model(..., quantization_config=OVQuantizationConfig(), ...)</code> and <code>backend="openvino"</code>.
            </li>
         </ul>
      </li>
   </ul>

   Note that the aggressive averaging across models, datasets, and batch sizes prevents some more intricate patterns from being visible. For example, ONNX seems to perform stronger at low batch sizes. However, ONNX and OpenVINO can even perform slightly worse than PyTorch, so we recommend testing the different backends with your specific model and data to find the best one for your use case.

   </details>
   <br>


.. image:: ../../img/ce_backends_benchmark_gpu.png
   :alt: Benchmark for GPUs
   :width: 45%

.. image:: ../../img/ce_backends_benchmark_cpu.png
   :alt: Benchmark for CPUs
   :width: 45%

Recommendations
^^^^^^^^^^^^^^^

Based on the benchmarks, this flowchart should help you decide which backend to use for your model:

.. mermaid::
   
   %%{init: {
      "theme": "neutral",
      "flowchart": {
         "curve": "bumpY"
      }
   }}%%
   graph TD
   A("What is your hardware?") -->|GPU| B("Are you using a small<br>batch size?")
   A -->|CPU| C("Are minor performance<br>degradations acceptable?")
   B -->|yes| D[onnx-O4]
   B -->|no| F[float16]
   C -->|yes| G[openvino-qint8]
   C -->|no| H("Do you have an Intel CPU?")
   H -->|yes| I[openvino]
   H -->|no| J[onnx]
   click D "#optimizing-onnx-models"
   click F "#pytorch"
   click G "#quantizing-openvino-models"
   click I "#openvino"
   click J "#onnx"

.. note::

   Your milage may vary, and you should always test the different backends with your specific model and data to find the best one for your use case.

User Interface
^^^^^^^^^^^^^^

This Hugging Face Space provides a user interface for exporting, optimizing, and quantizing models for either ONNX or OpenVINO:

- `sentence-transformers/backend-export <https://huggingface.co/spaces/sentence-transformers/backend-export>`_
