Kernel acceleration
===================

Kernels are optimized implementations of operations your model already uses.
For example, a MaxSim kernel compares query and document token embeddings without
building the full token similarity matrix in memory. This can make scoring faster
and leave more GPU memory available for larger batches.

Sentence Transformers can load kernels from the Hugging Face Hub and use them in
place of the corresponding operations. You keep using the same encoding, scoring,
and training APIs. Start with ``use_kernels=True`` to let the library choose its
default kernels. You only need ``kernel_config`` if you want to choose providers
yourself.

Enabling kernels
----------------

Install the optional dependencies:

.. code-block:: bash

   pip install "sentence-transformers[kernels]"

Then enable kernels when loading your model:

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder

   model = MultiVectorEncoder(
       "lightonai/mLateOn",
       device="cuda",
       use_kernels=True,
   )
   embeddings = model.encode_document(["The sky is blue.", "The sun is bright."])

For the :class:`Transformer <sentence_transformers.base.modules.Transformer>` backbone,
Sentence Transformers uses Transformers' kernel integration to select replacements
for supported layers.
Multi-vector models can also accelerate MaxSim scoring and hierarchical token
pooling. The following sections explain how to use each of these.

:class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`,
:class:`CrossEncoder <sentence_transformers.cross_encoder.model.CrossEncoder>`, and
:class:`SparseEncoder <sentence_transformers.sparse_encoder.model.SparseEncoder>` also accept
``use_kernels=True`` to enable kernels for their
:class:`Transformer <sentence_transformers.base.modules.Transformer>` backbones.

Kernels are selected for the device running each operation. Enabling them does
not mean that every operation has a replacement: when no compatible kernel is
available for a device or build, the original implementation remains in use.

Accelerating MaxSim scoring
---------------------------

A multi-vector model represents each text with several token vectors. MaxSim
scores a query against a document by finding the best document-token match for
each query token, then summing those matches. Comparing many queries and documents
can therefore involve much more work than comparing one vector per text.

With kernels enabled, :class:`MultiVectorEncoder <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder>` uses the
`sentence-transformers/late-interaction-kernels <https://huggingface.co/kernels/sentence-transformers/late-interaction-kernels>`_
kernel for supported MaxSim operations on CUDA. This kernel is implemented in
`Triton <https://triton-lang.org/>`_, which must be installed to accelerate scoring.
If Triton is not installed, the provider uses PyTorch scoring instead. Enable kernels on the model and score as usual:

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder

   model = MultiVectorEncoder("lightonai/mLateOn", device="cuda", use_kernels=True)

   queries = ["What is the capital of France?", "Which planet is known as the Red Planet?"]
   documents = ["Paris is the capital of France.", "Mars is known as the Red Planet."]

   query_embeddings = model.encode_query(queries)
   document_embeddings = model.encode_document(documents)

   scores = model.similarity(query_embeddings, document_embeddings)
   print(scores)
   # tensor([[ 9.6585,  9.3487],
   #         [11.2298, 11.5521]], device='cuda:0')

For MeanMaxSim (which normalizes scores between -1 and 1), pass
``similarity_fn_name="meanmaxsim"`` when loading the model.

The model's scoring is also used by the multi-vector losses by default, so enabling
kernels can accelerate training too:

.. code-block:: python

   from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss

   loss = CachedMultiVectorMultipleNegativesRankingLoss(model)

By default, losses use the model's kernel-enabled similarity function. Pass
``similarity_fct`` if you want different scoring for training and inference.
Standalone utilities such as ``util.maxsim`` keep their usual non-kernelized implementation.

Accelerating token pooling
--------------------------

Token pooling reduces the number of document vectors you store and score.
:class:`HierarchicalTokenPooling <sentence_transformers.multi_vector_encoder.modules.HierarchicalTokenPooling>`
does this by grouping similar token vectors with Ward
clustering and replacing each group with its mean. A Ward kernel accelerates the
clustering step while the pooler continues to handle protected tokens, cluster
means, and output normalization.

A pooler inherits the model's kernel settings by default. Set ``use_kernels=False``
on :class:`HierarchicalTokenPooling <sentence_transformers.multi_vector_encoder.modules.HierarchicalTokenPooling>`
to opt out.

.. code-block:: python

   from sentence_transformers import MultiVectorEncoder
   from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

   model = MultiVectorEncoder("lightonai/mLateOn", device="cuda", use_kernels=True)
   pooler = HierarchicalTokenPooling(pool_factor=2)

   documents = ["Paris is the capital of France.", "Mars is known as the Red Planet."]
   pooled_embeddings = model.encode_document(documents, token_pooling=pooler)

This enables the default
`sentence-transformers/ward-clustering <https://huggingface.co/kernels/sentence-transformers/ward-clustering>`_
kernel on the pooler. Alternatively, for standalone pooling of existing embeddings, you can also enable kernels on the pooler itself:

.. code-block:: python

   pooler = HierarchicalTokenPooling(pool_factor=2, use_kernels=True)
   pooled_embeddings = pooler.pool(document_embeddings)

An explicit pooler ``kernel_config`` also enables kernels and takes precedence
over the model's Ward configuration.

For example, consider encoding and pooling the first 5,000 abstracts (with titles)
from `SciFact <https://huggingface.co/datasets/BeIR/scifact>`_ with
`lightonai/mLateOn <https://huggingface.co/lightonai/mLateOn>`_. With ``pool_factor=2``, pooling reduces their 1,569,570 token
vectors to 786,032, roughly halving the vectors to store and score.

.. list-table:: Time for 5,000 documents (bf16, batch size 32, RTX 3090)
   :header-rows: 1

   * - Operation
     - Without kernels (fastcluster)
     - Ward kernel
   * - Pooling existing embeddings
     - 7.00s
     - 1.02s
   * - Encoding and pooling
     - 23.14s
     - 18.91s

Pooling itself is about 6.9 times faster here, reducing the total time for encoding
and pooling by 18%.

Choosing your own kernels
-------------------------

Use :class:`~transformers.KernelConfig` when you want to replace one of the defaults with a particular
provider. Each mapping connects an operation name to a repository and the class
implementing its replacement. For example, suppose you have a compatible MaxSim
provider in ``your-organization/your-kernels``:

.. code-block:: python

   from transformers import KernelConfig

   model = MultiVectorEncoder(
       "lightonai/mLateOn",
       device="cuda",
       kernel_config=KernelConfig(
           {
               "STMaxSim": {
                   "cuda": (
                       "your-organization/your-kernels:MaxSim",
                       {"version": 1, "trust_remote_code": True},
                   ),
               },
           },
       ),
   )

The configuration enables kernels and replaces all-pairs scoring on CUDA with
your repository's ``MaxSim`` class. Other operations keep their default kernels. Further,
``trust_remote_code=True`` allows the custom provider's code to run even if its
organization is not a `trusted kernel publisher <https://huggingface.co/docs/kernels/security#trusted-publishers>`_
on the Hub. See also the `Transformers' kernel loading guide <https://huggingface.co/docs/transformers/main/en/kernel_doc/loading_kernels#kernelconfig>`_
for more mapping options.

Choose the operation name according to what you want to replace:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Operation name
     - What it controls
   * - ``STMaxSim``
     - All-pairs scores from ``model.similarity``.
   * - ``STMaxSimPairwise``
     - Aligned-pair scores from ``model.similarity_pairwise``.
   * - ``STWardClustering``
     - Ward clustering in hierarchical token poolers that inherit model settings.
   * - Backbone layer names, such as ``RMSNorm``
     - Transformer layers, configured through Transformers' kernel integration.

For backbone-only settings, you can also pass ``use_kernels`` and ``kernel_config``
through ``model_kwargs``. Those settings take precedence for the backbone. For
example, ``use_kernels=True, model_kwargs={"use_kernels": False}`` enables the
multi-vector model's scoring and pooling kernels while keeping the backbone's
original implementations.

.. tip::

   The recommended way to select specialized attention implementations remains
   through Transformers' ``attn_implementation`` option in ``model_kwargs``. Options
   include ``"sdpa"``, ``"flash_attention_2"``, and ``"kernels-community/flash-attn2"``,
   depending on model and hardware support. For example,
   ``model_kwargs={"attn_implementation": "kernels-community/flash-attn2"}`` loads
   FlashAttention 2 from the Hub.

Reloading a model
-----------------

Kernel choices are runtime settings. They are not stored in ordinary model
checkpoints, so enable kernels again when loading:

.. code-block:: python

   model.save_pretrained("my_model")
   model = MultiVectorEncoder("my_model", device="cuda", use_kernels=True)
