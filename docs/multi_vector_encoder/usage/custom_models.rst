Creating Custom Models
======================

Modular Architecture
--------------------

A MultiVectorEncoder consists of modules that are executed sequentially, just like the other model
families. Everything that is shared across families is documented once, on the
`Sentence Transformers > Creating Custom Models <../../sentence_transformer/usage/custom_models.html>`_
page: the general module API, the saving format, writing your own
:class:`~sentence_transformers.base.modules.Module` or :class:`~sentence_transformers.base.modules.InputModule`,
passing keyword arguments through to custom modules, and distributing them via the Hub. This page
covers what is specific to multi-vector models. The default multi-vector pipeline is:

* :class:`~sentence_transformers.base.modules.Transformer`: processes the input and produces contextualized
  token embeddings. The multi-vector knobs (``query_length``, ``document_length``, ``query_expansion``) live
  here, and all start unset: inputs are capped only by the tokenizer's own maximum and queries are not
  expanded. ColBERT-style query expansion (``query_expansion={"strategy": "min", "length": 32}``, padding
  queries out to 32 ``[MASK]`` tokens, with ``"min"`` letting longer queries pass through where the classic
  ``"fixed"`` strategy would truncate them) is an explicit recipe choice, as in the MS MARCO examples.
* :class:`~sentence_transformers.base.modules.Dense` (token-level): projects each token embedding down to the
  multi-vector dimension, via ``module_input_name="token_embeddings"``. Defaults to 128 output dimensions
  with no bias and no activation, the classic ColBERT projection. Randomly initialized unless the checkpoint
  ships one, so a fresh model needs training before it is useful.
* :class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask`: overwrites ``attention_mask``
  with the per-row scoring mask (force-including query expansion positions, excluding document skiplist
  tokens). The skiplist starts empty, so every real token is scored. Legacy PyLate and Stanford-NLP
  checkpoints pre-seed it with punctuation instead.
* :class:`~sentence_transformers.sentence_transformer.modules.Normalize` (token-level): L2-normalizes each
  token embedding, so each MaxSim term is a cosine similarity. No configuration.

Beyond the modules, a fresh model scores with MaxSim and carries empty ``prompts``. The classic ColBERT
recipe also expands queries to 32 tokens with ``[MASK]`` padding, prepends "[Q] " and "[D] " marker tokens,
caps document length, and skips punctuation tokens during document scoring, which released checkpoints
configure themselves but a bare backbone does not.
To reproduce the full classic recipe, initialize the modules explicitly::

    import string

    from torch import nn

    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.base.modules import Dense, Transformer
    from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
    from sentence_transformers.base.modules import Normalize

    transformer = Transformer(
        "answerdotai/ModernBERT-base",
        query_expansion={"strategy": "fixed", "length": 32},  # pad queries to 32 tokens with [MASK], truncate longer ones
        document_length=300,  # also truncate (not pad) documents to 300 tokens
    )
    dense = Dense(
        in_features=transformer.get_embedding_dimension(),
        out_features=128,
        bias=False,
        activation_function=nn.Identity(),
        module_input_name="token_embeddings",
    )
    mask = MultiVectorMask(skiplist_words=list(string.punctuation))  # skip punctuation for documents
    normalize = Normalize(module_input_name="token_embeddings")

    model = MultiVectorEncoder(
        modules=[transformer, dense, mask, normalize],
        prompts={"query": "[Q] ", "document": "[D] "},
    )

An optional :class:`~sentence_transformers.multi_vector_encoder.modules.HierarchicalTokenPooling`
module can be appended after ``Normalize`` to bake document token pooling into the checkpoint.

What Released Checkpoints Configure
-----------------------------------

The loadable checkpoint families all end up in the same module system, but they carry different
knob values, and different pieces live in different places (the module count and order vary per
family). ``print(model)`` shows the stack with every knob in the module configs, and
``model.prompts`` holds the query / document markers::

    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder("colbert-ir/colbertv2.0")
    print(model)
    # MultiVectorEncoder(
    #   (0): Transformer({..., 'document_length': 180,
    #                     'query_expansion': {'strategy': 'fixed', 'attend': False, 'token': None, 'length': 32}})
    #   (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, ...})
    #   (2): MultiVectorMask({'skiplist_words': ['!', '"', '#', ..., '}', '~'], 'skiplist_tasks': ['document'], 'keep_only_token_ids': None})
    #   (3): Normalize({...})
    # )
    print(model.prompts)
    # {'query': '[unused0] ', 'document': '[unused1] '}

Native and PyLate Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Checkpoints saved by this library ship a ``modules.json`` that defines the stack directly, so
nothing is inferred. PyLate builds on the same schema but stored some knobs as top-level config
fields, which are translated on load: ``query_prefix`` / ``document_prefix`` become the
``prompts`` dict, ``do_query_expansion`` / ``attend_to_expansion_tokens`` / ``query_length`` fold
into the Transformer's ``query_expansion`` config, and ``skiplist_words`` seeds the
:class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask` (defaulting to
punctuation for PyLate saves). A PyLate ``modules.json`` lists only ``[Transformer, Dense]``,
because masking and normalization were inline in PyLate, so the scoring mask and the token-level
Normalize are appended on load. As a concrete example, ``lightonai/GTE-ModernColBERT-v1`` loads as
the four default modules with prompts ``"[Q] "`` / ``"[D] "``, no query expansion (LightOn disables
it), a query length cap of 48, a document length cap of 300, and a punctuation skiplist.

Stanford-NLP ColBERT Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stanford-NLP ColBERT checkpoints like `colbert-ir/colbertv2.0 <https://huggingface.co/colbert-ir/colbertv2.0>`_ are
detected via the ``HF_ColBERT`` architecture marker in ``config.json``. The projection weight is
stored inline at the repo root (``linear.weight``) and becomes the token-level ``Dense`` module,
and ``artifact.metadata`` supplies the recipe: ``query_token_id`` / ``doc_token_id`` become the
prompts (``"[unused0] "`` / ``"[unused1] "`` by default), ``doc_maxlen`` becomes
``document_length``, ``query_maxlen`` and ``attend_to_mask_tokens`` become a
``{"strategy": "fixed", ...}`` query expansion (Stanford-NLP ColBERT always ``[MASK]``-expands
queries), and ``mask_punctuation`` seeds the skiplist. ``colbert-ir/colbertv2.0`` loads exactly as
the inspection snippet above shows: queries processed at exactly 32 tokens (the ``fixed`` strategy
both pads shorter queries with non-attending ``[MASK]`` tokens and truncates longer ones),
documents truncated at 180 tokens, and punctuation skipped during document scoring.

Transformers-Native Retrievers (ColPali Family)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``*ForRetrieval`` architectures (ColPali, ColQwen2, ColModernVBert) like `vidore/colpali-v1.3-hf <https://huggingface.co/vidore/colpali-v1.3-hf>`_ load as just
``[Transformer, MultiVectorMask]``: the projection, the L2 normalization, and the zeroing of
padded positions all live inside the transformers model, and the processor bakes in the query
prefix, the query augmentation buffer, and the visual prompt. There are no recipe knobs to
replicate by hand: to get something like ColPali, load the ``ForRetrieval`` checkpoint and
fine-tune it (see the `multimodal training examples <../training/examples.html>`_). Text inputs are
always rendered as queries by the processor, and image documents flow through the same
``encode_document`` path. To filter document embeddings down to image-patch tokens only (as the
ColPali heatmap example does), set the mask module's ``keep_only_token_ids`` to ``model.processor.image_token_id``
or ``[model.processor.tokenizer.convert_tokens_to_ids(model.processor.image_token)]``.

Saving Multi-Vector Encoder Models
----------------------------------

``save_pretrained`` writes the native format regardless of what was loaded: converted Stanford-NLP
and PyLate checkpoints save as regular Sentence Transformers checkpoints (``modules.json`` plus a
config per module), with the prompts in ``config_sentence_transformers.json`` and the length /
expansion / skiplist knobs in the module configs. The conversion is one-time, so loading the save
afterwards repeats no translation.

A worked example that bakes document token pooling into the shipped model::

    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    model = MultiVectorEncoder("colbert-ir/colbertv2.0")
    model.append(HierarchicalTokenPooling(pool_factor=2))

    model.save_pretrained("colbertv2-pooled")
    # or share it on the Hugging Face Hub:
    model.push_to_hub("username/colbertv2-pooled")

Every consumer of the saved checkpoint now receives pooled document embeddings (queries stay
unpooled), with the pooling stored as a fifth entry in ``modules.json``.

Loading Multi-Vector Encoder Models
-----------------------------------

Every checkpoint family loads through the same call, with the format detected from the checkpoint
itself::

    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder("lightonai/LateOn")

Checkpoints in the native format ship a ``modules.json`` that defines the stack directly: it is read
to determine which modules make up the model, and each module is initialized with the configuration
stored in the corresponding module directory. The other families are recognized from their own
metadata instead (the ``HF_ColBERT`` marker or a ``*ForRetrieval`` architecture in ``config.json``,
or PyLate's top-level fields in ``config_sentence_transformers.json``), and the modules they do not
spell out are appended on load. See `what each checkpoint family configures <#what-released-checkpoints-configure>`_
above, and `Usage <usage.html>`_ for a loading call per family.

Extra Per-Token Features
------------------------

Every consumer in the pipeline (``encode``, the losses, gradient caching, ``model.similarity``, and
any vector index) transports exactly one per-token tensor, ``token_embeddings``, plus its
``attention_mask``. Custom modules that produce additional per-token scalars (e.g. learned token
weights or salience scores) should therefore append them as trailing columns of
``token_embeddings`` rather than as separate feature keys: a trailing column flows through encoding,
gradient caching, padding, multi-GPU gathering, and index storage without any further wiring.

Conventions for a feature-column module:

- Place the module after ``Normalize`` in the pipeline, so the L2 normalization never sees the
  extra column.
- Pair it with a scoring function that splits the column back off (e.g. slice ``[..., :-1]`` for the
  embeddings and ``[..., -1]`` for the weights) and pass it as ``similarity_fct`` to the losses. The
  default MaxSim functions are channel-blind and would fold the extra column into the dot products.
- Note that ``get_embedding_dimension()`` counts the extra columns.

For ad-hoc access to *named* module outputs (without the trailing-column convention), pass
``output_value=None`` to the encode methods to get the raw per-input feature dicts::

    outputs = model.encode_query(queries, output_value=None)
    print(outputs[0].keys())  # dict_keys(['input_ids', 'attention_mask', 'token_embeddings', ...])

Named keys are reachable this way for inspection, but they do not flow through ``model.similarity``
or the cached losses: features that must reach scoring belong in trailing columns.
