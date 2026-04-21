Creating Custom Models
======================

Modular Architecture
--------------------

Every CrossEncoder is composed of sequential modules, just like a
:class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`. The :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` class
is a subclass of :class:`~sentence_transformers.base.model.BaseModel`, which inherits from :class:`torch.nn.Sequential`.
You can inspect the module chain by printing the model:

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
   print(model)
   """
   CrossEncoder(
     (0): Transformer({'transformer_task': 'text-generation', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}, 'message': {'method': 'forward', 'method_output_name': 'logits', 'format': 'flat'}}, 'module_output_name': 'causal_logits', 'architecture': 'Qwen3ForCausalLM'})
     (1): LogitScore({'true_token_id': 9693, 'false_token_id': 2152, 'module_input_name': 'causal_logits'})
   )
   """

Module Chains
^^^^^^^^^^^^^

There are several standard module chain patterns for :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models:

**Encoder-based (Sequence Classification)**

For traditional encoder-based CrossEncoders (e.g. BERT, RoBERTa), a single module is used:

* :class:`~sentence_transformers.base.modules.Transformer` with ``transformer_task="sequence-classification"``: Loads the model via :class:`~transformers.AutoModelForSequenceClassification` and returns the classification scores directly.

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
   print(model)
   """
   CrossEncoder(
     (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
   )
   """

**CausalLM-based (Text Generation + LogitScore)**

For generative rerankers (e.g. Qwen, Llama), the modules are generally:

* :class:`~sentence_transformers.base.modules.Transformer` with ``transformer_task="text-generation"``: Loads the model via :class:`~transformers.AutoModelForCausalLM` and returns the raw logits from the language model head.
* :class:`~sentence_transformers.cross_encoder.modules.LogitScore`: Extracts the logits at the last token position and computes a score. If both ``true_token_id`` and ``false_token_id`` are set, the score is the log-odds: ``logit[true] - logit[false]``. If only ``true_token_id`` is set, the score is simply the logit for that token.

This is determined automatically in :meth:`CrossEncoder._load_default_modules() <sentence_transformers.cross_encoder.model.CrossEncoder._load_default_modules>`:
when the model architecture ends with ``ForCausalLM``, the :class:`~sentence_transformers.base.modules.Transformer` is loaded with ``transformer_task="text-generation"``
and a :class:`~sentence_transformers.cross_encoder.modules.LogitScore` module is appended with ``true_token_id`` and
``false_token_id`` based on the ``"yes"`` and ``"no"`` tokens, respectively. Otherwise, the
:class:`~sentence_transformers.base.modules.Transformer` is loaded with ``transformer_task="sequence-classification"`` (the traditional encoder-based approach).

**Feature Extraction + Pooling + Dense**

A more memory-efficient alternative to the Text Generation + LogitScore approach is to use feature extraction without the LM head:

* :class:`~sentence_transformers.base.modules.Transformer` with ``transformer_task="feature-extraction"``: Loads only the base model via :class:`~transformers.AutoModel` (no LM head), outputting hidden states.
* :class:`~sentence_transformers.sentence_transformer.modules.Pooling` with ``pooling_mode="lasttoken"``: Extracts the hidden state of the last token.
* :class:`~sentence_transformers.base.modules.Dense`: Projects the hidden state to a single score.

.. code-block:: python

   from sentence_transformers import CrossEncoder
   from sentence_transformers.cross_encoder.modules import Transformer, Dense
   from sentence_transformers.sentence_transformer.modules import Pooling

   transformer = Transformer("Qwen/Qwen3.5-0.8B", transformer_task="feature-extraction")
   pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="lasttoken")

   # Initialize Dense weights to approximate LogitScore: weight = embed("1") - embed("0")
   true_id = transformer.tokenizer.convert_tokens_to_ids("1")
   false_id = transformer.tokenizer.convert_tokens_to_ids("0")
   embeddings = transformer.model.get_input_embeddings().weight.data
   init_weight = (embeddings[true_id] - embeddings[false_id]).unsqueeze(0)

   dense = Dense(
       in_features=transformer.get_embedding_dimension(),
       out_features=1,
       bias=False,
       activation_function=None,
       init_weight=init_weight,
       module_output_name="scores",
   )

   model = CrossEncoder(modules=[transformer, pooling, dense])

Because most causal language models (LM) tie input embeddings with the LM head weights, initializing the Dense weight as ``embed("1") - embed("0")`` gives a starting point equivalent to computing log-odds over the ``"1"`` and ``"0"`` tokens. This approach skips the expensive LM head computation over the full vocabulary.

.. tip::

   See the `multimodal training examples <https://github.com/UKPLab/sentence-transformers/tree/master/examples/cross_encoder/training/multimodal>`_ for a complete comparison of the Text Generation + LogitScore and Feature Extraction + Pooling + Dense approaches.

Constructing Custom Module Chains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can manually construct a :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` with custom modules. For example, to use ``"1"`` and ``"0"`` as the true/false tokens instead of the default ``"yes"`` and ``"no"``:

.. code-block:: python

   from sentence_transformers import CrossEncoder
   from sentence_transformers.cross_encoder.modules import Transformer, LogitScore

   transformer = Transformer("Qwen/Qwen3-Reranker-0.6B", transformer_task="text-generation")

   # Look up the token IDs for "1" and "0" in the tokenizer
   true_id = transformer.tokenizer.convert_tokens_to_ids("1")
   false_id = transformer.tokenizer.convert_tokens_to_ids("0")

   model = CrossEncoder(
       modules=[transformer, LogitScore(true_token_id=true_id, false_token_id=false_id)]
   )

Saving CrossEncoder Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever a CrossEncoder model is saved via :meth:`CrossEncoder.save_pretrained <sentence_transformers.cross_encoder.model.CrossEncoder.save_pretrained>`, three types of files are generated:

* ``modules.json``: A list of module names, paths, and types used to reconstruct the model.
* ``config_sentence_transformers.json``: Configuration options including the model type, saved prompts, the default prompt name, and the activation function.
* **Module-specific files**: Each module is saved in a separate subfolder named after the module index and class name (e.g., ``1_LogitScore``), except the first module which may be saved in the root directory if it has ``save_in_root = True`` (this is the case for the :class:`~sentence_transformers.base.modules.Transformer` module).

For example, saving a CausalLM-based :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` produces the following directory structure:

.. code-block:: bash

   my-cross-encoder/
   ├── 1_LogitScore
   │   └── config.json
   ├── README.md
   ├── chat_template.jinja
   ├── config.json
   ├── config_sentence_transformers.json
   ├── generation_config.json
   ├── model.safetensors
   ├── modules.json
   ├── sentence_bert_config.json
   ├── tokenizer.json
   └── tokenizer_config.json

The ``modules.json`` contains metadata about each module:

.. code-block:: json

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.base.modules.transformer.Transformer"
     },
     {
       "idx": 1,
       "name": "1",
       "path": "1_LogitScore",
       "type": "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
     }
   ]

The ``config_sentence_transformers.json`` contains model-level configuration:

.. code-block:: json

   {
     "__version__": {
       "sentence_transformers": "5.4.0",
       "transformers": "5.5.0",
       "pytorch": "2.10.0"
     },
     "activation_fn": "torch.nn.modules.linear.Identity",
     "default_prompt_name": "query",
     "model_type": "CrossEncoder",
     "prompts": {
       "query": "Given a web search query, retrieve relevant passages that answer the query"
     }
   }

And the ``1_LogitScore/config.json`` stores the LogitScore module's configuration:

.. code-block:: json

   {
       "true_token_id": 9693,
       "false_token_id": 2152,
       "module_input_name": "causal_logits"
   }

The ``sentence_bert_config.json`` stores the :class:`~sentence_transformers.base.modules.Transformer` module's configuration:

.. code-block:: json

   {
       "transformer_task": "text-generation",
       "modality_config": {
           "text": {
               "method": "forward",
               "method_output_name": "logits"
           },
           "message": {
               "method": "forward",
               "method_output_name": "logits",
               "format": "flat"
           }
       },
       "module_output_name": "causal_logits"
   }

The ``"format"`` key in the ``"message"`` modality controls how chat-template inputs are structured:

* ``"structured"``: Content is a list of typed dicts, e.g. ``[{"type": "text", "text": "hello"}]``.
* ``"flat"``: Content is the direct value, e.g. ``"hello"``.

This is inferred automatically from the model's chat template. Most vision-language models use structured format, while text-only causal LMs typically use flat format.

Loading CrossEncoder Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` model, ``modules.json`` is read to determine the modules that make up the model. Each module class is resolved from its ``type`` field, and the module is initialized from the configuration stored in the corresponding module directory. For example, the :class:`~sentence_transformers.cross_encoder.modules.LogitScore` module is loaded by reading ``1_LogitScore/config.json`` and passing the values as keyword arguments to ``LogitScore(...)``.

If no ``modules.json`` is found (e.g., loading a pure ``transformers`` model), the :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder._load_default_modules` method automatically determines the module chain based on the model architecture: ``ForCausalLM`` models get ``[Transformer, LogitScore]``, while all other models get ``[Transformer]`` with ``transformer_task="sequence-classification"``.

Multimodal CrossEncoder Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~sentence_transformers.base.modules.Transformer` module handles multimodal inputs natively. It automatically detects the supported modalities of the underlying model (text, image, audio, video) via ``modality_config`` and routes inputs to the correct processing method. You can build a multimodal :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` by using a multimodal backbone:

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("Qwen/Qwen3-VL-Reranker-2B")
   print(model)
   """
   CrossEncoder(
     (0): Transformer({'transformer_task': 'any-to-any', 'modality_config': {'text': ..., 'image': ..., 'video': ..., 'message': {..., 'format': 'structured'}}, 'module_output_name': 'causal_logits', 'processing_kwargs': {...}, 'unpad_inputs': False, 'architecture': 'Qwen3VLForConditionalGeneration'})
     (1): LogitScore({'true_token_id': 9693, 'false_token_id': 2152, 'module_input_name': 'causal_logits'})
   )
   """

   # Score text-only and image-text pairs
   scores = model.predict([
       ("A bee on a pink flower", "This image shows a bee on a pink flower"),
       ("A bee on a pink flower", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"),
   ])

When building a multimodal :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` from scratch (e.g. from a base multimodal model), use ``transformer_task="any-to-any"`` to load the full causal LM with its language model head:

.. code-block:: python

   from sentence_transformers import CrossEncoder
   from sentence_transformers.cross_encoder.modules import Transformer, LogitScore

   transformer = Transformer("Qwen/Qwen3.5-0.8B", transformer_task="any-to-any")
   score_head = LogitScore(
       true_token_id=transformer.tokenizer.convert_tokens_to_ids("1"),
       false_token_id=transformer.tokenizer.convert_tokens_to_ids("0"),
   )

   model = CrossEncoder(
       modules=[transformer, score_head],
       prompts={
           "image_to_text": "Given the image, judge whether the text matches it. Respond with 1 if they match, 0 if they don't.",
           "text_to_image": "Given the text, judge whether the image matches it. Respond with 1 if they match, 0 if they don't.",
       },
   )

Advanced: Custom Modules
------------------------

All modules available for building a :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` can be imported from :mod:`sentence_transformers.cross_encoder.modules`:

* :class:`~sentence_transformers.base.modules.Transformer`: The primary input module for loading Transformers models.
* :class:`~sentence_transformers.cross_encoder.modules.LogitScore`: Converts causal LM logits into relevance scores.
* :class:`~sentence_transformers.base.modules.Dense`: A feed-forward layer for dimensionality reduction or score projection.
* :class:`~sentence_transformers.base.modules.Router`: Routes inputs through different module chains based on task or modality.
* :class:`~sentence_transformers.base.modules.Module`: Base class for creating custom non-input modules.
* :class:`~sentence_transformers.base.modules.InputModule`: Base class for creating custom input modules.

Input Modules
^^^^^^^^^^^^^

The first module in a pipeline is called the input module. It is responsible for preprocessing the inputs and generating the input features for the subsequent modules. The input module can be any module that implements the :class:`~sentence_transformers.base.modules.InputModule` class, which is a subclass of the :class:`~sentence_transformers.base.modules.Module` class.

It has two abstract methods that you need to implement, and one method that you should override:

* A :meth:`~sentence_transformers.base.modules.Module.forward` method that accepts a ``features`` dictionary and returns an updated ``features`` dictionary.
* A :meth:`~sentence_transformers.base.modules.Module.save` method that saves the module's configuration and optionally weights to a provided directory.
* A :meth:`~sentence_transformers.base.modules.InputModule.preprocess` method that accepts a list of inputs and an optional ``prompt`` string, and returns a ``features`` dictionary that will be passed to the module's own ``forward`` method. The keys should match what ``forward`` expects (e.g. ``input_ids``, ``attention_mask``, ``pixel_values``, etc.). The base :class:`~sentence_transformers.base.modules.InputModule` class provides a default implementation that delegates to ``tokenize()``, but you should override it. Its full signature is:

  .. code-block:: python

     def preprocess(
         self,
         inputs: list[SingleInput | PairInput],
         prompt: str | None = None,
         **kwargs,
     ) -> dict[str, torch.Tensor | Any]:
         ...

  For :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models, the ``inputs`` parameter contains pairs of inputs. Each element in a pair can be a text string, a PIL image, a numpy/torch array for audio or video, a multimodal dictionary with modality keys (e.g. ``{"text": ..., "image": ...}``), or a chat-style message list.

Optionally, you can also implement or override the following:

* A :attr:`~sentence_transformers.base.modules.InputModule.modalities` property that returns the list of modalities supported by the module. Defaults to ``["text"]``. Each entry can be a single modality string (e.g. ``"text"``) or a tuple for compound modalities (e.g. ``("image", "text")``); tuples should be sorted alphabetically. :meth:`BaseModel.preprocess <sentence_transformers.base.model.BaseModel.preprocess>` validates input modalities against this list before calling the module's ``preprocess()``, so you don't need to handle unsupported modalities yourself.
* A :meth:`~sentence_transformers.base.modules.Module.load` class method that loads the module from a saved directory or Hugging Face model.
* A ``max_seq_length`` property that returns the maximum sequence length the module can process.

.. note::

   The ``tokenize()`` method is deprecated in favor of ``preprocess()``. If you are writing a new custom input module, implement ``preprocess()`` instead.

Subsequent Modules
^^^^^^^^^^^^^^^^^^

Subsequent modules in the pipeline are called non-input modules. They process the features dictionary produced by the input module and either transform it or extract final scores. Non-input modules can be any module that implements the :class:`~sentence_transformers.base.modules.Module` class.

It has two abstract methods that you need to implement:

* A :meth:`~sentence_transformers.base.modules.Module.forward` method that accepts a ``features`` dictionary and returns an updated ``features`` dictionary. For :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models, the final module should set the ``"scores"`` key.
* A :meth:`~sentence_transformers.base.modules.Module.save` method that saves the module's configuration and optionally weights to a provided directory.

Optionally, you can also implement:

* A :meth:`~sentence_transformers.base.modules.Module.load` class method that loads the module from a saved directory or Hugging Face model.

Example Custom Module
^^^^^^^^^^^^^^^^^^^^^

For example, we can create a custom scoring module that applies temperature scaling to the logits before computing the log-odds score:

.. code-block:: python

   # temperature_logit_score.py

   import torch
   from sentence_transformers.cross_encoder.modules import Module


   class TemperatureLogitScore(Module):
       config_keys: list[str] = ["true_token_id", "false_token_id", "temperature", "module_input_name"]

       def __init__(
           self,
           true_token_id: int,
           false_token_id: int | None = None,
           temperature: float = 1.0,
           module_input_name: str = "causal_logits",
           **kwargs,
       ) -> None:
           super().__init__()
           self.true_token_id = true_token_id
           self.false_token_id = false_token_id
           self.temperature = temperature
           self.module_input_name = module_input_name

       def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
           logits = features[self.module_input_name][:, -1]
           # Apply temperature scaling
           logits = logits / self.temperature

           if self.false_token_id is None:
               scores = logits[:, self.true_token_id]
           else:
               scores = logits[:, self.true_token_id] - logits[:, self.false_token_id]

           features["scores"] = scores.unsqueeze(1)
           return features

       def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
           self.save_config(output_path)

       # The default `load` method reads `config.json` and passes the values
       # (i.e. the `config_keys`) as kwargs to __init__. This works for us,
       # so no need to override it.

Key points about this example:

* ``config_keys``: Lists the instance attributes that should be saved in ``config.json``. These are automatically extracted by :meth:`~sentence_transformers.base.modules.Module.get_config_dict` and used by the default :meth:`~sentence_transformers.base.modules.Module.load` method to reconstruct the module.
* ``forward``: Reads from the ``features`` dictionary and writes the ``"scores"`` key, which is what :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.predict` expects from the final module.
* ``save``: Calls :meth:`~sentence_transformers.base.modules.Module.save_config` to write the config. Since this module has no trainable weights, there is nothing else to save. A module with trainable parameters should also call :meth:`~sentence_transformers.base.modules.Module.save_torch_weights`.

This can now be used as a module in a :class:`~sentence_transformers.cross_encoder.model.CrossEncoder`:

.. code-block:: python

   from sentence_transformers import CrossEncoder
   from sentence_transformers.cross_encoder.modules import Transformer
   from temperature_logit_score import TemperatureLogitScore

   transformer = Transformer("Qwen/Qwen3-Reranker-0.6B", transformer_task="text-generation")
   score_head = TemperatureLogitScore(
       true_token_id=transformer.tokenizer.convert_tokens_to_ids("yes"),
       false_token_id=transformer.tokenizer.convert_tokens_to_ids("no"),
       temperature=2.0,
   )

   model = CrossEncoder(modules=[transformer, score_head])
   print(model)
   """
   CrossEncoder(
     (0): Transformer({'transformer_task': 'text-generation', ...})
     (1): TemperatureLogitScore({'true_token_id': 9693, 'false_token_id': 2152, 'temperature': 2.0, 'module_input_name': 'causal_logits'})
   )
   """

   scores = model.predict([("How many people live in Berlin?", "Berlin has a population of 3,520,031.")])

You can save this model with :meth:`CrossEncoder.save_pretrained <sentence_transformers.cross_encoder.model.CrossEncoder.save_pretrained>`, resulting in a ``modules.json`` of:

.. code-block:: json

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.base.modules.transformer.Transformer"
     },
     {
       "idx": 1,
       "name": "1",
       "path": "1_TemperatureLogitScore",
       "type": "temperature_logit_score.TemperatureLogitScore"
     }
   ]

To ensure that ``temperature_logit_score.TemperatureLogitScore`` can be imported, you should copy the ``temperature_logit_score.py`` file to the directory where you saved the model. If you push the model to the `Hugging Face Hub <https://huggingface.co/models>`_, then you should also upload the ``temperature_logit_score.py`` file to the model's repository. Then, everyone can use your custom module by calling :meth:`CrossEncoder("your-username/your-model-id", trust_remote_code=True) <sentence_transformers.cross_encoder.model.CrossEncoder>`.

.. note::

   Using a custom module with remote code stored on the Hugging Face Hub requires that your users specify ``trust_remote_code`` as ``True`` when loading the model. This is a security measure to prevent remote code execution attacks.

.. note::

   Adding ``**kwargs`` to the ``__init__``, ``forward``, ``save``, ``load``, and ``preprocess`` methods is recommended to ensure that the methods remain compatible with future updates to the Sentence Transformers library.

If you have your models and custom modelling code on the Hugging Face Hub, then it might make sense to separate your custom modules into a separate repository. This way, you only have to maintain one implementation of your custom module, and you can reuse it across multiple models. You can do this by updating the ``type`` in ``modules.json`` to include the path to the repository where the custom module is stored like ``{repository_id}--{dot_path_to_module}``. For example, if the ``temperature_logit_score.py`` file is stored in a repository called ``my-user/my-model-implementation``, then the ``modules.json`` file may look like this:

.. code-block:: json

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.base.modules.transformer.Transformer"
     },
     {
       "idx": 1,
       "name": "1",
       "path": "1_TemperatureLogitScore",
       "type": "my-user/my-model-implementation--temperature_logit_score.TemperatureLogitScore"
     }
   ]

Advanced: Keyword Argument Passthrough in Custom Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want your users to be able to specify custom keyword arguments via the :meth:`CrossEncoder.predict <sentence_transformers.cross_encoder.model.CrossEncoder.predict>` method, then you can add their names to the ``modules.json`` file. For example, if your module should behave differently based on a ``task`` keyword argument, then your ``modules.json`` might look like:

.. code-block:: json

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "custom_transformer.CustomTransformer",
       "kwargs": ["task"]
     },
     {
       "idx": 1,
       "name": "1",
       "path": "1_LogitScore",
       "type": "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
     }
   ]

Then, you can access the ``task`` keyword argument in the ``forward`` method of your custom module:

.. code-block:: python

   from sentence_transformers.cross_encoder.modules import Transformer

   class CustomTransformer(Transformer):
       def forward(self, features: dict[str, torch.Tensor], task: str | None = None, **kwargs) -> dict[str, torch.Tensor]:
           if task == "default":
               # Do something
               ...
           else:
               # Do something else
               ...
           return features

This way, users can specify the ``task`` keyword argument when calling :meth:`CrossEncoder.predict <sentence_transformers.cross_encoder.model.CrossEncoder.predict>`:

.. code-block:: python

   from sentence_transformers import CrossEncoder

   model = CrossEncoder("your-username/your-model-id", trust_remote_code=True)
   pairs = [("query", "document")]
   model.predict(pairs, task="default")
