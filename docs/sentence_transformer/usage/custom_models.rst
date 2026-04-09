Creating Custom Models
=======================

Modular Architecture
--------------------

A Sentence Transformer model consists of a collection of modules (`docs <../../package_reference/sentence_transformer/modules.html>`_) that are executed sequentially. The most common architecture is a combination of a :class:`~sentence_transformers.base.modules.Transformer` module, a :class:`~sentence_transformers.sentence_transformer.modules.Pooling` module, and optionally, a :class:`~sentence_transformers.base.modules.Dense` module and/or a :class:`~sentence_transformers.sentence_transformer.modules.Normalize` module.

* :class:`~sentence_transformers.base.modules.Transformer`: This module is responsible for processing the input text and generating contextualized embeddings.
* :class:`~sentence_transformers.sentence_transformer.modules.Pooling`: This module reduces the dimensionality of the output from the Transformer module by aggregating the embeddings. Common pooling strategies include mean pooling and CLS pooling.
* :class:`~sentence_transformers.base.modules.Dense`: This module contains a linear layer that post-processes the embedding output from the Pooling module.
* :class:`~sentence_transformers.sentence_transformer.modules.Normalize`: This module normalizes the embedding from the previous layer.

For example, the popular `sentence-transformers/all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ model can also be loaded by initializing the 3 specific modules that make up that model:

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sentence_transformers.sentence_transformer.modules import Transformer, Pooling, Normalize

   transformer = Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
   pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
   normalize = Normalize()

   model = SentenceTransformer(modules=[transformer, pooling, normalize])

Saving Sentence Transformer Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever a Sentence Transformer model is saved, three types of files are generated:

* ``modules.json``: This file contains a list of module names, paths, and types that are used to reconstruct the model.
* ``config_sentence_transformers.json``: This file contains some configuration options of the Sentence Transformer model, including saved prompts, the model its similarity function, and the Sentence Transformer package version used by the model author.
* **Module-specific files**: Each module is saved in separate subfolders named after the module index and the model name (e.g., ``1_Pooling``, ``2_Normalize``), except the first module may be saved in the root directory if it has a ``save_in_root`` attribute set to ``True``. In Sentence Transformers, this is the case for e.g. the :class:`~sentence_transformers.base.modules.Transformer` and :class:`~sentence_transformers.sentence_transformer.modules.StaticEmbedding` modules.
  Most module folders contain a ``config.json`` (or ``sentence_bert_config.json`` for the :class:`~sentence_transformers.base.modules.Transformer` module) file that stores the module's configuration parameters (i.e. the values listed in ``config_keys``). For example, the ``1_Pooling/config.json`` for the :class:`~sentence_transformers.sentence_transformer.modules.Pooling` module might contain::

    {
      "embedding_dimension": 384,
      "pooling_mode": "mean",
      "include_prompt": true
    }

  These values are passed as keyword arguments when initializing the module.

As a result, if I call :meth:`SentenceTransformer.save_pretrained("local-all-MiniLM-L6-v2") <sentence_transformers.sentence_transformer.model.SentenceTransformer.save_pretrained>` on the ``model`` from the previous snippet, the following files are generated:

.. code-block:: bash

   local-all-MiniLM-L6-v2/
   ├── 1_Pooling
   │   └── config.json
   ├── 2_Normalize
   ├── README.md
   ├── config.json
   ├── config_sentence_transformers.json
   ├── model.safetensors
   ├── modules.json
   ├── sentence_bert_config.json
   ├── tokenizer.json
   └── tokenizer_config.json

This contains a ``modules.json`` with these contents:

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
       "path": "1_Pooling",
       "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize"
     }
   ]

And a ``config_sentence_transformers.json`` with these contents:

.. code-block:: json

   {
     "__version__": {
       "sentence_transformers": "5.4.0",
       "transformers": "5.5.0",
       "pytorch": "2.10.0"
     },
     "model_type": "SentenceTransformer",
     "prompts": {
       "document": "",
       "query": ""
     },
     "default_prompt_name": null,
     "similarity_fn_name": "cosine"
   }

Additionally, the ``1_Pooling`` directory contains the configuration file for the :class:`~sentence_transformers.sentence_transformer.modules.Pooling` module, while the ``2_Normalize`` directory is empty because the :class:`~sentence_transformers.sentence_transformer.modules.Normalize` module does not require any configuration. The ``sentence_bert_config.json`` file contains the configuration of the :class:`~sentence_transformers.base.modules.Transformer` module:

.. code-block:: json

   {
       "transformer_task": "feature-extraction",
       "modality_config": {
           "text": {
               "method": "forward",
               "method_output_name": "last_hidden_state"
           }
       },
       "module_output_name": "token_embeddings"
   }

When the model supports the ``"message"`` modality (i.e. models with a chat template), the ``"message"`` entry in ``modality_config`` includes a ``"format"`` key that controls how chat-template inputs are structured:

* ``"structured"``: Content is a list of typed dicts, e.g. ``[{"type": "text", "text": "hello"}]``.
* ``"flat"``: Content is the direct value, e.g. ``"hello"``.

This is inferred automatically from the model's chat template. Most vision-language models use structured format, while text-only causal LMs typically use flat format. The ``"format"`` key is only saved when the model supports the ``"message"`` modality, which is why it does not appear in the example above.

This module also saved a lot of files related to the tokenizer and the model itself in the root directory.

Loading Sentence Transformer Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a Sentence Transformer model from a saved model directory, the ``modules.json`` is read to determine the modules that make up the model. Each module is initialized with the configuration stored in the corresponding module directory, after which the SentenceTransformer class is instantiated with the loaded modules.

Sentence Transformer Model from a Transformers Model
----------------------------------------------------

When you initialize a Sentence Transformer model with a pure ``transformers`` model (e.g., BERT, RoBERTa, DistilBERT, T5), Sentence Transformers creates a Transformer module and a mean Pooling module by default, except for CausalLM-based models (e.g., Llama, Qwen, Mistral) which use last token Pooling instead.

To be specific, these two snippets are identical::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("google-bert/bert-base-uncased")

::

   from sentence_transformers import SentenceTransformer
   from sentence_transformers.sentence_transformer.modules import Transformer, Pooling

   transformer = Transformer("google-bert/bert-base-uncased")
   pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
   model = SentenceTransformer(modules=[transformer, pooling])

Advanced: Custom Modules
------------------------

Input Modules
^^^^^^^^^^^^^

The first module in a pipeline is called the input module. It is responsible for preprocessing the inputs and generating the input features for the subsequent modules. The input module can be any module that implements the :class:`~sentence_transformers.base.modules.InputModule` class, which is a subclass of the :class:`~sentence_transformers.base.modules.Module` class.

It has two abstract methods that you need to implement, and one method that you should override:

* A :meth:`~sentence_transformers.base.modules.Module.forward` method that accepts a ``features`` dictionary with keys like ``input_ids``, ``attention_mask``, ``token_type_ids``, ``modality``, ``token_embeddings``, and ``sentence_embedding``, depending on where the module is in the model pipeline.
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

  The ``inputs`` parameter can contain text strings, PIL images, numpy/torch arrays for audio or video, multimodal dictionaries with modality keys (e.g. ``{"text": ..., "image": ...}``), or chat-style message lists. The ``prompt`` parameter, when provided, is prepended to text inputs or injected as a system message for message-format inputs.

.. note::

   The ``tokenize()`` method is deprecated in favor of ``preprocess()``. The ``tokenize()`` method still works for backward compatibility but will be removed in a future version. If you are writing a new custom input module, implement ``preprocess()`` instead.

Optionally, you can also implement or override the following:

* A :attr:`~sentence_transformers.base.modules.InputModule.modalities` property that returns the list of modalities supported by the module (e.g. ``["text"]``, ``["text", "image"]``). Defaults to ``["text"]``. Override this if your module supports non-text inputs.
* A :meth:`~sentence_transformers.base.modules.Module.load` class method that accepts a ``model_name_or_path`` argument, keyword arguments for loading from Hugging Face (``subfolder``, ``token``, ``cache_folder``, etc.) and module kwargs (``model_kwargs``, ``trust_remote_code``, ``backend``, etc.) and initializes the Module given the module's configuration from that directory or model name.
* A ``get_embedding_dimension`` method that returns the dimensionality of the embeddings produced by the module. This is required if the module generates the embeddings or updates the embeddings' dimensionality.
* A ``max_seq_length`` property that returns the maximum sequence length the module can process. Only required if the module processes input text.

Multimodal Input Modules
""""""""""""""""""""""""

The built-in :class:`~sentence_transformers.base.modules.Transformer` module handles multimodal inputs natively. It automatically detects the supported modalities of the underlying model (text, image, audio, video) via ``modality_config`` and routes inputs to the correct processing method. For most use cases, you can simply use :class:`~sentence_transformers.base.modules.Transformer` with a multimodal backbone:

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sentence_transformers.sentence_transformer.modules import Transformer, Pooling

   transformer = Transformer("Qwen/Qwen2-VL-2B-Instruct")
   pooling = Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
   model = SentenceTransformer(modules=[transformer, pooling])

   # Encode text and images together
   embeddings = model.encode([
       "A photo of a cat",
       {"text": "Describe this image", "image": "path/to/image.jpg"},
   ])

If you need to create a custom multimodal input module, override the :attr:`~sentence_transformers.base.modules.InputModule.modalities` property and implement ``preprocess()``. Each ``modalities`` entry can be a single modality string (e.g. ``"image"``) or a tuple for compound modalities (e.g. ``("image", "text")``), but note that tuples should be sorted alphabetically. :meth:`BaseModel.preprocess <sentence_transformers.base.model.BaseModel.preprocess>` validates input modalities against this list before calling the module's ``preprocess()``, so you don't need to handle unsupported modalities yourself. The ``"modality"`` key in the returned features dictionary indicates the input type for downstream modules.

.. code-block:: python

   @property
   def modalities(self):
       return ["image", ("image", "text")]

   def preprocess(self, inputs, prompt=None, **kwargs):
       # Parse and detect the modality of inputs...
       features = self.processor(
           text=[...],
           images=[...],
           return_tensors="pt",
           padding=True,
       )
       # features may contain: input_ids, attention_mask, pixel_values, etc., whatever your forward needs
       features["modality"] = ("image", "text")
       return features

Subsequent Modules
^^^^^^^^^^^^^^^^^^

Subsequent modules in the pipeline are called non-input modules. They are responsible for processing the input features generated by the input module and generating the final embeddings. Non-input modules can be any module that implements the :class:`~sentence_transformers.base.modules.Module` class.

It has two abstract methods that you need to implement:

* A :meth:`~sentence_transformers.base.modules.Module.forward` method that accepts a ``features`` dictionary with keys like ``input_ids``, ``attention_mask``, ``token_type_ids``, ``token_embeddings``, and ``sentence_embedding``, depending on where the module is in the model pipeline.
* A :meth:`~sentence_transformers.base.modules.Module.save` method that saves the module's configuration and optionally weights to a provided directory.

Optionally, you can also implement the following methods:

* A :meth:`~sentence_transformers.base.modules.Module.load` class method that accepts a ``model_name_or_path`` argument, keyword arguments for loading from Hugging Face (``subfolder``, ``token``, ``cache_folder``, etc.) and module kwargs (``model_kwargs``, ``trust_remote_code``, ``backend``, etc.) and initializes the Module given the module's configuration from that directory or model name.
* A ``get_embedding_dimension`` method that returns the dimensionality of the embeddings produced by the module. This is required if the module generates the embeddings or updates the embeddings' dimensionality.

Example Module
^^^^^^^^^^^^^^

For example, we can create a custom pooling method by implementing a custom Module.

.. code-block:: python

   # decay_pooling.py

   import torch
   from sentence_transformers.sentence_transformer.modules import Module


   class DecayMeanPooling(Module):
       config_keys: list[str] = ["dimension", "decay"]

       def __init__(self, dimension: int, decay: float = 0.95, **kwargs) -> None:
           super(DecayMeanPooling, self).__init__()
           self.dimension = dimension
           self.decay = decay

       def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
           # This module is expected to be used after some modules that provide "token_embeddings"
           # and "attention_mask" in the features dictionary.
           token_embeddings = features["token_embeddings"]
           attention_mask = features["attention_mask"].unsqueeze(-1)

           # Apply the attention mask to filter away padding tokens
           token_embeddings = token_embeddings * attention_mask
           # Calculate mean of token embeddings
           sentence_embeddings = token_embeddings.sum(1) / attention_mask.sum(1)
           # Apply exponential decay
           importance_per_dim = self.decay ** torch.arange(
               sentence_embeddings.size(1), device=sentence_embeddings.device
           )
           features["sentence_embedding"] = sentence_embeddings * importance_per_dim
           return features

       def get_embedding_dimension(self) -> int:
           return self.dimension

       def save(self, output_path, *args, safe_serialization=True, **kwargs) -> None:
           self.save_config(output_path)

       # The `load` method by default loads the config.json file from the model directory
       # and initializes the class with the loaded parameters, i.e. the `config_keys`.
       # This works for us, so no need to override it.

.. note::

   Adding ``**kwargs`` to the ``__init__``, ``forward``, ``save``, ``load``, and ``preprocess`` methods is recommended to ensure that the methods remain compatible with future updates to the Sentence Transformers library.

This can now be used as a module in a Sentence Transformer model::
   
   from sentence_transformers import SentenceTransformer
   from sentence_transformers.sentence_transformer.modules import Transformer, Pooling, Normalize
   from decay_pooling import DecayMeanPooling

   transformer = Transformer("google-bert/bert-base-uncased", max_seq_length=256)
   decay_mean_pooling = DecayMeanPooling(transformer.get_embedding_dimension(), decay=0.99)
   normalize = Normalize()

   model = SentenceTransformer(modules=[transformer, decay_mean_pooling, normalize])
   print(model)
   """
   SentenceTransformer(
       (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
       (1): DecayMeanPooling({'dimension': 768, 'decay': 0.99})
       (2): Normalize({})
   )
   """

   texts = [
       "Hello, World!",
       "The quick brown fox jumps over the lazy dog.",
       "I am a sentence that is used for testing purposes.",
       "This is a test sentence.",
       "This is another test sentence.",
   ]
   embeddings = model.encode(texts)
   print(embeddings.shape)
   # [5, 768]

You can save this model with :meth:`SentenceTransformer.save_pretrained <sentence_transformers.sentence_transformer.model.SentenceTransformer.save_pretrained>`, resulting in a ``modules.json`` of::

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
       "path": "1_DecayMeanPooling",
       "type": "decay_pooling.DecayMeanPooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize"
     }
   ]

To ensure that ``decay_pooling.DecayMeanPooling`` can be imported, you should copy over the ``decay_pooling.py`` file to the directory where you saved the model. If you push the model to the `Hugging Face Hub <https://huggingface.co/models>`_, then you should also upload the ``decay_pooling.py`` file to the model's repository. Then, everyone can use your custom module by calling :meth:`SentenceTransformer("your-username/your-model-id", trust_remote_code=True) <sentence_transformers.sentence_transformer.model.SentenceTransformer>`.

.. note::

   Using a custom module with remote code stored on the Hugging Face Hub requires that your users specify ``trust_remote_code`` as ``True`` when loading the model. This is a security measure to prevent remote code execution attacks.

If you have your models and custom modelling code on the Hugging Face Hub, then it might make sense to separate your custom modules into a separate repository. This way, you only have to maintain one implementation of your custom module, and you can reuse it across multiple models. You can do this by updating the ``type`` in ``modules.json`` file to include the path to the repository where the custom module is stored like ``{repository_id}--{dot_path_to_module}``. For example, if the ``decay_pooling.py`` file is stored in a repository called ``my-user/my-model-implementation`` and the module is called ``DecayMeanPooling``, then the ``modules.json`` file may look like this::

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
       "path": "1_DecayMeanPooling",
       "type": "my-user/my-model-implementation--decay_pooling.DecayMeanPooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize"
     }
   ]

Advanced: Keyword argument passthrough in Custom Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want your users to be able to specify custom keyword arguments via the :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>` method, then you can add their names to the ``modules.json`` file. For example, if my module should behave differently if your users specify a ``task`` keyword argument, then your ``modules.json`` might look like::

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
       "path": "1_Pooling",
       "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize"
     }
   ]

Then, you can access the ``task`` keyword argument in the ``forward`` method of your custom module::

   from sentence_transformers.sentence_transformer.modules import Transformer

   class CustomTransformer(Transformer):
       def forward(self, features: dict[str, torch.Tensor], task: Optional[str] = None, **kwargs) -> dict[str, torch.Tensor]:
           if task == "default":
               # Do something
           else:
               # Do something else
           return features

This way, users can specify the ``task`` keyword argument when calling :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("your-username/your-model-id", trust_remote_code=True)
   texts = [...]
   model.encode(texts, task="default")
