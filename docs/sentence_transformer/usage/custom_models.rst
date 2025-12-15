Creating Custom Models
=======================

Structure of Sentence Transformer Models
----------------------------------------

A Sentence Transformer model consists of a collection of modules (`docs <../../package_reference/sentence_transformer/models.html>`_) that are executed sequentially. The most common architecture is a combination of a :class:`~sentence_transformers.models.Transformer` module, a :class:`~sentence_transformers.models.Pooling` module, and optionally, a :class:`~sentence_transformers.models.Dense` module and/or a :class:`~sentence_transformers.models.Normalize` module.

* :class:`~sentence_transformers.models.Transformer`: This module is responsible for processing the input text and generating contextualized embeddings.
* :class:`~sentence_transformers.models.Pooling`: This module reduces the dimensionality of the output from the Transformer module by aggregating the embeddings. Common pooling strategies include mean pooling and CLS pooling.
* :class:`~sentence_transformers.models.Dense`: This module contains a linear layer that post-processes the embedding output from the Pooling module.
* :class:`~sentence_transformers.models.Normalize`: This module normalizes the embedding from the previous layer.

For example, the popular `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_ model can also be loaded by initializing the 3 specific modules that make up that model:

.. code-block:: python

   from sentence_transformers import models, SentenceTransformer

   transformer = models.Transformer("sentence-transformers/all-MiniLM-L6-v2", max_seq_length=256)
   pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
   normalize = models.Normalize()

   model = SentenceTransformer(modules=[transformer, pooling, normalize])

Saving Sentence Transformer Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Whenever a Sentence Transformer model is saved, three types of files are generated:

* ``modules.json``: This file contains a list of module names, paths, and types that are used to reconstruct the model.
* ``config_sentence_transformers.json``: This file contains some configuration options of the Sentence Transformer model, including saved prompts, the model its similarity function, and the Sentence Transformer package version used by the model author.
* **Module-specific files**: Each module is saved in separate subfolders named after the module index and the model name (e.g., ``1_Pooling``, ``2_Normalize``), except the first module may be saved in the root directory if it has a ``save_in_root`` attribute set to ``True``. In Sentence Transformers, this is the case for the :class:`~sentence_transformers.models.Transformer` and :class:`~sentence_transformers.models.CLIPModel` modules.
  Most module folders contain a ``config.json`` (or ``sentence_bert_config.json`` for the :class:`~sentence_transformers.models.Transformer` module) file that stores default values for keyword arguments passed to that Module. So, a ``sentence_bert_config.json`` of::

    {
      "max_seq_length": 4096,
      "do_lower_case": false
    }
  
  means that the :class:`~sentence_transformers.models.Transformer` module will be initialized with ``max_seq_length=4096`` and ``do_lower_case=False``.

As a result, if I call :meth:`SentenceTransformer.save_pretrained("local-all-MiniLM-L6-v2") <sentence_transformers.SentenceTransformer.save_pretrained>` on the ``model`` from the previous snippet, the following files are generated:

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
   ├── special_tokens_map.json
   ├── tokenizer.json
   ├── tokenizer_config.json
   └── vocab.txt

This contains a ``modules.json`` with these contents:

.. code-block:: json

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.models.Transformer"
     },
     {
       "idx": 1,
       "name": "1",
       "path": "1_Pooling",
       "type": "sentence_transformers.models.Pooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.models.Normalize"
     }
   ]

And a ``config_sentence_transformers.json`` with these contents:

.. code-block:: json

   {
     "__version__": {
       "sentence_transformers": "3.0.1",
       "transformers": "4.43.4",
       "pytorch": "2.5.0"
     },
     "prompts": {},
     "default_prompt_name": null,
     "similarity_fn_name": null
   }

Additionally, the ``1_Pooling`` directory contains the configuration file for the :class:`~sentence_transformers.models.Pooling` module, while the ``2_Normalize`` directory is empty because the :class:`~sentence_transformers.models.Normalize` module does not require any configuration. The ``sentence_bert_config.json`` file contains the configuration of the :class:`~sentence_transformers.models.Transformer` module, and this module also saved a lot of files related to the tokenizer and the model itself in the root directory.

Loading Sentence Transformer Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a Sentence Transformer model from a saved model directory, the ``modules.json`` is read to determine the modules that make up the model. Each module is initialized with the configuration stored in the corresponding module directory, after which the SentenceTransformer class is instantiated with the loaded modules.

Sentence Transformer Model from a Transformers Model
----------------------------------------------------

When you initialize a Sentence Transformer model with a pure Transformers model (e.g., BERT, RoBERTa, DistilBERT, T5), Sentence Transformers creates a Transformer module and a Mean Pooling module by default. This provides a simple way to leverage pre-trained language models for sentence embeddings.

To be specific, these two snippets are identical::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("bert-base-uncased")

::

   from sentence_transformers import models, SentenceTransformer
   
   transformer = models.Transformer("bert-base-uncased")
   pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
   model = SentenceTransformer(modules=[transformer, pooling])

Advanced: Custom Modules
------------------------

Input Modules
^^^^^^^^^^^^^

The first module in a pipeline is called the input module. It is responsible for tokenizing the input text and generating the input features for the subsequent modules. The input module can be any module that implements the :class:`~sentence_transformers.models.InputModule` class, which is a subclass of the :class:`~sentence_transformers.models.Module` class.

It has three abstract methods that you need to implement:

* A :meth:`~sentence_transformers.models.Module.forward` method that accepts a ``features`` dictionary with keys like ``input_ids``, ``attention_mask``, ``token_type_ids``, ``token_embeddings``, and ``sentence_embedding``, depending on where the module is in the model pipeline.
* A :meth:`~sentence_transformers.models.Module.save` method that saves the module's configuration and optionally weights to a provided directory.
* A :meth:`~sentence_transformers.models.InputModule.tokenize` method that accepts a list of inputs and returns a dictionary with keys like ``input_ids``, ``attention_mask``, ``token_type_ids``, ``pixel_values``, etc. This dictionary will be passed along to the module's ``forward`` method.

Optionally, you can also implement the following methods:

* A :meth:`~sentence_transformers.models.Module.load` static method that accepts a ``model_name_or_path`` argument, keyword arguments for loading from Hugging Face (``subfolder``, ``token``, ``cache_folder``, etc.) and module kwargs (``model_kwargs``, ``trust_remote_code``, ``backend``, etc.) and initializes the Module given the module's configuration from that directory or model name.
* A :meth:`~sentence_transformers.models.Module.get_sentence_embedding_dimension` method that returns the dimensionality of the sentence embeddings produced by the module. This is required if the module generates the embeddings or updates the embeddings' dimensionality.
* A :meth:`~sentence_transformers.models.InputModule.get_max_seq_length` method that returns the maximum sequence length the module can process. Only required if the module processes input text.

Subsequent Modules
^^^^^^^^^^^^^^^^^^

Subsequent modules in the pipeline are called non-input modules. They are responsible for processing the input features generated by the input module and generating the final sentence embeddings. Non-input modules can be any module that implements the :class:`~sentence_transformers.models.Module` class.

It has two abstract methods that you need to implement:

* A :meth:`~sentence_transformers.models.Module.forward` method that accepts a ``features`` dictionary with keys like ``input_ids``, ``attention_mask``, ``token_type_ids``, ``token_embeddings``, and ``sentence_embedding``, depending on where the module is in the model pipeline.
* A :meth:`~sentence_transformers.models.Module.save` method that saves the module's configuration and optionally weights to a provided directory.

Optionally, you can also implement the following methods:

* A :meth:`~sentence_transformers.models.Module.load` static method that accepts a ``model_name_or_path`` argument, keyword arguments for loading from Hugging Face (``subfolder``, ``token``, ``cache_folder``, etc.) and module kwargs (``model_kwargs``, ``trust_remote_code``, ``backend``, etc.) and initializes the Module given the module's configuration from that directory or model name.
* A :meth:`~sentence_transformers.models.Module.get_sentence_embedding_dimension` method that returns the dimensionality of the sentence embeddings produced by the module. This is required if the module generates the embeddings or updates the embeddings' dimensionality.

Example Module
^^^^^^^^^^^^^^

For example, we can create a custom pooling method by implementing a custom Module.

.. code-block:: python

   # decay_pooling.py

   import torch
   from sentence_transformers.models import Module


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

       def get_sentence_embedding_dimension(self) -> int:
           return self.dimension

       def save(self, output_path, *args, safe_serialization=True, **kwargs) -> None:
           self.save_config(output_path)

       # The `load` method by default loads the config.json file from the model directory
       # and initializes the class with the loaded parameters, i.e. the `config_keys`.
       # This works for us, so no need to override it.

.. note::

   Adding ``**kwargs`` to the ``__init__``, ``forward``, ``save``, ``load``, and ``tokenize`` methods is recommended to ensure that the methods remain compatible with future updates to the Sentence Transformers library.

This can now be used as a module in a Sentence Transformer model::
   
   from sentence_transformers import models, SentenceTransformer
   from decay_pooling import DecayMeanPooling

   transformer = models.Transformer("bert-base-uncased", max_seq_length=256)
   decay_mean_pooling = DecayMeanPooling(transformer.get_word_embedding_dimension(), decay=0.99)
   normalize = models.Normalize()

   model = SentenceTransformer(modules=[transformer, decay_mean_pooling, normalize])
   print(model)
   """
   SentenceTransformer(
       (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
       (1): DecayMeanPooling()
       (2): Normalize()
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

You can save this model with :meth:`SentenceTransformer.save_pretrained <sentence_transformers.SentenceTransformer.save_pretrained>`, resulting in a ``modules.json`` of::

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.models.Transformer"
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
       "type": "sentence_transformers.models.Normalize"
     }
   ]

To ensure that ``decay_pooling.DecayMeanPooling`` can be imported, you should copy over the ``decay_pooling.py`` file to the directory where you saved the model. If you push the model to the `Hugging Face Hub <https://huggingface.co/models>`_, then you should also upload the ``decay_pooling.py`` file to the model's repository. Then, everyone can use your custom module by calling :meth:`SentenceTransformer("your-username/your-model-id", trust_remote_code=True) <sentence_transformers.SentenceTransformer>`.

.. note::

   Using a custom module with remote code stored on the Hugging Face Hub requires that your users specify ``trust_remote_code`` as ``True`` when loading the model. This is a security measure to prevent remote code execution attacks.

If you have your models and custom modelling code on the Hugging Face Hub, then it might make sense to separate your custom modules into a separate repository. This way, you only have to maintain one implementation of your custom module, and you can reuse it across multiple models. You can do this by updating the ``type`` in ``modules.json`` file to include the path to the repository where the custom module is stored like ``{repository_id}--{dot_path_to_module}``. For example, if the ``decay_pooling.py`` file is stored in a repository called ``my-user/my-model-implementation`` and the module is called ``DecayMeanPooling``, then the ``modules.json`` file may look like this::

   [
     {
       "idx": 0,
       "name": "0",
       "path": "",
       "type": "sentence_transformers.models.Transformer"
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
       "type": "sentence_transformers.models.Normalize"
     }
   ]

Advanced: Keyword argument passthrough in Custom Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want your users to be able to specify custom keyword arguments via the :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>` method, then you can add their names to the ``modules.json`` file. For example, if my module should behave differently if your users specify a ``task`` keyword argument, then your ``modules.json`` might look like::

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
       "type": "sentence_transformers.models.Pooling"
     },
     {
       "idx": 2,
       "name": "2",
       "path": "2_Normalize",
       "type": "sentence_transformers.models.Normalize"
     }
   ]

Then, you can access the ``task`` keyword argument in the ``forward`` method of your custom module::

   from sentence_transformers.models import Transformer

   class CustomTransformer(Transformer):
       def forward(self, features: dict[str, torch.Tensor], task: Optional[str] = None, **kwargs) -> dict[str, torch.Tensor]:
           if task == "default":
               # Do something
           else:
               # Do something else
           return features

This way, users can specify the ``task`` keyword argument when calling :meth:`SentenceTransformer.encode <sentence_transformers.SentenceTransformer.encode>`::

   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer("your-username/your-model-id", trust_remote_code=True)
   texts = [...]
   model.encode(texts, task="default")
