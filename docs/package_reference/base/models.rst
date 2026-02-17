Modules
=======

`sentence_transformers.base.models` defines different building blocks, a.k.a. Modules, that can be used to create models from scratch.

Common Modules
--------------

.. autoclass:: sentence_transformers.base.models.Transformer

.. autoclass:: sentence_transformers.base.models.Router
    :members: for_query_document

Base Modules
------------

.. autoclass:: sentence_transformers.base.models.Module
    :members: config_file_name, config_keys, save_in_root, forward, get_config_dict, load, load_config, load_file_path, load_dir_path, load_torch_weights, save, save_config, save_torch_weights

.. autoclass:: sentence_transformers.base.models.InputModule
    :members: save_in_root, tokenizer, tokenize, save_tokenizer
