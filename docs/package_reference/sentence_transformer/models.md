# Modules

`sentence_transformers.models` defines different building blocks, a.k.a. Modules, that can be used to create SentenceTransformer models from scratch. For more details, see [Creating Custom Models](../../sentence_transformer/usage/custom_models.rst).

## Main Modules

```{eval-rst}
.. autoclass:: sentence_transformers.models.Transformer
.. autoclass:: sentence_transformers.models.Pooling
.. autoclass:: sentence_transformers.models.Dense
.. autoclass:: sentence_transformers.models.Normalize
.. autoclass:: sentence_transformers.models.Router
    :members: for_query_document
.. autoclass:: sentence_transformers.models.StaticEmbedding
    :members: from_model2vec, from_distillation
```

## Further Modules

```{eval-rst}
.. autoclass:: sentence_transformers.models.BoW
.. autoclass:: sentence_transformers.models.CNN
.. autoclass:: sentence_transformers.models.LSTM
.. autoclass:: sentence_transformers.models.WeightedLayerPooling
.. autoclass:: sentence_transformers.models.WordEmbeddings
.. autoclass:: sentence_transformers.models.WordWeights
```

## Base Modules

```{eval-rst}
.. autoclass:: sentence_transformers.models.Module
    :members: config_file_name, config_keys, save_in_root, forward, get_config_dict, load, load_config, load_file_path, load_dir_path, load_torch_weights, save, save_config, save_torch_weights
.. autoclass:: sentence_transformers.models.InputModule
    :members: save_in_root, tokenizer, tokenize, save_tokenizer
```
