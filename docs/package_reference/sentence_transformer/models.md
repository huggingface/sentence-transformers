# Modules

`sentence_transformers.sentence_transformer.models` defines different building blocks, a.k.a. Modules, that can be used to create SentenceTransformer models from scratch. For more details, see [Creating Custom Models](../../sentence_transformer/usage/custom_models.rst).

See also the modules from `sentence_transformers.base.models` in [Base > Modules](../base/models.rst).

## Main Modules

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.models.Pooling
.. autoclass:: sentence_transformers.sentence_transformer.models.Dense
.. autoclass:: sentence_transformers.sentence_transformer.models.Normalize
.. autoclass:: sentence_transformers.sentence_transformer.models.StaticEmbedding
    :members: from_model2vec, from_distillation
```

## Further Modules

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.models.BoW
.. autoclass:: sentence_transformers.sentence_transformer.models.CNN
.. autoclass:: sentence_transformers.sentence_transformer.models.LSTM
.. autoclass:: sentence_transformers.sentence_transformer.models.WeightedLayerPooling
.. autoclass:: sentence_transformers.sentence_transformer.models.WordEmbeddings
.. autoclass:: sentence_transformers.sentence_transformer.models.WordWeights
```
