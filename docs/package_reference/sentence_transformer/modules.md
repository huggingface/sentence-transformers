# Modules

`sentence_transformers.sentence_transformer.modules` defines different building blocks, a.k.a. Modules, that can be used to create SentenceTransformer models from scratch. For more details, see [Creating Custom Models](../../sentence_transformer/usage/custom_models.rst).

See also the modules from `sentence_transformers.base.modules` in [Base > Modules](../base/modules.rst).

## Main Modules

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.modules.Pooling
.. autoclass:: sentence_transformers.sentence_transformer.modules.Dense
.. autoclass:: sentence_transformers.sentence_transformer.modules.Normalize
.. autoclass:: sentence_transformers.sentence_transformer.modules.StaticEmbedding
    :members: from_model2vec, from_distillation
```

## Further Modules

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.modules.BoW
.. autoclass:: sentence_transformers.sentence_transformer.modules.CNN
.. autoclass:: sentence_transformers.sentence_transformer.modules.LSTM
.. autoclass:: sentence_transformers.sentence_transformer.modules.WeightedLayerPooling
.. autoclass:: sentence_transformers.sentence_transformer.modules.WordEmbeddings
.. autoclass:: sentence_transformers.sentence_transformer.modules.WordWeights
```
