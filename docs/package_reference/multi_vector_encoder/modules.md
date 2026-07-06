# Modules

`sentence_transformers.multi_vector_encoder.modules` defines the building blocks specific to
multi-vector models. Combined with the shared backbone in `sentence_transformers.base.modules`
(see [Base > Modules](../base/modules.rst)), they make up the standard ColBERT-style stack:
``Transformer`` -> ``Dense`` -> ``MultiVectorMask`` -> ``Normalize``.

See also [Training Overview](../../multi_vector_encoder/training_overview.md).

## MultiVectorMask
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.modules.MultiVectorMask
```

## HierarchicalPooling
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.modules.HierarchicalPooling
```
