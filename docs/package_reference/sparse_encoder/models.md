# Modules

`sentence_transformers.sparse_encoder.models` defines different building blocks, that can be used to create SparseEncoder networks from scratch. For more details, see [Training Overview](../../sparse_encoder/training_overview.md).
Note that modules from `sentence_transformers.sentence_transformer.models` can also be used for Sparse models, such as `sentence_transformers.sentence_transformer.models.Transformer` from [SentenceTransformer > Modules](../sentence_transformer/models.md)
## SPLADE Pooling
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.SpladePooling
```

## MLM Transformer
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.MLMTransformer
```

## SparseAutoEncoder
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.SparseAutoEncoder
```

## SparseStaticEmbedding
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.models.SparseStaticEmbedding
``` 