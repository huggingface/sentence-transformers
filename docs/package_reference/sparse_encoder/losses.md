# Losses

`sentence_transformers.sparse_encoder.losses` defines different loss functions that can be used to fine-tune saprse embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../sparse_encoder/loss_overview.md) to help narrow down your choice of loss function(s).

```{eval-rst}
.. warning:: 
    To train a :class:`~sentence_transformers.sparse_encoder.SparseEncoder`, you need either :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` or :class:`~sentence_transformers.sparse_encoder.losses.CSRLoss`, depending on the architecture. These are wrapper losses that add sparsity regularization on top of a main loss function, which must be provided as a parameter. The only loss that can be used independently is :class:`~sentence_transformers.sparse_encoder.losses.SparseMSELoss`, as it performs embedding-level distillation, ensuring sparsity by directly copying the teacher's sparse embedding.

```

## SpladeLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SpladeLoss
```

## CachedSpladeLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CachedSpladeLoss
```

## FlopsLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.FlopsLoss
```

## CSRLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRLoss
```

## CSRReconstructionLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.CSRReconstructionLoss
```

## SparseMultipleNegativesRankingLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss
```

## SparseMarginMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss
```

## SparseDistillKLDivLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss
``` 

## SparseTripletLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseTripletLoss
```

## SparseCosineSimilarityLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss
```

## SparseCoSENTLoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss
```

## SparseAnglELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseAnglELoss
```

## SparseMSELoss
```{eval-rst}
.. autoclass:: sentence_transformers.sparse_encoder.losses.SparseMSELoss
```