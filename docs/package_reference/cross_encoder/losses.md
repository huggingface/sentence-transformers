# Losses

`sentence_transformers.cross_encoder.losses` defines different loss functions that can be used to fine-tune cross-encoder models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../cross_encoder/loss_overview.md) to help narrow down your choice of loss function(s).

## BinaryCrossEntropyLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss
```

## CrossEntropyLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.CrossEntropyLoss
```

## LambdaLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.LambdaLoss

.. autoclass:: sentence_transformers.cross_encoder.losses.LambdaLoss.BaseWeightingScheme
.. autoclass:: sentence_transformers.cross_encoder.losses.NoWeightingScheme
.. autoclass:: sentence_transformers.cross_encoder.losses.NDCGLoss1Scheme
.. autoclass:: sentence_transformers.cross_encoder.losses.NDCGLoss2Scheme
.. autoclass:: sentence_transformers.cross_encoder.losses.LambdaRankScheme
.. autoclass:: sentence_transformers.cross_encoder.losses.NDCGLoss2PPScheme
```

## ListMLELoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.ListMLELoss
```

## PListMLELoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.PListMLELoss

.. autoclass:: sentence_transformers.cross_encoder.losses.PListMLELambdaWeight
```

## ListNetLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.ListNetLoss
```

## MultipleNegativesRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss
```

## CachedMultipleNegativesRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss
```

## MSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.MSELoss
```

## MarginMSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.MarginMSELoss
```

## RankNetLoss

```{eval-rst}
.. autoclass:: sentence_transformers.cross_encoder.losses.RankNetLoss
```
