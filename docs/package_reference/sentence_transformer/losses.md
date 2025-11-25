# Losses

`sentence_transformers.losses` defines different loss functions that can be used to fine-tune embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../sentence_transformer/loss_overview.md) to help narrow down your choice of loss function(s).

## BatchAllTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.BatchAllTripletLoss
```

## BatchHardSoftMarginTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.BatchHardSoftMarginTripletLoss
```

## BatchHardTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.BatchHardTripletLoss
```

## BatchSemiHardTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.BatchSemiHardTripletLoss
```

## ContrastiveLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.ContrastiveLoss
```

## OnlineContrastiveLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.OnlineContrastiveLoss
```

## ContrastiveTensionLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.ContrastiveTensionLoss
```

## ContrastiveTensionLossInBatchNegatives

```{eval-rst}
.. autoclass:: sentence_transformers.losses.ContrastiveTensionLossInBatchNegatives
```

## CoSENTLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.CoSENTLoss
```

## AnglELoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.AnglELoss
```

## CosineSimilarityLoss

<img src="https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/SBERT_Siamese_Network.png" alt="SBERT Siamese Network Architecture" width="250"/>

For each sentence pair, we pass sentence A and sentence B through our network which yields the embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score.

This allows our network to be fine-tuned to recognize the similarity of sentences.

```{eval-rst}
.. autoclass:: sentence_transformers.losses.CosineSimilarityLoss
```

## DenoisingAutoEncoderLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.DenoisingAutoEncoderLoss
```

## GISTEmbedLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.GISTEmbedLoss
```

## CachedGISTEmbedLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.CachedGISTEmbedLoss
```

## MSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MSELoss
```

## MarginMSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MarginMSELoss
```

## MatryoshkaLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MatryoshkaLoss
```

## Matryoshka2dLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.Matryoshka2dLoss
```

## AdaptiveLayerLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.AdaptiveLayerLoss
```

## MegaBatchMarginLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MegaBatchMarginLoss
```

## MultipleNegativesRankingLoss

*MultipleNegativesRankingLoss* is a great loss function if you only have positive pairs, for example, only pairs of similar texts like pairs of paraphrases, pairs of duplicate questions, pairs of (query, response), or pairs of (source_language, target_language).

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MultipleNegativesRankingLoss
```

## CachedMultipleNegativesRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.CachedMultipleNegativesRankingLoss
```

## MultipleNegativesSymmetricRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss
```

## CachedMultipleNegativesSymmetricRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss
```

## SoftmaxLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.SoftmaxLoss
```

## TripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.TripletLoss
```

## DistillKLDivLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.DistillKLDivLoss
```
