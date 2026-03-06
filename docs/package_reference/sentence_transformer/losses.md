# Losses

`sentence_transformers.sentence_transformer.losses` defines different loss functions that can be used to fine-tune embedding models on training data. The choice of loss function plays a critical role when fine-tuning the model. It determines how well our embedding model will work for the specific downstream task.

Sadly, there is no "one size fits all" loss function. Which loss function is suitable depends on the available training data and on the target task. Consider checking out the [Loss Overview](../../sentence_transformer/loss_overview.md) to help narrow down your choice of loss function(s).

## BatchAllTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.BatchAllTripletLoss
```

## BatchHardSoftMarginTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.BatchHardSoftMarginTripletLoss
```

## BatchHardTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.BatchHardTripletLoss
```

## BatchSemiHardTripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.BatchSemiHardTripletLoss
```

## ContrastiveLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.ContrastiveLoss
```

## OnlineContrastiveLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.OnlineContrastiveLoss
```

## ContrastiveTensionLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.ContrastiveTensionLoss
```

## ContrastiveTensionLossInBatchNegatives

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.ContrastiveTensionLossInBatchNegatives
```

## CoSENTLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.CoSENTLoss
```

## AnglELoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.AnglELoss
```

## CosineSimilarityLoss

<img src="https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/SBERT_Siamese_Network.png" alt="SBERT Siamese Network Architecture" width="250"/>

For each sentence pair, we pass sentence A and sentence B through our network which yields the embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score.

This allows our network to be fine-tuned to recognize the similarity of sentences.

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.CosineSimilarityLoss
```

## DenoisingAutoEncoderLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.DenoisingAutoEncoderLoss
```

## GISTEmbedLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.GISTEmbedLoss
```

## CachedGISTEmbedLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.CachedGISTEmbedLoss
```

## GlobalOrthogonalRegularizationLoss

```{eval-rst}
.. autoclass:: sentence_transformers.losses.GlobalOrthogonalRegularizationLoss
```

## MSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MSELoss
```

## MarginMSELoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MarginMSELoss
```

## MatryoshkaLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MatryoshkaLoss
```

## Matryoshka2dLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.Matryoshka2dLoss
```

## AdaptiveLayerLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.AdaptiveLayerLoss
```

## MegaBatchMarginLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MegaBatchMarginLoss
```

## MultipleNegativesRankingLoss

*MultipleNegativesRankingLoss* is a great loss function if you only have positive pairs, for example, only pairs of similar texts like pairs of paraphrases, pairs of duplicate questions, pairs of (query, response), or pairs of (source_language, target_language).

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss
```

## CachedMultipleNegativesRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss
```

## MultipleNegativesSymmetricRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.MultipleNegativesSymmetricRankingLoss
```

## CachedMultipleNegativesSymmetricRankingLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesSymmetricRankingLoss
```

## SoftmaxLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.SoftmaxLoss
```

## TripletLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.TripletLoss
```

## DistillKLDivLoss

```{eval-rst}
.. autoclass:: sentence_transformers.sentence_transformer.losses.DistillKLDivLoss
```
