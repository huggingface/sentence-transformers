# Evaluation

`sentence_transformers.multi_vector_encoder.evaluation` defines evaluators tailored to multi-vector
(late-interaction) models. All evaluators score with MaxSim and accept the multi-vector model's
ragged per-token embeddings.

## MultiVectorInformationRetrievalEvaluator

```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.evaluation.MultiVectorInformationRetrievalEvaluator
```

## MultiVectorNanoBEIREvaluator

```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.evaluation.MultiVectorNanoBEIREvaluator
```

## MultiVectorTripletEvaluator

```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.evaluation.MultiVectorTripletEvaluator
```

## MultiVectorRerankingEvaluator

```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.evaluation.MultiVectorRerankingEvaluator
```

## MultiVectorDistillationEvaluator

```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.evaluation.MultiVectorDistillationEvaluator
```
