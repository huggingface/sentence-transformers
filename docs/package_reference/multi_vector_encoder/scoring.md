# Scoring

`sentence_transformers.multi_vector_encoder.scoring` provides late-interaction scoring functions
for the `similarity_fct` parameter of multi-vector losses. See
[Kernel acceleration](../../multi_vector_encoder/usage/kernels.rst) for kernelized model scoring.

## ColBERT scoring
```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.colbert_scores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.colbert_scores_pairwise
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.colbert_kd_scores
```

## MeanMaxSim scoring

MaxSim divided by each query's real token count, so scores are comparable across query lengths.
Rankings within a query are unchanged. Pair these with ``model.similarity_fn_name = "meanmaxsim"``
so evaluation and the model card score the way training did.

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.mean_colbert_scores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.mean_colbert_scores_pairwise
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.mean_colbert_kd_scores
```

## XTRScores
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.scoring.XTRScores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.xtr_scores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.xtr_scores_pairwise
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.xtr_kd_scores
```

## XTRKDScores
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.scoring.XTRKDScores
```
