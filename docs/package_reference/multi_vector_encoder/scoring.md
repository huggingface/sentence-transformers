# Scoring

`sentence_transformers.multi_vector_encoder.scoring` provides the late-interaction scoring functions
used by training losses. Pass one of these (or a configured callable) as the ``similarity_fct``
parameter on the multi-vector losses to switch between ColBERT-style MaxSim and XTR-style global
top-k scoring.

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
