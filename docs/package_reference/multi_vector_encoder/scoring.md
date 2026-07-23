# Scoring

`sentence_transformers.multi_vector_encoder.scoring` provides the late-interaction scoring functions
used by training losses. Pass one of these (or a configured callable) as the ``score_metric``
parameter on the multi-vector losses to switch between ColBERT-style MaxSim and XTR-style global
top-k scoring.

## ColBERT scoring
```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.colbert_scores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.colbert_kd_scores
```

## XTRScores
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.scoring.XTRScores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.xtr_scores
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.scoring.xtr_kd_scores
```

## XTRKDScores
```{eval-rst}
.. autoclass:: sentence_transformers.multi_vector_encoder.scoring.XTRKDScores
```
