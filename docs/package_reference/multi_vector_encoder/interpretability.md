# Interpretability

`sentence_transformers.multi_vector_encoder.interpretability` provides a per-query-token MaxSim
heatmap utility for ColPali-style image documents. Useful for spot-checking which patch
positions in an image contribute most to a given query.

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.maxsim_heatmap
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.real_query_token_slice
```
