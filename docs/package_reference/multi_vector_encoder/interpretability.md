# Interpretability

`sentence_transformers.multi_vector_encoder.interpretability` provides a per-query-token MaxSim
heatmap utility for ColPali-style image documents. Useful for spot-checking which patch
positions in an image contribute most to a given query.

## Heatmaps

`maxsim_heatmap` is the one-shot entry point. `get_n_patches` supplies its ``n_patches`` argument,
and `real_query_token_slice` selects the query tokens worth visualizing.

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.maxsim_heatmap
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.get_n_patches
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.real_query_token_slice
```

## Building blocks

`maxsim_heatmap` composes these two. Use them directly to reach the raw similarity tensor, e.g. for a
custom colormap, a matplotlib figure, or an aggregation other than the built-in ones.

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.maxsim_similarity_map
```

```{eval-rst}
.. autofunction:: sentence_transformers.multi_vector_encoder.interpretability.render_similarity_map_on_image
```
