# Loss Overview

```{eval-rst}
All multi-vector losses embed both sides into token-level embeddings and score them with late
interaction inside the loss. The scoring is swappable: the ranking and KD losses accept a
``score_metric=`` kwarg that defaults to ColBERT-style MaxSim
(:func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores`), and can be switched to
XTR-style global top-k scoring by passing
:class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` (or
:class:`~sentence_transformers.multi_vector_encoder.scoring.XTRKDScores` for the KD loss).
:class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMarginMSELoss` is the
exception: it scores matched pairs with a pairwise ``similarity_fct`` (default
:func:`~sentence_transformers.util.maxsim_pairwise`) instead.
```

## Loss Table

Loss functions play a critical role in the performance of your fine-tuned model. Sadly, there is no "one size fits all" loss function. Ideally, this table should help narrow down your choice of loss function(s) by matching them to your data formats.

```{eval-rst}
.. note::

    You can often convert one training data format into another, allowing more loss functions to be viable for your scenario. For example, ``(anchor, positive) pairs`` can be extended into ``(anchor, positive, negative) triplets`` by mining hard negatives with a first-stage retriever.
```

**Legend:** Loss functions marked with `★` are commonly recommended default choices.

| Inputs                                            | Labels | Appropriate Loss Functions                                                                                                                                                                                                                                                                                                                                       |
|---------------------------------------------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(anchor, positive) pairs`                        | `none` | <a href="../package_reference/multi_vector_encoder/losses.html#multivectormultiplenegativesrankingloss">`MultiVectorMultipleNegativesRankingLoss`</a> ★                                                                                                                                                                                                          |
| `(anchor, positive, negative) triplets`           | `none` | <a href="../package_reference/multi_vector_encoder/losses.html#multivectormultiplenegativesrankingloss">`MultiVectorMultipleNegativesRankingLoss`</a> ★                                                                                                                                                                                                          |
| `(anchor, positive, negative_1, ..., negative_n)` | `none` | <a href="../package_reference/multi_vector_encoder/losses.html#multivectormultiplenegativesrankingloss">`MultiVectorMultipleNegativesRankingLoss`</a> ★                                                                                                                                                                                                          |

```{eval-rst}
:class:`~sentence_transformers.multi_vector_encoder.losses.CachedMultiVectorMultipleNegativesRankingLoss`
is a drop-in replacement for
:class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss`
adopting `GradCache <https://huggingface.co/papers/2101.06983>`_: it computes and caches the embedding
gradients in mini-batches, allowing much larger effective batch sizes without additional GPU memory.
In-batch negatives losses benefit heavily from larger batch sizes, as they yield more negatives and a
stronger training signal.
```

## Distillation

These loss functions are specifically designed to be used when distilling the knowledge from one model into another. Distillation from a strong cross-encoder teacher is how the strongest late-interaction models (e.g. ColBERTv2, GTE-ModernColBERT) are trained.

| Inputs                                           | Labels                                                                    | Appropriate Loss Functions                                                                                                                              |
|--------------------------------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(query, positive, negative)`                    | `gold_sim(query, positive) - gold_sim(query, negative)`                   | <a href="../package_reference/multi_vector_encoder/losses.html#multivectormarginmseloss">`MultiVectorMarginMSELoss`</a>                                 |
| `(query, positive, negative_1, ..., negative_n)` | `[gold_sim(query, positive) - gold_sim(query, negative_i) for i in 1..n]` | <a href="../package_reference/multi_vector_encoder/losses.html#multivectormarginmseloss">`MultiVectorMarginMSELoss`</a>                                 |
| `(query, [doc_1, ..., doc_n])`                   | `[gold_sim(query, doc_i) for i in 1..n]`                                  | <a href="../package_reference/multi_vector_encoder/losses.html#multivectordistillkldivloss">`MultiVectorDistillKLDivLoss`</a> ★                         |

```{eval-rst}
For the n-way KD format with externally-stored query / document texts (e.g.
`lightonai/ms-marco-en-bge <https://huggingface.co/datasets/lightonai/ms-marco-en-bge>`_), use
:class:`~sentence_transformers.multi_vector_encoder.KDProcessing` to resolve IDs against the query and
document datasets on the fly.
```

## Commonly used Loss Functions

In practice, not all loss functions get used equally often. The most common scenarios are:

* `(anchor, positive) pairs` without any labels: <a href="../package_reference/multi_vector_encoder/losses.html#multivectormultiplenegativesrankingloss"><code>MultiVectorMultipleNegativesRankingLoss</code></a> (a.k.a. InfoNCE or in-batch negatives loss) is cheap to obtain data for and generally very performant, especially combined with mined hard negatives. <a href="../package_reference/multi_vector_encoder/losses.html#cachedmultivectormultiplenegativesrankingloss"><code>CachedMultiVectorMultipleNegativesRankingLoss</code></a> extends it to very large effective batch sizes.
* `(query, [doc_1, ..., doc_n])` with teacher scores: <a href="../package_reference/multi_vector_encoder/losses.html#multivectordistillkldivloss"><code>MultiVectorDistillKLDivLoss</code></a> implements the n-way knowledge distillation recipe behind the strongest late-interaction models.

## Custom Loss Functions

```{eval-rst}
Advanced users can create and train with their own loss functions. Custom loss functions only have a few requirements:

- They must be a subclass of :class:`torch.nn.Module`.
- They must have ``model`` as the first argument in the constructor.
- They must implement a ``forward`` method that accepts ``sentence_features`` and ``labels``. The former is a list of tokenized batches, one element for each column. These tokenized batches can be fed directly to the ``model`` being trained to produce the token-level embeddings (read ``token_embeddings`` and the scoring ``attention_mask`` from the model output). The latter is an optional tensor of labels. The method must return a single loss value or a dictionary of loss components (component names to loss values) that will be summed to produce the final loss value. When returning a dictionary, the individual components will be logged separately in addition to the summed loss, allowing you to monitor the individual components of the loss.

To get full support with the automatic model card generation, you may also wish to implement:

- a ``get_config_dict`` method that returns a dictionary of loss parameters.
- a ``citation`` property so your work gets cited in all models that train with the loss.

Consider inspecting existing loss functions to get a feel for how loss functions are commonly implemented.
```
