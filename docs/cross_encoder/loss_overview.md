# Loss Overview

## Loss Table

Loss functions play a critical role in the performance of your fine-tuned Cross Encoder model. Sadly, there is no "one size fits all" loss function. Ideally, this table should help narrow down your choice of loss function(s) by matching them to your data formats.

```{eval-rst}
.. note:: 

    You can often convert one training data format into another, allowing more loss functions to be viable for your scenario. For example, ``(sentence_A, sentence_B) pairs`` with ``class`` labels can be converted into ``(anchor, positive, negative) triplets`` by sampling sentences with the same or different classes.

    Additionally, :func:`~sentence_transformers.util.mine_hard_negatives` can easily be used to turn ``(anchor, positive)`` to:

    - ``(anchor, positive, negative) triplets`` with ``output_format="triplet"``, 
    - ``(anchor, positive, negative_1, …, negative_n) tuples`` with ``output_format="n-tuple"``.
    - ``(anchor, passage, label) labeled pairs`` with a label of 0 for negative and 1 for positive with ``output_format="labeled-pair"``,
    - ``(anchor, [doc1, doc2, ..., docN], [label1, label2, ..., labelN]) triplets`` with labels of 0 for negative and 1 for positive with ``output_format="labeled-list"``, 
```

| Inputs                                            | Labels                                   | Number of Model Output Labels | Appropriate Loss Functions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|---------------------------------------------------|------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `(sentence_A, sentence_B) pairs`                  | `class`                                  | `num_classes`                 | <a href="../package_reference/cross_encoder/losses.html#crossentropyloss">`CrossEntropyLoss`</a>                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `(anchor, positive) pairs`                        | `none`                                   | `1`                           | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/cross_encoder/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a>                                                                                                                                                                                                                                                                                    |
| `(anchor, positive/negative) pairs`               | `1 if positive, 0 if negative`           | `1`                           | <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss">`BinaryCrossEntropyLoss`</a>                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `(sentence_A, sentence_B) pairs`                  | `float similarity score between 0 and 1` | `1`                           | <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss">`BinaryCrossEntropyLoss`</a>                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `(anchor, positive, negative) triplets`           | `none`                                   | `1`                           | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/cross_encoder/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a>                                                                                                                                                                                                                                                                                    |
| `(anchor, positive, negative_1, ..., negative_n)` | `none`                                   | `1`                           | <a href="../package_reference/cross_encoder/losses.html#multiplenegativesrankingloss">`MultipleNegativesRankingLoss`</a><br><a href="../package_reference/cross_encoder/losses.html#cachedmultiplenegativesrankingloss">`CachedMultipleNegativesRankingLoss`</a>                                                                                                                                                                                                                                                                                    |
| `(query, [doc1, doc2, ..., docN])`                | `[score1, score2, ..., scoreN]`          | `1`                           | <ol style="margin-bottom: 0;line-height: inherit;"><li><a href="../package_reference/cross_encoder/losses.html#lambdaloss">`LambdaLoss`</a></li><li><a href="../package_reference/cross_encoder/losses.html#plistmleloss">`PListMLELoss`</a></li><li><a href="../package_reference/cross_encoder/losses.html#listnetloss">`ListNetLoss`</a></li><li><a href="../package_reference/cross_encoder/losses.html#ranknetloss">`RankNetLoss`</a></li><li><a href="../package_reference/cross_encoder/losses.html#listmleloss">`ListMLELoss`</a></li></ol> |

## Distillation

These loss functions are specifically designed to be used when distilling the knowledge from one model into another.
For example, when finetuning a small model to behave more like a larger & stronger one, or when finetuning a model to become multi-lingual.

| Texts                                             | Labels                                                                    | Appropriate Loss Functions                                                                 |
|---------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `(sentence_A, sentence_B) pairs`                  | `similarity score`                                                        | <a href="../package_reference/cross_encoder/losses.html#mseloss">`MSELoss`</a>             |
| `(query, passage_one, passage_two) triplets`      | `gold_sim(query, passage_one) - gold_sim(query, passage_two)`             | <a href="../package_reference/cross_encoder/losses.html#marginmseloss">`MarginMSELoss`</a> |
| `(query, positive, negative_1, ..., negative_n)`  | `[gold_sim(query, positive) - gold_sim(query, negative_i) for i in 1..n]` | <a href="../package_reference/cross_encoder/losses.html#marginmseloss">`MarginMSELoss`</a> |
| `(query, positive, negative)`                     | `[gold_sim(query, positive), gold_sim(query, negative)]`                  | <a href="../package_reference/cross_encoder/losses.html#marginmseloss">`MarginMSELoss`</a> |
| `(query, positive, negative_1, ..., negative_n) ` | `[gold_sim(query, positive), gold_sim(query, negative_i)...] `            | <a href="../package_reference/cross_encoder/losses.html#marginmseloss">`MarginMSELoss`</a> |

## Commonly used Loss Functions

In practice, not all loss functions get used equally often. The most common scenarios are:

- `(sentence_A, sentence_B) pairs` with `float similarity score` or `1 if positive, 0 if negative`: <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss"><code>BinaryCrossEntropyLoss</code></a> is a traditional option that remains very challenging to outperform.
- `(anchor, positive) pairs` without any labels: combined with <a href="../package_reference/util.html#sentence_transformers.util.mine_hard_negatives"><code>mine_hard_negatives</code></a>
  - with <code>output_format="labeled-list"</code>, then <a href="../package_reference/cross_encoder/losses.html#lambdaloss"><code>LambdaLoss</code></a> is frequently used for learning-to-rank tasks.
  - with <code>output_format="labeled-pair"</code>, then <a href="../package_reference/cross_encoder/losses.html#binarycrossentropyloss"><code>BinaryCrossEntropyLoss</code></a> remains a strong option.

## Custom Loss Functions

```{eval-rst}
Advanced users can create and train with their own loss functions. Custom loss functions only have a few requirements:

- They must be a subclass of :class:`torch.nn.Module`.
- They must have ``model`` as the first argument in the constructor.
- They must implement a ``forward`` method that accepts ``inputs`` and ``labels``. The former is a nested list of texts in the batch, with each element in the outer list representing a column in the training dataset. You have to combine these texts into pairs that can be 1) tokenized and 2) fed to the model. The latter is an optional (list of) tensor(s) of labels from a ``label``, ``labels``, ``score``, or ``scores`` column in the dataset. The method must return a single loss value or a dictionary of loss components (component names to loss values) that will be summed to produce the final loss value. When returning a dictionary, the individual components will be logged separately in addition to the summed loss, allowing you to monitor the individual components of the loss.

To get full support with the automatic model card generation, you may also wish to implement:

- a ``get_config_dict`` method that returns a dictionary of loss parameters.
- a ``citation`` property so your work gets cited in all models that train with the loss.

Consider inspecting existing loss functions to get a feel for how loss functions are commonly implemented.
```
