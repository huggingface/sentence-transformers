# Datasets

```{eval-rst}

.. note::
    The ``sentence_transformers.datasets`` classes have been deprecated, and only exist for compatibility with the `deprecated training <../../sentence_transformer/training_overview.html#deprecated-training>`_.

    * Instead of :class:`~sentence_transformers.datasets.SentenceLabelDataset`, you can now use ``BatchSamplers.GROUP_BY_LABEL`` to use the :class:`~sentence_transformers.sampler.GroupByLabelBatchSampler`.
    * Instead of :class:`~sentence_transformers.datasets.NoDuplicatesDataLoader`, you can now use the ``BatchSamplers.NO_DUPLICATES`` to use the :class:`~sentence_transformers.sampler.NoDuplicatesBatchSampler`.
```

`sentence_transformers.datasets` contains classes to organize your training input examples.

## ParallelSentencesDataset

`ParallelSentencesDataset` is used for multilingual training. For details, see [multilingual training](../../../examples/sentence_transformer/training/multilingual/README.md).

```{eval-rst}
.. autoclass:: sentence_transformers.datasets.ParallelSentencesDataset
```

## SentenceLabelDataset

`SentenceLabelDataset` can be used if you have labeled sentences and want to train with triplet loss.

```{eval-rst}
.. autoclass:: sentence_transformers.datasets.SentenceLabelDataset
```

## DenoisingAutoEncoderDataset

`DenoisingAutoEncoderDataset` is used for unsupervised training with the TSDAE method.

```{eval-rst}
.. autoclass:: sentence_transformers.datasets.DenoisingAutoEncoderDataset
```

## NoDuplicatesDataLoader

`NoDuplicatesDataLoader`can be used together with MultipleNegativeRankingLoss to ensure that no duplicates are within the same batch.

```{eval-rst}
.. autoclass:: sentence_transformers.datasets.NoDuplicatesDataLoader
```
