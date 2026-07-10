# MS MARCO

[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query. This page shows how to train Multi-Vector Encoder (ColBERT-style) models on this dataset so that they can be used for searching text passages given queries (key words, phrases or questions).

There are pre-trained models available, which you can directly use without the need of training your own models. For more information, see: [Pretrained Models](../../../../docs/multi_vector_encoder/pretrained_models.md).

## MultiVectorMultipleNegativesRankingLoss

**Training code: [training_contrastive.py](training_contrastive.py)**

```{eval-rst}
When we use :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss`, we provide triplets: ``(query, positive_passage, negative_passage)`` where ``positive_passage`` is the relevant passage to the query and ``negative_passage`` is a non-relevant passage, mined with BM25 in the `sentence-transformers/msmarco-bm25 <https://huggingface.co/datasets/sentence-transformers/msmarco-bm25>`_ dataset. Every query token is compared against every passage token, and the resulting MaxSim score of the ``(query, positive_passage)`` pair is optimized to be higher than the scores against the negative passage and against all other in-batch passages.

Losses that use in-batch negatives benefit heavily from larger batch sizes. If GPU memory is the bottleneck, **training code:** `training_cached_contrastive.py <training_cached_contrastive.py>`_ demonstrates the same recipe with :class:`~sentence_transformers.multi_vector_encoder.losses.CachedMultiVectorMultipleNegativesRankingLoss`, which reaches much larger effective batch sizes at a small speed cost via `GradCache <https://huggingface.co/papers/2101.06983>`_.
```

## MultiVectorDistillKLDivLoss

**Training code: [training_kd.py](training_kd.py)**

```{eval-rst}
The strongest late-interaction models (e.g. ColBERTv2, GTE-ModernColBERT) are trained with **knowledge distillation**: instead of binary relevance, the model learns to reproduce the score distribution of a stronger teacher model over N candidate documents per query. The `lightonai/ms-marco-en-bge <https://huggingface.co/datasets/lightonai/ms-marco-en-bge>`_ dataset provides per-query candidate document IDs together with teacher scores from a BGE model; :class:`~sentence_transformers.multi_vector_encoder.KDProcessing` resolves the IDs against the query and document texts on the fly, and :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss` minimizes the KL divergence between the (softmaxed) teacher scores and the student's MaxSim scores.
```
