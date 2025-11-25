# Model Distillation

This page contains an example of knowledge distillation for SparseEncoder models. Knowledge distillation is essential for training the strongest sparse models, as the most effective sparse encoders are trained partially or fully with distillation from powerful teacher models.

Knowledge distillation allows us to compress knowledge from larger, more computationally expensive models (teacher models) into smaller, more efficient sparse models (student models). This approach can leverage bigger model results, including non-sparse models like Cross-Encoders and dense bi-encoders, to compress the knowledge into our small sparse model while maintaining much of the original performance.

## MarginMSE

**Training code: [train_splade_msmarco_margin_mse.py](train_splade_msmarco_margin_mse.py)**

```{eval-rst}
:class:`~sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss` is based on the paper of `Hofstätter et al <https://huggingface.co/papers/2010.02666>`_. Like when training with :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`, we can use triplets: ``(query, passage1, passage2)``. However, in contrast to :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`, we use a similarity score for ``passage1`` and ``passage2``, so these do not have to be strictly positive/negative, both can be relevant or not relevant for a given query.

The distillation process works by transferring knowledge from a powerful teacher model (like a Cross-Encoder ensemble) to our efficient sparse encoder student model. We compute the `Cross-Encoder <../../../cross_encoder/applications/README.html>`_ score for ``(query, passage1)`` and ``(query, passage2)`` using the teacher model. We provide scores for 160 million such pairs in our `msmarco-hard-negatives dataset <https://huggingface.co/datasets/sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2>`_, which contains pre-computed scores from a BERT ensemble Cross-Encoder. We then compute the distance: ``CE_distance = CEScore(query, passage1) - CEScore(query, passage2)``.

For our SparseEncoder (here a Splade model) student training, we encode ``query``, ``passage1``, and ``passage2`` into embeddings and then measure the dot-product between  ``(query, passage1)`` and ``(query, passage2)``. Again, we measure the distance: ``SE_distance = DotScore(query, passage1) - DotScore(query, passage2)``

The knowledge transfer happens by ensuring that the distance predicted by the Splade model matches the distance predicted by the teacher cross-encoder, i.e., we optimize the mean-squared error (MSE) between ``CE_distance`` and ``SE_distance``. This allows the sparse model to learn the sophisticated ranking behavior of the much larger teacher model.

```
