# XTR

[XTR (ConteXtualized Token Retriever)](https://huggingface.co/papers/2304.01982) is a late-interaction training objective that changes how token matches are scored: instead of ColBERT-style MaxSim over all document tokens, each query token retrieves its top-k matches globally across all in-batch document tokens, simulating retrieval from a token index. This trains the model to produce tokens that are retrievable on their own, allowing (in the original paper's setup) retrieval without the full MaxSim gathering stage.

**Training code: [training_contrastive.py](training_contrastive.py)**

```{eval-rst}
In Sentence Transformers, XTR is a drop-in scoring metric rather than a separate loss: pass :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` as the ``score_metric`` of :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss` (or its cached variant) to switch from ColBERT-style MaxSim to XTR-style global top-k scoring without changing anything else about the training setup::

    from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss
    from sentence_transformers.multi_vector_encoder.scoring import XTRScores

    loss = MultiVectorMultipleNegativesRankingLoss(model=model, score_metric=XTRScores(k=256))

Note that XTR scoring is set-dependent: the top-k is taken across the whole candidate set, so a (query, document) pair has no standalone score. It can therefore not be set as the model's ``similarity_fn_name``, and the evaluators reject it: evaluation and inference score with MaxSim via ``model.similarity``, also for XTR-trained models. To compute XTR scores ad hoc over a fixed candidate set, call :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores` directly.
```
