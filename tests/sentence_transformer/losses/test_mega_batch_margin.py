from __future__ import annotations

import torch
from transformers import set_seed

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MegaBatchMarginLoss


def test_mega_batch_margin_all_minibatches_contribute(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Every mini-batch must contribute to the gradient, not only the last one.

    ``forward_mini_batched`` back-propagates each non-last mini-batch inside the loop and leaves the
    last mini-batch to the outer training loop. The mini-batched and non-mini-batched paths select the
    same hard negatives and use the same per-triplet hinge loss, differing only in the reduction: the
    mini-batched path averages over ``mini_batch_size`` samples per mini-batch and sums the resulting
    gradients, while the non-mini-batched path averages over the whole batch. Hence the mini-batched
    gradient must equal ``num_mini_batches`` times the non-mini-batched gradient.

    Before the fix, ``forward_mini_batched`` did not even run: it sliced the non-tensor ``modality``
    marker (``"text"`` -> ``"te"``) and raised ``KeyError``; and once that was fixed, the in-loop guard
    ``end_idx < len(cos_scores)`` was always False (``cos_scores`` only has ``mini_batch_size`` rows),
    so only the last mini-batch contributed and this invariant broke.
    """
    model = stsb_bert_tiny_model.to("cpu")
    # Disable dropout so the eval-mode hard-negative selection and the train-mode differentiable pass
    # produce identical embeddings, making both loss variants deterministic and directly comparable.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    anchors = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f"]
    positives = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f"]
    batch_size, mini_batch_size = len(anchors), 2
    num_mini_batches = batch_size // mini_batch_size
    labels = torch.zeros(batch_size, dtype=torch.long)

    def grad_for(use_mini_batched_version: bool, mbs: int) -> dict[str, torch.Tensor]:
        set_seed(42)
        model.zero_grad()
        loss = MegaBatchMarginLoss(model, use_mini_batched_version=use_mini_batched_version, mini_batch_size=mbs)
        features = [model.preprocess(anchors), model.preprocess(positives)]
        loss(features, labels).backward()
        return {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}

    grad_mini = grad_for(use_mini_batched_version=True, mbs=mini_batch_size)
    grad_non_mini = grad_for(use_mini_batched_version=False, mbs=batch_size)

    assert grad_mini and grad_non_mini
    assert sum(g.abs().sum() for g in grad_mini.values()) > 0
    for name, grad in grad_mini.items():
        expected = num_mini_batches * grad_non_mini[name]
        assert torch.allclose(grad, expected, rtol=1e-3, atol=1e-4), name
