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

    # The two paths are mathematically equivalent up to the reduction, so the mini-batched
    # gradient must equal ``num_mini_batches`` times the whole-batch gradient. They are not
    # bit-identical: the mini-batched path re-embeds each hard negative with a fresh forward pass
    # while the non-mini path reuses the positive embeddings, so word-embedding gradients differ by
    # floating-point recompute noise whose size varies across transformers versions. Compare the
    # relative norm, which is robust to that per-element noise, instead of an element-wise tolerance.
    flat_mini = torch.cat([grad_mini[name].flatten() for name in sorted(grad_mini)])
    flat_expected = num_mini_batches * torch.cat([grad_non_mini[name].flatten() for name in sorted(grad_mini)])
    relative_error = (flat_mini - flat_expected).norm() / flat_expected.norm()
    assert relative_error < 1e-3, relative_error
