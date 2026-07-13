from __future__ import annotations

import torch

from sentence_transformers import SentenceTransformer


def disable_dropout(model: SentenceTransformer) -> None:
    """Disable every dropout, so that gradient-equivalence tests are deterministic.

    transformers<5 keeps the SDPA attention dropout as a plain float attribute rather than an ``nn.Dropout``
    module, so zeroing only the modules would leave attention dropout active there.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        for attribute in ("dropout_prob", "attention_dropout", "attention_probs_dropout_prob"):
            if isinstance(getattr(module, attribute, None), float):
                setattr(module, attribute, 0.0)

    # If a future transformers release keeps the rate somewhere this doesn't look, the equivalence tests
    # would start failing as a confusing "gradients do not match", pointing at the loss rather than here.
    was_training = model.training
    model.train()
    features = model.preprocess(["a canary sentence", "another canary sentence"])
    with torch.no_grad():
        first, second = model(features)["sentence_embedding"], model(features)["sentence_embedding"]
    model.train(was_training)
    assert torch.equal(first, second), "dropout is still active in training mode; disable_dropout needs updating"


def gradients(model: SentenceTransformer) -> dict[str, torch.Tensor]:
    return {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}


def assert_trained(grads: dict[str, torch.Tensor]) -> None:
    """Guard against a vacuous comparison: an empty or all-zero gradient dict passes any assert_close loop."""
    assert grads, "no parameter received a gradient at all"
    assert sum(grad.abs().sum() for grad in grads.values()) > 0, "every gradient is zero"
