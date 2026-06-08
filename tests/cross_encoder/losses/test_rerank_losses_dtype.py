from __future__ import annotations

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import (
    ADRMSELoss,
    LambdaLoss,
    ListMLELoss,
    ListNetLoss,
    PListMLELoss,
    RankNetLoss,
)

# Learning-to-rank losses that build an internal logits matrix from the model's
# logits. Regression test for #3793: training in low precision (bf16/fp16) used to
# crash with "Index put requires the source and destination dtypes match" because
# the matrix defaulted to float32 while the model emitted bf16/fp16 logits.
LISTWISE_LOSSES = [
    PListMLELoss,
    LambdaLoss,
    ListNetLoss,
    ListMLELoss,  # subclass of PListMLELoss
    RankNetLoss,  # subclass of LambdaLoss
    ADRMSELoss,
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("loss_cls", LISTWISE_LOSSES)
def test_listwise_loss_supports_low_precision(
    reranker_bert_tiny_model_v6: CrossEncoder, loss_cls, dtype: torch.dtype
) -> None:
    model = reranker_bert_tiny_model_v6
    model.model.to(dtype)
    loss_fn = loss_cls(model)

    queries = ["What is Python?", "What is PyTorch?"]
    docs_list = [
        ["A programming language.", "A snake."],
        ["A deep learning framework.", "A lunch box.", "A board game."],
    ]
    labels = [torch.tensor([1.0, 0.0], device=model.device), torch.tensor([2.0, 1.0, 0.0], device=model.device)]

    # Must not raise a dtype-mismatch RuntimeError, and the loss should be computed
    # in float32 for numerical stability of exp/log/cumsum/softmax.
    loss_value = loss_fn((queries, docs_list), labels)
    assert loss_value.dtype == torch.float32
    assert torch.isfinite(loss_value)

    # Gradients must flow back into the low-precision model without dtype errors.
    loss_value.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Expected at least one parameter to receive a gradient."
    assert all(torch.isfinite(grad).all() for grad in grads)
