from __future__ import annotations

import sys

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

# Learning-to-rank losses that scatter the model's logits into an internal matrix.
# Regression test for #3793: bf16/fp16 training crashed on a float32 dtype mismatch.
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
    reranker_bert_tiny_model_v54: CrossEncoder, loss_cls, dtype: torch.dtype
) -> None:
    model = reranker_bert_tiny_model_v54
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("float16 CrossEncoder forward is only supported when CUDA is available.")
    if dtype == torch.bfloat16 and sys.platform == "win32" and not torch.cuda.is_available():
        pytest.skip(
            "bfloat16 CPU matmul can hard-crash (0xc000001d) on some Windows machines. Skipping to avoid CI failures."
        )

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


@pytest.mark.parametrize("loss_cls", [PListMLELoss, ListMLELoss])  # ListMLELoss subclasses PListMLELoss
@pytest.mark.parametrize("respect_input_order", [True, False])
def test_listwise_loss_is_padding_invariant(reranker_bert_tiny_model_v54: CrossEncoder, loss_cls, respect_input_order):
    # A query's loss must not change because another query in the batch has more documents
    # (which pads this query's row). Batching must equal the mean of the per-query losses.
    # Labels are intentionally unsorted so the respect_input_order=False branch actually
    # permutes via sort/gather (exercising the sorted_mask = gather(mask, indices) path).
    model = reranker_bert_tiny_model_v54
    loss_fn = loss_cls(model, respect_input_order=respect_input_order)

    queries_a = ["what is the capital of france"]
    docs_a = [["berlin is the capital of germany", "paris is the capital of france"]]
    labels_a = [torch.tensor([0.0, 1.0], device=model.device)]

    queries_b = ["what is the largest planet"]
    docs_b = [["mercury is the smallest", "jupiter is the largest planet", "the moon orbits earth"]]
    labels_b = [torch.tensor([1.0, 2.0, 0.0], device=model.device)]

    loss_a = loss_fn((queries_a, docs_a), labels_a)
    loss_b = loss_fn((queries_b, docs_b), labels_b)
    loss_batched = loss_fn((queries_a + queries_b, docs_a + docs_b), labels_a + labels_b)

    # On the buggy implementation, padding the 2-doc query to width 3 inflates its loss, so the
    # batched mean diverges from the mean of the separately computed per-query losses.
    assert torch.allclose(loss_batched, (loss_a + loss_b) / 2, atol=1e-5)
