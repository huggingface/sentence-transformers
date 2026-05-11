from __future__ import annotations

import pytest
import torch
from torch import nn

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import BaseTrainer
from sentence_transformers.sentence_transformer.losses import AdaptiveLayerLoss, MultipleNegativesRankingLoss


class _FakeDDP(nn.Module):
    """Stand-in for `torch.nn.parallel.DistributedDataParallel` that exposes the wrapped
    module via `.module` and proxies `forward` like real DDP. Notably it does NOT support
    `__getitem__`, so `model[0]` raises TypeError."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.module(*args, **kwargs)


class _FakeCompile(nn.Module):
    """Stand-in for `torch._dynamo.OptimizedModule` (output of `torch.compile`): exposes
    the wrapped module via `_orig_mod` and proxies `forward`. Also not subscriptable."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self._orig_mod(*args, **kwargs)


class _StubTrainer:
    """Borrow `BaseTrainer.override_model_in_loss` without instantiating the full trainer.
    The method only uses `self` for its recursive call, so any object with the same bound
    attribute works."""

    override_model_in_loss = BaseTrainer.override_model_in_loss


def _make_loss(model: SentenceTransformer) -> AdaptiveLayerLoss:
    inner_loss = MultipleNegativesRankingLoss(model)
    return AdaptiveLayerLoss(model, inner_loss)


def _features_and_labels(model: SentenceTransformer) -> tuple[list[dict], torch.Tensor]:
    features = [
        {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
        for feats in [model.preprocess(["a", "b"]), model.preprocess(["c", "d"])]
    ]
    labels = torch.tensor([0, 1], device=model.device)
    return features, labels


def test_adaptive_layer_loss_runs_without_wrapper(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Sanity check for the non-DDP, non-compile path: a bare SentenceTransformer skips
    the unwrap loop entirely."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "wrapper_cls",
    [_FakeDDP, _FakeCompile],
    ids=["ddp", "torch_compile"],
)
def test_adaptive_layer_loss_unwraps_wrapped_model(
    stsb_bert_tiny_model: SentenceTransformer,
    wrapper_cls: type[nn.Module],
) -> None:
    """AdaptiveLayerLoss.forward reaches into `self.model[0]`. Under DDP / torch.compile
    the trainer rebinds `loss.model` to the wrapper (and binds BaseModel methods like
    `preprocess` onto it). Verify the loss still unwraps to the inner BaseModel for the
    `model[0]` decoration. See #3170.
    """
    adaptive = _make_loss(stsb_bert_tiny_model)
    wrapped = wrapper_cls(stsb_bert_tiny_model)
    with pytest.raises(TypeError):
        _ = wrapped[0]  # type: ignore[index]

    # Run the real trainer codepath: this also setattrs `preprocess` /
    # `get_embedding_dimension` onto the wrapper, which would fool a
    # `hasattr(model, "preprocess")` stop condition.
    _StubTrainer().override_model_in_loss(adaptive, wrapped)  # type: ignore[arg-type]
    assert adaptive.model is wrapped
    assert hasattr(wrapped, "preprocess")

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda m: _FakeCompile(_FakeDDP(m)),
        lambda m: _FakeDDP(_FakeCompile(m)),
    ],
    ids=["compile_of_ddp", "ddp_of_compile"],
)
def test_adaptive_layer_loss_unwraps_nested_wrappers(
    stsb_bert_tiny_model: SentenceTransformer,
    wrap,
) -> None:
    """Compile-of-DDP and DDP-of-compile both need to unwrap to the inner BaseModel."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    wrapped = wrap(stsb_bert_tiny_model)
    _StubTrainer().override_model_in_loss(adaptive, wrapped)  # type: ignore[arg-type]

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    loss = adaptive(features, labels)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_adaptive_layer_loss_raises_on_unrecognised_wrapper(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """If `self.model` is something we can't unwrap to a BaseModel, fail loudly instead of
    silently raising the cryptic `'X' object is not subscriptable`."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    adaptive.model = nn.Linear(2, 2)  # neither a BaseModel nor a recognised wrapper

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    with pytest.raises(TypeError, match="could not unwrap"):
        adaptive(features, labels)


def test_adaptive_layer_loss_error_names_inner_stuck_point(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """For nested wrappers the error should name both the outer wrapper (what the user
    passed in) and the inner object the unwrap got stuck on."""
    adaptive = _make_loss(stsb_bert_tiny_model)
    adaptive.model = _FakeDDP(nn.Linear(2, 2))  # DDP wraps an unrecognised inner

    features, labels = _features_and_labels(stsb_bert_tiny_model)
    with pytest.raises(TypeError, match=r"could not unwrap _FakeDDP .*stopped at Linear"):
        adaptive(features, labels)
