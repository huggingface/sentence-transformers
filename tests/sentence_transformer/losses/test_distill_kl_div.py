from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import DistillKLDivLoss


def test_distill_kl_div_warns_once_when_the_teacher_collapses(caplog) -> None:
    """A teacher_temperature well below the teacher's score spread underflows the target to exact
    zeros, so those candidates carry no gradient. Checked on the first forward only, since reading
    the count off an accelerator costs a device synchronization."""
    generator = torch.Generator().manual_seed(7)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(3)]
    # A bge-reranker style row: a 19-unit spread over 0.1 is a ratio of 190, past float32's ~100.
    labels = torch.tensor([[8.5, -10.0], [7.0, -12.0]])

    loss = DistillKLDivLoss(model=None, teacher_temperature=0.1)
    with caplog.at_level("WARNING"):
        loss.compute_loss_from_embeddings(embeddings, labels)
    assert "teacher_temperature=0.1" in caplog.text
    assert "carry no gradient" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        loss.compute_loss_from_embeddings(embeddings, labels)
    assert caplog.text == "", "the check is latched to the first forward"

    quiet = DistillKLDivLoss(model=None)
    with caplog.at_level("WARNING"):
        quiet.compute_loss_from_embeddings(embeddings, labels)
    assert caplog.text == "", "a temperature matched to the spread must not warn"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), False])
def test_distill_kl_div_rejects_unusable_temperatures(bad: float) -> None:
    """A negative temperature silently inverts the objective and NaN poisons every weight on the
    first backward, so both are rejected at construction."""
    for kwargs in ({"temperature": bad}, {"student_temperature": bad}, {"teacher_temperature": bad}):
        with pytest.raises(ValueError, match="must be a positive finite number"):
            DistillKLDivLoss(model=None, **kwargs)


def test_distill_kl_div_rejects_mismatched_label_width() -> None:
    """A narrower label tensor broadcasts across the candidates instead of erroring, which trains the
    student toward a uniform distribution at a plausible-looking loss."""
    generator = torch.Generator().manual_seed(5)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(3)]
    loss = DistillKLDivLoss(model=None)
    with pytest.raises(ValueError, match=r"teacher scores of shape \(batch_size, N\)"):
        loss.compute_loss_from_embeddings(embeddings, torch.tensor([[1.0], [1.0]]))


def test_distill_kl_div_rejects_nan_teacher_scores() -> None:
    """NaN labels make every gradient NaN on the first backward, and the exact-zero collapse check
    cannot see them since ``nan == 0`` is False."""
    generator = torch.Generator().manual_seed(9)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(3)]
    loss = DistillKLDivLoss(model=None)
    with pytest.raises(ValueError, match="teacher scores contain NaN"):
        loss.compute_loss_from_embeddings(embeddings, torch.tensor([[1.0, 2.0], [float("nan"), 1.0]]))


def test_distill_kl_div_does_not_warn_on_excluded_candidates(caplog) -> None:
    """An infinite score marks a candidate the caller excluded on purpose, so its zero probability is
    not a temperature problem and must not be reported as one."""
    generator = torch.Generator().manual_seed(11)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(3)]
    loss = DistillKLDivLoss(model=None)
    with caplog.at_level("WARNING"):
        loss.compute_loss_from_embeddings(embeddings, torch.tensor([[5.0, -float("inf")], [4.0, 1.0]]))
    assert caplog.text == ""


def test_distill_kl_div_rejects_a_single_candidate() -> None:
    """A softmax over one candidate is constant, so the loss and gradient are identically zero: that
    must fail loudly rather than report a perfect loss while training nothing."""
    generator = torch.Generator().manual_seed(3)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(2)]
    loss = DistillKLDivLoss(model=None)
    with pytest.raises(ValueError, match="at least 3 columns"):
        loss.compute_loss_from_embeddings(embeddings, torch.tensor([[1.0], [1.0]]))


def test_distill_kl_div_separate_temperatures() -> None:
    """student_temperature / teacher_temperature default to the shared temperature and split the two
    softmaxes when set, with the loss scaled by the student temperature squared."""
    generator = torch.Generator().manual_seed(11)
    embeddings = [torch.randn(4, 8, generator=generator) for _ in range(3)]
    labels = torch.tensor([[4.0, 1.0], [3.5, 0.5], [2.0, 1.5], [5.0, 0.0]])

    shared = DistillKLDivLoss(model=None, temperature=0.5)
    shared_value = shared.compute_loss_from_embeddings(embeddings, labels).item()
    aliased = DistillKLDivLoss(model=None, student_temperature=0.5, teacher_temperature=0.5)
    assert aliased.compute_loss_from_embeddings(embeddings, labels).item() == pytest.approx(shared_value)

    split = DistillKLDivLoss(model=None, student_temperature=0.5, teacher_temperature=2.0)
    assert split.compute_loss_from_embeddings(embeddings, labels).item() != pytest.approx(shared_value)
    assert split.get_config_dict() == {
        "similarity_fct": "pairwise_dot_score",
        "temperature": 1.0,
        "student_temperature": 0.5,
        "teacher_temperature": 2.0,
    }


def test_distill_kl_div_min_max_normalization() -> None:
    """normalize_student / normalize_teacher rescale each row to [0, 1] before the softmax, so the loss
    matches the one computed on pre-normalized scores and no longer depends on the teacher's scale."""
    generator = torch.Generator().manual_seed(13)
    embeddings = [torch.randn(4, 8, generator=generator) for _ in range(4)]
    labels = torch.tensor([[9.0, 2.0, -3.0], [4.0, 6.0, 1.0], [0.5, 0.2, 0.1], [-1.0, -8.0, 3.0]])

    both = DistillKLDivLoss(model=None, normalize_student=True, normalize_teacher=True)
    value = both.compute_loss_from_embeddings(embeddings, labels)

    student_scores = torch.stack([(embeddings[0] * other).sum(-1) for other in embeddings[1:]], dim=1)
    student = (student_scores - student_scores.amin(1, keepdim=True)) / (
        student_scores.amax(1, keepdim=True) - student_scores.amin(1, keepdim=True)
    )
    teacher = (labels - labels.amin(1, keepdim=True)) / (labels.amax(1, keepdim=True) - labels.amin(1, keepdim=True))
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=1), torch.softmax(teacher, dim=1), reduction="batchmean"
    )
    assert value.item() == pytest.approx(expected.item(), abs=1e-6)

    rescaled = both.compute_loss_from_embeddings(embeddings, labels * 25.0 + 7.0)
    assert rescaled.item() == pytest.approx(value.item(), abs=1e-6)
    assert DistillKLDivLoss(model=None).compute_loss_from_embeddings(embeddings, labels).item() != pytest.approx(
        value.item()
    )
    assert both.get_config_dict() == {
        "similarity_fct": "pairwise_dot_score",
        "temperature": 1.0,
        "normalize_student": True,
        "normalize_teacher": True,
    }


def test_distill_kl_div_min_max_normalization_keeps_excluded_candidates() -> None:
    """An infinite teacher score marks an excluded candidate: it must not leak into the row's min or
    max, and must stay at zero probability instead of turning the loss into NaN."""
    generator = torch.Generator().manual_seed(17)
    embeddings = [torch.randn(2, 8, generator=generator) for _ in range(4)]
    loss = DistillKLDivLoss(model=None, normalize_student=True, normalize_teacher=True)
    value = loss.compute_loss_from_embeddings(embeddings, torch.tensor([[5.0, 1.0, -float("inf")], [4.0, 1.0, 2.0]]))
    assert torch.isfinite(value)
