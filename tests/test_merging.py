# Tests drafted with Claude Code (each case manually reviewed and run against
# the actual merge implementation).
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

mergekit = pytest.importorskip(
    "mergekit",
    reason="Model merging tests require `pip install sentence-transformers[merge]`",
)


@pytest.fixture()
def two_st_copies(stsb_bert_tiny_model: SentenceTransformer):
    """Save the tiny ST model to two on-disk copies and yield their paths."""
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"
        b = Path(tmp) / "b"
        stsb_bert_tiny_model.save(str(a))
        stsb_bert_tiny_model.save(str(b))
        yield str(a), str(b), tmp


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# SentenceTransformer
# ---------------------------------------------------------------------------


def test_linear_merge_of_identical_copies(stsb_bert_tiny_model: SentenceTransformer, two_st_copies):
    """Linear-merging a model with itself must reproduce the original embeddings."""
    a, b, tmp = two_st_copies
    sentences = ["A test sentence.", "Another piece of text."]
    original = stsb_bert_tiny_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    merged_path = Path(tmp) / "merged"
    merged = SentenceTransformer.merge(
        models=[a, b],
        weights=[0.5, 0.5],
        method="linear",
        output_path=str(merged_path),
        dtype="float32",
        device="cpu",
    )
    merged_emb = merged.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    assert _max_abs_diff(original, merged_emb) < 1e-3


@pytest.mark.parametrize(
    "method,needs_base",
    [
        ("slerp", False),
        ("ties", True),
        ("dare_ties", True),
        ("dare_linear", True),
        ("task_arithmetic", True),
    ],
)
def test_non_linear_methods_identical_copies(
    stsb_bert_tiny_model: SentenceTransformer, two_st_copies, method, needs_base
):
    """Each merge method, applied to two identical copies, must reproduce the
    original embeddings (within numerical precision). Catches regressions in
    method-specific YAML synthesis (e.g. slerp's `t` parameter placement)."""
    a, b, tmp = two_st_copies
    sentences = ["A test sentence.", "Another piece of text."]
    original = stsb_bert_tiny_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    kwargs = {"base_model": a} if needs_base else {}
    merged = SentenceTransformer.merge(
        models=[a, b],
        method=method,
        output_path=str(Path(tmp) / f"merged_{method}"),
        dtype="float32",
        device="cpu",
        **kwargs,
    )
    merged_emb = merged.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    assert _max_abs_diff(original, merged_emb) < 1e-3, f"{method} diverged from original"


def test_uniform_weights_default(stsb_bert_tiny_model: SentenceTransformer, two_st_copies):
    """Omitting ``weights`` should use uniform distribution and still reproduce original."""
    a, b, tmp = two_st_copies
    sentences = ["Default-weight test."]
    original = stsb_bert_tiny_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    merged = SentenceTransformer.merge(
        models=[a, b],
        method="linear",
        output_path=str(Path(tmp) / "merged_default"),
        dtype="float32",
        device="cpu",
    )
    merged_emb = merged.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    assert _max_abs_diff(original, merged_emb) < 1e-3


def test_pooling_mode_mismatch_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    pool_cfg_path = Path(b) / "1_Pooling" / "config.json"
    cfg = json.loads(pool_cfg_path.read_text())
    cfg["pooling_mode"] = "cls"
    pool_cfg_path.write_text(json.dumps(cfg, indent=2))

    with pytest.raises(ValueError, match="Module #1 config mismatch"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged_pool_mismatch"),
            dtype="float32",
            device="cpu",
        )


def test_mixed_modules_json_presence_is_rejected(two_st_copies):
    """One input has modules.json, the other doesn't — reject with a clear error."""
    a, b, tmp = two_st_copies
    (Path(b) / "modules.json").unlink()
    with pytest.raises(ValueError, match="disagree on modules.json"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged_mixed"),
            dtype="float32",
            device="cpu",
        )


def test_module_count_mismatch_is_rejected(stsb_bert_tiny_model: SentenceTransformer, two_st_copies):
    a, b, tmp = two_st_copies
    # Drop the last module entry from ``b``'s modules.json so the two models differ in length.
    modules_path = Path(b) / "modules.json"
    modules = json.loads(modules_path.read_text())
    modules.pop()
    modules_path.write_text(json.dumps(modules, indent=2))
    with pytest.raises(ValueError, match="Module count mismatch"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged_count_mismatch"),
            dtype="float32",
            device="cpu",
        )


def test_unsupported_method_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="Unsupported merge method"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="not_a_real_method",
            output_path=str(Path(tmp) / "merged_bad_method"),
            dtype="float32",
            device="cpu",
        )


def test_missing_base_model_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="requires `base_model`"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="ties",
            output_path=str(Path(tmp) / "merged_no_base"),
            dtype="float32",
            device="cpu",
        )


def test_slerp_requires_two_models(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="requires exactly 2 input models"):
        SentenceTransformer.merge(
            models=[a, b, a],
            method="slerp",
            output_path=str(Path(tmp) / "merged_slerp_3"),
            dtype="float32",
            device="cpu",
        )


def test_slerp_rejects_duplicate_models(two_st_copies):
    a, _b, tmp = two_st_copies
    with pytest.raises(ValueError, match="two distinct input models"):
        SentenceTransformer.merge(
            models=[a, a],
            method="slerp",
            output_path=str(Path(tmp) / "merged_slerp_dup"),
            dtype="float32",
            device="cpu",
        )


def test_slerp_base_model_must_be_in_models(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="base must be one of `models`"):
        SentenceTransformer.merge(
            models=[a, b],
            method="slerp",
            base_model="some/other-model",
            output_path=str(Path(tmp) / "merged_slerp_bad_base"),
            dtype="float32",
            device="cpu",
        )


def test_single_model_rejected(two_st_copies):
    a, _b, tmp = two_st_copies
    with pytest.raises(ValueError, match="At least two models are required"):
        SentenceTransformer.merge(
            models=[a],
            method="linear",
            output_path=str(Path(tmp) / "merged_one"),
            dtype="float32",
            device="cpu",
        )


def test_delta_method_default_weights_are_one(two_st_copies):
    """For task_arithmetic, default weights must be [1.0]*n (raw scaling),
    not the simplex [1/n]*n used for blend methods."""
    from sentence_transformers.base.merging import _resolve_weights

    assert _resolve_weights(None, 2, "linear") == [0.5, 0.5]
    assert _resolve_weights(None, 3, "linear") == [1.0 / 3] * 3
    assert _resolve_weights(None, 2, "task_arithmetic") == [1.0, 1.0]
    assert _resolve_weights(None, 3, "ties") == [1.0, 1.0, 1.0]
    # Explicit weights pass through unchanged for both families.
    assert _resolve_weights([0.6, 0.4], 2, "linear") == [0.6, 0.4]
    assert _resolve_weights([0.6, 0.4], 2, "task_arithmetic") == [0.6, 0.4]


def test_output_path_required(two_st_copies):
    a, b, _ = two_st_copies
    with pytest.raises(ValueError, match="output_path"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=None,
        )


# ---------------------------------------------------------------------------
# SparseEncoder
# ---------------------------------------------------------------------------


def test_sparse_encoder_merge(splade_bert_tiny_model: SparseEncoder):
    sentences = ["Sparse encoder test sentence."]
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"
        b = Path(tmp) / "b"
        splade_bert_tiny_model.save(str(a))
        splade_bert_tiny_model.save(str(b))

        merged = SparseEncoder.merge(
            models=[str(a), str(b)],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged"),
            dtype="float32",
            device="cpu",
        )
        emb_orig = splade_bert_tiny_model.encode(sentences)
        emb_merged = merged.encode(sentences)
        diff = (emb_orig.to_dense() - emb_merged.to_dense()).abs().max().item()
        assert diff < 1e-2


# ---------------------------------------------------------------------------
# CrossEncoder
# ---------------------------------------------------------------------------


def test_cross_encoder_merge(reranker_bert_tiny_model: CrossEncoder):
    pairs = [("query 1", "candidate doc"), ("query 2", "another candidate")]
    with tempfile.TemporaryDirectory() as tmp:
        a = Path(tmp) / "a"
        b = Path(tmp) / "b"
        reranker_bert_tiny_model.save(str(a))
        reranker_bert_tiny_model.save(str(b))

        merged = CrossEncoder.merge(
            models=[str(a), str(b)],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged"),
            dtype="float32",
            device="cpu",
        )
        original = np.asarray(reranker_bert_tiny_model.predict(pairs))
        merged_scores = np.asarray(merged.predict(pairs))
        assert _max_abs_diff(original, merged_scores) < 1e-2
