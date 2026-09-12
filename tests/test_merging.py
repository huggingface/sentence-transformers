from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder


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

    merged = SentenceTransformer.merge(
        models=[a, b],
        weights=[0.5, 0.5],
        method="linear",
        output_path=str(Path(tmp) / "merged"),
        dtype="float32",
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
    original embeddings (deltas are zero, so every method reduces to the base)."""
    a, b, tmp = two_st_copies
    sentences = ["A test sentence.", "Another piece of text."]
    original = stsb_bert_tiny_model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    kwargs = {"base_model": a} if needs_base else {}
    merged = SentenceTransformer.merge(
        models=[a, b],
        method=method,
        output_path=str(Path(tmp) / f"merged_{method}"),
        dtype="float32",
        **kwargs,
    )
    merged_emb = merged.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    assert _max_abs_diff(original, merged_emb) < 1e-3, f"{method} diverged from original"


def test_merge_with_normalize_module(stsb_bert_tiny_model: SentenceTransformer):
    """A model whose stack includes a file-less stateless module (Normalize) must
    merge and reload. Normalize saves no dir, which previously crashed the merge."""
    from sentence_transformers.sentence_transformer.modules import Normalize

    model = deepcopy(stsb_bert_tiny_model)
    model.append(Normalize())
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        model.save(str(a))
        model.save(str(b))
        merged = SentenceTransformer.merge(
            models=[str(a), str(b)], weights=[0.5, 0.5], method="linear",
            output_path=str(Path(tmp) / "merged"), dtype="float32",
        )
        emb = merged.encode(["A normalized sentence."], convert_to_numpy=True)
        assert emb.shape[1] == model.get_sentence_embedding_dimension()
        assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-4  # Normalize survived


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
    )
    merged_emb = merged.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    assert _max_abs_diff(original, merged_emb) < 1e-3


def test_pooling_mode_mismatch_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    pool_cfg_path = Path(b) / "1_Pooling" / "config.json"
    cfg = json.loads(pool_cfg_path.read_text())
    cfg["pooling_mode"] = "cls"
    cfg["pooling_mode_mean_tokens"] = False
    cfg["pooling_mode_cls_token"] = True
    pool_cfg_path.write_text(json.dumps(cfg, indent=2))

    with pytest.raises(ValueError, match="Module #1 config mismatch"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged_pool_mismatch"),
            dtype="float32",
        )


def test_hidden_size_mismatch_is_rejected(two_st_copies):
    """Different hidden_size must fail up front with a clear message, not a
    low-level tensor shape error during the body merge."""
    a, b, tmp = two_st_copies
    cfg_path = Path(b) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["hidden_size"] = cfg["hidden_size"] + 64
    cfg_path.write_text(json.dumps(cfg, indent=2))
    with pytest.raises(ValueError, match="Hidden size mismatch"):
        SentenceTransformer.merge(
            models=[a, b],
            method="linear",
            output_path=str(Path(tmp) / "merged_dim_mismatch"),
            dtype="float32",
        )


def test_architecture_mismatch_is_rejected(two_st_copies):
    """Different model_type must fail up front with a clear message."""
    a, b, tmp = two_st_copies
    cfg_path = Path(b) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["model_type"] = "roberta"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    with pytest.raises(ValueError, match="Architecture mismatch"):
        SentenceTransformer.merge(
            models=[a, b],
            method="linear",
            output_path=str(Path(tmp) / "merged_arch_mismatch"),
            dtype="float32",
        )


def test_mixed_modules_json_presence_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    (Path(b) / "modules.json").unlink()
    with pytest.raises(ValueError, match="disagree on modules.json"):
        SentenceTransformer.merge(
            models=[a, b],
            weights=[0.5, 0.5],
            method="linear",
            output_path=str(Path(tmp) / "merged_mixed"),
            dtype="float32",
        )


def test_module_count_mismatch_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
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
        )


def test_unsupported_method_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="Unsupported merge method"):
        SentenceTransformer.merge(
            models=[a, b],
            method="not_a_real_method",
            output_path=str(Path(tmp) / "merged_bad_method"),
            dtype="float32",
        )


def test_missing_base_model_is_rejected(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="requires `base_model`"):
        SentenceTransformer.merge(
            models=[a, b],
            method="ties",
            output_path=str(Path(tmp) / "merged_no_base"),
            dtype="float32",
        )


def test_unresolvable_base_model_is_rejected(two_st_copies):
    """A delta method with a base_model that doesn't resolve must error, not
    silently fall back to a linear merge."""
    a, b, tmp = two_st_copies
    with pytest.raises(FileNotFoundError, match="base_model"):
        SentenceTransformer.merge(
            models=[a, b],
            method="ties",
            base_model="does/not-exist-locally",
            output_path=str(Path(tmp) / "merged_bad_base"),
            dtype="float32",
            local_files_only=True,
        )


def test_slerp_requires_two_models(two_st_copies):
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="requires exactly 2 input models"):
        SentenceTransformer.merge(
            models=[a, b, a],
            method="slerp",
            output_path=str(Path(tmp) / "merged_slerp_3"),
            dtype="float32",
        )


def test_slerp_rejects_duplicate_models(two_st_copies):
    a, _b, tmp = two_st_copies
    with pytest.raises(ValueError, match="two distinct input models"):
        SentenceTransformer.merge(
            models=[a, a],
            method="slerp",
            output_path=str(Path(tmp) / "merged_slerp_dup"),
            dtype="float32",
        )


def test_single_model_rejected(two_st_copies):
    a, _b, tmp = two_st_copies
    with pytest.raises(ValueError, match="At least two models are required"):
        SentenceTransformer.merge(
            models=[a],
            method="linear",
            output_path=str(Path(tmp) / "merged_one"),
            dtype="float32",
        )


def test_delta_method_default_weights_are_one():
    """For delta methods, default weights must be [1.0]*n (raw scaling),
    not the simplex [1/n]*n used for blend methods."""
    from sentence_transformers.base.merging import _resolve_weights

    assert _resolve_weights(None, 2, "linear") == [0.5, 0.5]
    assert _resolve_weights(None, 3, "linear") == [1.0 / 3] * 3
    assert _resolve_weights(None, 2, "task_arithmetic") == [1.0, 1.0]
    assert _resolve_weights(None, 3, "ties") == [1.0, 1.0, 1.0]
    assert _resolve_weights([0.6, 0.4], 2, "linear") == [0.6, 0.4]
    assert _resolve_weights([0.6, 0.4], 2, "task_arithmetic") == [0.6, 0.4]


def test_output_path_required(two_st_copies):
    a, b, _ = two_st_copies
    with pytest.raises(ValueError, match="output_path"):
        SentenceTransformer.merge(models=[a, b], method="linear", output_path=None)


def test_base_model_rejected_for_non_delta_method(two_st_copies):
    """Passing base_model to linear/slerp is meaningless; reject it instead of
    silently downloading and ignoring the base."""
    a, b, tmp = two_st_copies
    with pytest.raises(ValueError, match="only used by delta"):
        SentenceTransformer.merge(
            models=[a, b], method="linear", base_model=a,
            output_path=str(Path(tmp) / "merged_lin_base"), dtype="float32",
        )


def test_ties_uses_mean_not_sum_of_agreeing_deltas():
    """TIES must average the sign-agreeing deltas, not sum them (Yadav et al. 2023).
    Two identical positive deltas must merge to that delta, not 2x it."""
    from sentence_transformers.base.merging import _merge_delta

    base = OrderedDict({"w": torch.zeros(4)})
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0])
    sd = OrderedDict({"w": delta.clone()})
    out = _merge_delta(
        [sd, sd], [1.0, 1.0], base, "ties", [1.0, 1.0],
        torch.device("cpu"), torch.Generator().manual_seed(0),
    )
    assert torch.allclose(out["w"], delta)  # mean, not [2, 4, 6, 8]


def test_dare_density_zero_does_not_nan():
    """density=0 must drop every delta (contribute nothing), not produce 0/0 NaN."""
    from sentence_transformers.base.merging import _merge_delta

    base = OrderedDict({"w": torch.ones(4)})
    sd = OrderedDict({"w": torch.tensor([1.5, 2.0, 0.5, 1.0])})
    out = _merge_delta(
        [sd, sd], [1.0, 1.0], base, "dare_linear", [0.0, 0.0],
        torch.device("cpu"), torch.Generator().manual_seed(0),
    )
    assert not torch.isnan(out["w"]).any()
    assert torch.allclose(out["w"], torch.ones(4))  # all deltas dropped -> base


def test_merged_model_does_not_inherit_input_readme(two_st_copies):
    """The merged output must not carry the first input's model card verbatim."""
    a, b, tmp = two_st_copies
    (Path(a) / "README.md").write_text("INPUT-MODEL-CARD-MARKER")
    merged_path = Path(tmp) / "merged_readme"
    SentenceTransformer.merge(
        models=[a, b], method="linear", output_path=str(merged_path), dtype="float32",
    )
    readme = merged_path / "README.md"
    assert not readme.exists() or "INPUT-MODEL-CARD-MARKER" not in readme.read_text()


def test_weighted_linear_merge_interpolates(stsb_bert_tiny_model, two_st_copies):
    """A non-trivial weighting of two *distinct* models must land between them.

    We perturb one copy's Dense-free body so the two inputs actually differ, then
    check the 0.5/0.5 merge sits strictly between the individual outputs.
    """
    import torch
    from safetensors.torch import load_file, save_file

    a, b, tmp = two_st_copies
    sf = Path(b) / "model.safetensors"
    state = load_file(str(sf))
    # Nudge every float tensor in copy `b` so it diverges from `a`.
    state = {k: (v + 0.01 if v.is_floating_point() else v) for k, v in state.items()}
    save_file(state, str(sf))

    sentences = ["Interpolation check sentence."]
    emb_a = SentenceTransformer(a).encode(sentences, convert_to_numpy=True)
    emb_b = SentenceTransformer(b).encode(sentences, convert_to_numpy=True)
    merged = SentenceTransformer.merge(
        models=[a, b], weights=[0.5, 0.5], method="linear",
        output_path=str(Path(tmp) / "merged_interp"), dtype="float32",
    )
    emb_m = merged.encode(sentences, convert_to_numpy=True)
    # Merged embedding differs from both inputs (inputs themselves differ).
    assert _max_abs_diff(emb_a, emb_b) > 1e-4
    assert _max_abs_diff(emb_a, emb_m) > 1e-6
    assert _max_abs_diff(emb_b, emb_m) > 1e-6


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
        )
        emb_orig = splade_bert_tiny_model.encode(sentences)
        emb_merged = merged.encode(sentences)
        diff = (emb_orig.to_dense() - emb_merged.to_dense()).abs().max().item()
        assert diff < 1e-2


# ---------------------------------------------------------------------------
# CrossEncoder
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MultiVectorEncoder (ColBERT-style: Transformer + Dense projection + mask)
# ---------------------------------------------------------------------------


def test_multi_vector_encoder_merge():
    """Merging must carry the ColBERT projection Dense and the file-less stateless
    modules (MultiVectorMask, Normalize) through to a valid, reloadable model."""
    docs = ["Model merging blends checkpoints.", "The cat sat on the mat."]
    with tempfile.TemporaryDirectory() as tmp:
        # Bare tiny BERT -> MVE auto-appends a projection Dense + MultiVectorMask.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        model.save(str(a))
        model.save(str(b))

        merged = MultiVectorEncoder.merge(
            models=[str(a), str(b)], weights=[0.5, 0.5], method="linear",
            output_path=str(Path(tmp) / "merged"), dtype="float32",
        )
        orig = model.encode(docs)
        got = merged.encode(docs)
        assert [tuple(x.shape) for x in got] == [tuple(x.shape) for x in orig]
        # ColBERT token-level projection dimension is preserved.
        assert got[0].shape[1] == orig[0].shape[1]
        max_diff = max(float((o.float() - m.float()).abs().max()) for o, m in zip(orig, got))
        assert max_diff < 1e-3


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
        )
        original = np.asarray(reranker_bert_tiny_model.predict(pairs))
        merged_scores = np.asarray(merged.predict(pairs))
        assert _max_abs_diff(original, merged_scores) < 1e-2
