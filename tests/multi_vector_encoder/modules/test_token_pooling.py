from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations

import pytest
import torch
from torch import Tensor

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import (
    HierarchicalTokenPooling,
    LambdaTokenPooling,
)
from sentence_transformers.util import batch_to_device


def _normed(shape: tuple[int, int]) -> Tensor:
    return torch.nn.functional.normalize(torch.randn(*shape), p=2, dim=-1)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


class TestPoolShapeInvariants:
    def test_list_input_returns_list(self) -> None:
        docs = [_normed((12, 8)), _normed((20, 8))]
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(docs)
        assert isinstance(out, list)
        assert [e.shape for e in out] == [(6, 8), (10, 8)]

    def test_3d_input_returns_3d(self) -> None:
        docs = [_normed((12, 8)), _normed((20, 8))]
        padded = torch.nn.utils.rnn.pad_sequence(docs, batch_first=True)
        attn = padded.abs().sum(-1) > 0
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(padded, attention_mask=attn)
        assert isinstance(out, Tensor)
        # 20 // 2 = 10 tokens is the max. Batch is (2, 10, 8).
        assert out.shape == (2, 10, 8)

    def test_3d_input_left_padding_round_trip(self) -> None:
        docs = [_normed((6, 4)), _normed((10, 4))]
        # Left-pad by hand.
        max_len = max(len(d) for d in docs)
        padded = torch.stack([torch.cat([torch.zeros(max_len - len(d), 4), d], dim=0) for d in docs])
        pooling = HierarchicalTokenPooling(pool_factor=2)
        # No mask: rely on zero-detection with padding_side="left".
        out = pooling.pool(padded, padding_side="left")
        assert isinstance(out, Tensor)
        # Padding preserved on the left after the round-trip.
        assert torch.all(out[0, : out.shape[1] - 3] == 0)
        assert (out[1, :5] != 0).any(dim=-1).any()

    def test_empty_list_input(self) -> None:
        pooling = HierarchicalTokenPooling(pool_factor=2)
        assert pooling.pool([]) == []

    def test_task_outside_tasks_returns_input_unchanged(self) -> None:
        # encode() delegates its per-call gating to this: with the default tasks (documents only),
        # pool(task="query") must be an identity.
        docs = [_normed((12, 8))]
        pooling = HierarchicalTokenPooling(pool_factor=2)
        assert pooling.pool(docs, task="query") is docs
        assert isinstance(pooling.pool(docs, task="document"), list)

    def test_bad_input_type_raises(self) -> None:
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="Tensor input must be 3D"):
            pooling.pool(_normed((5, 8)))  # 2D bare tensor rejected

    def test_3d_unbind_preserves_middle_zero_row(self) -> None:
        # A real token with an all-zero embedding must not be clipped off the end of its sequence.
        # (Regression guard for the vectorized fallback path that uses row-is-real detection.)
        emb = torch.stack(
            [
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 0.0]),  # zero-valued but "real"
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.0, 0.0]),  # padding
            ]
        )
        padded = emb.unsqueeze(0)  # (1, 4, 2)
        # Round-trip through LambdaTokenPooling with an identity func, no attention_mask.
        out = LambdaTokenPooling(pool_fn=lambda x: x).pool(padded, padding_side="right")
        # Expected: real rows [0..2] preserved (including middle zero). Trailing pad row dropped.
        assert out.shape == (1, 3, 2)


class TestHierarchicalTokenPooling:
    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_class_pool_matches_module_helper_directly(self, normalize_embeddings: bool) -> None:
        # The pool() class method and the internal _hierarchical_pool_one produce identical output
        # when given the same input (this pins the wiring between the public API and the math).
        from sentence_transformers.multi_vector_encoder.modules.token_pooling import _hierarchical_pool_one

        emb = _normed((15, 8))
        direct = _hierarchical_pool_one(
            emb, pool_factor=2, num_protected_tokens=1, normalize_embeddings=normalize_embeddings
        )
        via_pool = HierarchicalTokenPooling(
            pool_factor=2, num_protected_tokens=1, normalize_embeddings=normalize_embeddings
        ).pool([emb])[0]
        assert torch.allclose(direct, via_pool)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_pooling_supports_low_precision_dtypes(self, dtype: torch.dtype, normalize_embeddings: bool) -> None:
        # The ColPali/ColQwen2 family runs bf16 by default, and numpy has no bfloat16: the
        # distance/linkage step must run in fp32 while the pooled output keeps the input dtype.
        emb = _normed((15, 8)).to(dtype)
        pooling = HierarchicalTokenPooling(pool_factor=2, normalize_embeddings=normalize_embeddings)
        out = pooling.pool([emb])[0]
        assert out.dtype == dtype
        # Same values in fp32 must yield the same clustering: only the mean accumulation dtype differs.
        reference = pooling.pool([emb.float()])[0]
        assert out.shape == reference.shape
        assert torch.allclose(out.float(), reference, atol=1e-2)

    def test_scipy_fallback_matches_fastcluster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # fastcluster is optional. Without it the pooling must silently fall back to scipy and
        # produce the same clustering (None in sys.modules makes the import raise ImportError).
        pytest.importorskip("fastcluster")
        from sentence_transformers.multi_vector_encoder.modules.token_pooling import _ward_linkage_fn

        emb = _normed((40, 8))
        pooling = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=1)
        default = pooling.pool([emb])[0]
        monkeypatch.setitem(sys.modules, "fastcluster", None)
        _ward_linkage_fn.cache_clear()
        try:
            fallback = pooling.pool([emb])[0]
        finally:
            _ward_linkage_fn.cache_clear()
        assert torch.equal(fallback, default)

    def test_no_op_when_pool_factor_1(self) -> None:
        docs = [_normed((10, 4))]
        pooling = HierarchicalTokenPooling(pool_factor=1, normalize_embeddings=False)
        # Pipeline forward path is a fast no-op.
        token_embeddings = docs[0].unsqueeze(0)
        features = {
            "token_embeddings": token_embeddings,
            "attention_mask": torch.ones(1, 10, dtype=torch.bool),
        }
        out = pooling.forward(features, task="document")
        assert out is features  # returned unchanged (fast path)
        assert out["token_embeddings"] is token_embeddings
        torch.testing.assert_close(pooling.pool(docs)[0], docs[0], rtol=0, atol=0)

    def test_num_protected_tokens_untouched(self) -> None:
        emb = _normed((12, 8))
        out = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=2, normalize_embeddings=False).pool([emb])[
            0
        ]
        # First 2 rows are the protected tokens verbatim.
        assert torch.allclose(out[:2], emb[:2])

    @pytest.mark.parametrize(("pool_factor", "expected_tokens"), [(1.5, 21), (2.5, 13), (3.5, 9)])
    def test_fractional_pool_factor(self, pool_factor: float, expected_tokens: int) -> None:
        # Fractional factors give compression steps between the integers:
        # 1 protected + max(int(30 // pool_factor), 1) cluster means.
        emb = _normed((31, 8))
        out = HierarchicalTokenPooling(pool_factor=pool_factor, num_protected_tokens=1).pool([emb])[0]
        assert out.shape == (expected_tokens, 8)

    def test_pooled_rows_are_cluster_means(self) -> None:
        # Every non-protected row of the output must equal the mean of some non-empty subset of the
        # non-protected input rows, and the subsets must partition the non-protected input rows.
        # Verifies the scatter_add + dense-remap math.
        torch.manual_seed(1)
        emb = _normed((7, 4))  # 6 non-protected rows -> brute-force over 2^6 subsets is trivial.
        num_protected_tokens = 1
        pool_factor = 3
        pooling = HierarchicalTokenPooling(
            pool_factor=pool_factor, num_protected_tokens=num_protected_tokens, normalize_embeddings=False
        )
        out = pooling.pool([emb])[0]
        pooled_rows = out[num_protected_tokens:]
        source_rows = emb[num_protected_tokens:]
        n = source_rows.size(0)
        used = [False] * n
        for pooled_row in pooled_rows:
            match: tuple[int, ...] | None = None
            for size in range(1, n + 1):
                for combo in combinations([i for i in range(n) if not used[i]], size):
                    if torch.allclose(source_rows[list(combo)].mean(dim=0), pooled_row, atol=1e-5):
                        match = combo
                        break
                if match is not None:
                    break
            assert match is not None, "pooled row is not the mean of any unused subset of input rows"
            for i in match:
                used[i] = True
        assert all(used), "some input rows never contributed to a pooled row"


class TestLambdaTokenPooling:
    def test_applies_pool_fn_per_sample(self) -> None:
        def halve(emb: Tensor) -> Tensor:
            n = emb.size(0)
            return emb[: n - n % 2].view(-1, 2, emb.size(-1)).mean(dim=1)

        docs = [_normed((10, 4)), _normed((6, 4))]
        out = LambdaTokenPooling(pool_fn=halve).pool(docs)
        assert [e.shape for e in out] == [(5, 4), (3, 4)]

    def test_save_raises_not_implemented(self, tmp_path) -> None:
        pooling = LambdaTokenPooling(pool_fn=lambda x: x)
        with pytest.raises(NotImplementedError, match="cannot be saved"):
            pooling.save(str(tmp_path))


def _keep_first_two(emb: Tensor) -> Tensor:
    """Module-level (pickle-friendly) pool function used by the pickle smoke test."""
    return emb[:2]


class TestPicklability:
    def test_hierarchical_pooling_round_trips_through_pickle(self) -> None:
        # Ensures `pool={...}` (multi-process encode) works with a saveable pooling.
        pooling = HierarchicalTokenPooling(pool_factor=3, num_protected_tokens=1)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, HierarchicalTokenPooling)
        assert restored.pool_factor == 3 and restored.num_protected_tokens == 1
        emb = _normed((12, 8))
        assert torch.allclose(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_top_level_func_pickles(self) -> None:
        # LambdaTokenPooling survives pickle as long as `pool_fn` is picklable itself.
        pooling = LambdaTokenPooling(pool_fn=_keep_first_two)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, LambdaTokenPooling)
        emb = _normed((10, 4))
        assert torch.equal(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_lambda_does_not_pickle(self) -> None:
        # Anonymous lambdas / nested functions won't pickle. Confirms the docstring caveat.
        pooling = LambdaTokenPooling(pool_fn=lambda x: x)
        with pytest.raises((pickle.PicklingError, AttributeError)):
            pickle.dumps(pooling)


class TestPipelineModuleForward:
    def test_forward_pools_documents(self) -> None:
        pooling = HierarchicalTokenPooling(pool_factor=2)
        features = {
            "token_embeddings": _normed((2, 12, 8)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="document")
        assert out["token_embeddings"].shape[1] == 6

    @pytest.mark.parametrize("pool_factor", [1, 2])
    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_forward_skips_queries(self, pool_factor: float, normalize_embeddings: bool) -> None:
        pooling = HierarchicalTokenPooling(pool_factor=pool_factor, normalize_embeddings=normalize_embeddings)
        embeddings = _normed((2, 12, 8))
        features = {
            "token_embeddings": embeddings,
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"] is embeddings  # unchanged reference

    def test_forward_pools_queries_when_opted_in(self) -> None:
        # CRISP-style strategies compress queries too: listing "query" in tasks lifts the gate.
        pooling = HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"])
        features = {
            "token_embeddings": _normed((2, 12, 8)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"].shape[1] == 6


class TestTrainingGradientFlow:
    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_forward_backward_with_pooling_in_pipeline(self, normalize_embeddings: bool) -> None:
        # A checkpoint with baked-in pooling must remain finetunable: with grad enabled, the
        # clustering used to raise "Can't call numpy() on Tensor that requires grad" in forward.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        model.append(HierarchicalTokenPooling(pool_factor=2, normalize_embeddings=normalize_embeddings))
        model.train()
        features = model.preprocess(
            ["a fairly long document with plenty of distinct tokens to cluster together here"],
            task="document",
        )
        features = batch_to_device(features, model.device)
        num_tokens = features["input_ids"].shape[1]
        out = model(features, task="document")
        assert out["token_embeddings"].shape[1] < num_tokens, "pooling did not trigger"
        # Row 0 is the protected passthrough and carries gradient on its own, so backward through
        # the pooled rows only. This fails too if the cluster means are detached alongside the
        # clustering.
        out["token_embeddings"][:, 1:].sum().backward()
        grads = [p.grad for p in model[0].parameters() if p.grad is not None]
        assert grads, "no transformer parameter received a gradient"
        assert any(g.abs().sum() > 0 for g in grads), "no gradient reached the transformer through the cluster means"


class TestMaxTokensBudget:
    def test_long_documents_are_clustered_to_the_budget(self) -> None:
        pooling = HierarchicalTokenPooling(max_tokens=4)
        out = pooling.pool([_normed((12, 8)), _normed((20, 8))])
        # Exactly max_tokens output tokens, the protected token included.
        assert [e.shape for e in out] == [(4, 8), (4, 8)]

    def test_short_documents_pass_through_unchanged(self) -> None:
        pooling = HierarchicalTokenPooling(max_tokens=16)
        doc = _normed((10, 8))
        out = pooling.pool([doc])
        assert torch.allclose(out[0], doc, atol=1e-6)

    def test_budget_boundary_is_inclusive(self) -> None:
        pooling = HierarchicalTokenPooling(max_tokens=10)
        doc = _normed((10, 8))
        assert torch.allclose(pooling.pool([doc])[0], doc, atol=1e-6)
        pooled = pooling.pool([_normed((11, 8))])[0]
        assert 2 <= pooled.shape[0] <= 10
        assert pooled.shape[1] == 8

    def test_forward_applies_the_budget(self) -> None:
        pooling = HierarchicalTokenPooling(max_tokens=4)
        features = {
            "token_embeddings": torch.stack([_normed((12, 8)), _normed((12, 8))]),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling(features, task="document")
        assert out["token_embeddings"].shape == (2, 4, 8)
        assert out["attention_mask"].shape == (2, 4)

    def test_config_round_trip(self, tmp_path) -> None:
        HierarchicalTokenPooling(max_tokens=32, num_protected_tokens=0).save(str(tmp_path))
        restored = HierarchicalTokenPooling.load(str(tmp_path))
        assert restored.max_tokens == 32
        assert restored.pool_factor == 1
        assert restored.num_protected_tokens == 0


class TestAdaptivePooling:
    def test_redundant_documents_receive_fewer_tokens(self) -> None:
        repeated = torch.eye(8)[:1].repeat(8, 1)
        diverse = torch.eye(8)
        threshold = HierarchicalTokenPooling.calibrate_distance_threshold(
            [repeated, diverse], target_tokens=4.5, num_protected_tokens=0
        )
        assert isinstance(threshold, float)
        assert threshold == 0
        pooling = HierarchicalTokenPooling(distance_threshold=threshold, num_protected_tokens=0)
        assert [len(emb) for emb in pooling.pool([repeated, diverse])] == [1, 8]

    def test_calibration_matches_budget_with_protected_tokens(self, tmp_path) -> None:
        documents = [_normed((length, 8)) for length in (12, 20, 30)]
        threshold = HierarchicalTokenPooling.calibrate_distance_threshold(iter(documents), target_tokens=8)
        pooling = HierarchicalTokenPooling(distance_threshold=threshold, normalize_embeddings=False)
        pooled = pooling.pool(documents)
        assert sum(map(len, pooled)) == 24
        for original, compressed in zip(documents, pooled):
            assert torch.equal(original[0], compressed[0])
        output_path = tmp_path / "adaptive" / "pooling"
        pooling.save(str(output_path))
        restored = HierarchicalTokenPooling.load(str(output_path))
        assert restored.distance_threshold == pooling.distance_threshold
        assert all(torch.equal(a, b) for a, b in zip(pooled, restored.pool(documents)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_mask_and_gradient(self, dtype: torch.dtype) -> None:
        embeddings = _normed((2, 8, 8)).to(dtype).requires_grad_()
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[1, 5:] = False
        pooling = HierarchicalTokenPooling(distance_threshold=100)
        features = {"token_embeddings": embeddings, "attention_mask": mask}
        assert pooling(dict(features), task="query")["token_embeddings"] is embeddings
        pooled = pooling(dict(features), task="document")
        assert pooled["attention_mask"].sum(dim=1).tolist() == [2, 2]
        assert pooled["token_embeddings"].dtype == dtype
        pooled["token_embeddings"].sum().backward()
        assert torch.isfinite(embeddings.grad).all()
        assert embeddings.grad[1, 5:].count_nonzero() == 0
        assert embeddings.grad[1, 1:5].abs().sum() > 0

    def test_calibration_handles_empty_and_short_documents(self) -> None:
        documents = [torch.empty(0, 4), _normed((1, 4)), _normed((6, 4))]
        threshold = HierarchicalTokenPooling.calibrate_distance_threshold(documents, target_tokens=1)
        pooling = HierarchicalTokenPooling(distance_threshold=threshold)
        assert [len(emb) for emb in pooling.pool(documents)] == [0, 1, 2]
        for target in [sum(map(len, documents)) / len(documents), 100]:
            threshold = HierarchicalTokenPooling.calibrate_distance_threshold(documents, target_tokens=target)
            assert threshold is None
            no_compression = HierarchicalTokenPooling(distance_threshold=threshold)
            assert [len(emb) for emb in no_compression.pool(documents)] == [0, 1, 6]

    def test_calibration_ties_can_compress_below_target(self) -> None:
        documents = [torch.eye(4)[:1].repeat(10, 1)]
        threshold = HierarchicalTokenPooling.calibrate_distance_threshold(
            documents, target_tokens=5, num_protected_tokens=0
        )
        pooling = HierarchicalTokenPooling(distance_threshold=threshold, num_protected_tokens=0)
        assert len(pooling.pool(documents)[0]) == 1

    @pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
    def test_invalid_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="distance_threshold"):
            HierarchicalTokenPooling(distance_threshold=threshold)

    @pytest.mark.parametrize("kwargs", [{"pool_factor": 2}, {"max_tokens": 8}])
    def test_threshold_is_exclusive(self, kwargs) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            HierarchicalTokenPooling(distance_threshold=0.5, **kwargs)

    @pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
    def test_invalid_budget(self, target: float) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            HierarchicalTokenPooling.calibrate_distance_threshold([_normed((8, 4))], target_tokens=target)

    def test_impossible_budget_and_invalid_samples(self) -> None:
        with pytest.raises(ValueError, match="num_protected_tokens must be >= 0"):
            HierarchicalTokenPooling.calibrate_distance_threshold(
                [_normed((8, 4))], target_tokens=4, num_protected_tokens=-1
            )
        with pytest.raises(ValueError, match="at least 2"):
            HierarchicalTokenPooling.calibrate_distance_threshold([_normed((8, 4))], target_tokens=1)
        with pytest.raises(ValueError, match="at least one"):
            HierarchicalTokenPooling.calibrate_distance_threshold([], target_tokens=8)
        with pytest.raises(ValueError, match="2D"):
            HierarchicalTokenPooling.calibrate_distance_threshold([torch.ones(4)], target_tokens=8)


class TestStraightThroughGradients:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_forward_values_match_with_and_without_gradients(self, dtype: torch.dtype) -> None:
        embeddings = _normed((128, 128)).to(dtype).requires_grad_()
        pooling = HierarchicalTokenPooling(max_tokens=16)
        pooled = pooling.pool_one(embeddings)
        with torch.no_grad():
            reference = pooling.pool_one(embeddings)
        assert torch.equal(pooled, reference)

    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_each_member_receives_the_full_cluster_gradient(
        self, normalize_embeddings: bool, dtype: torch.dtype
    ) -> None:
        emb = _normed((5, 8)).to(dtype)
        mean_input = emb.clone().requires_grad_(True)
        ste_input = emb.clone().requires_grad_(True)

        mean_out = torch.cat([mean_input[:1], mean_input[1:].mean(dim=0, keepdim=True)])
        if normalize_embeddings:
            mean_out = torch.nn.functional.normalize(mean_out.float(), dim=-1).to(dtype)
        pooling = HierarchicalTokenPooling(pool_factor=4, normalize_embeddings=normalize_embeddings)
        ste_out = pooling.pool([ste_input])[0]
        assert mean_out.shape == (2, 8)
        assert torch.equal(mean_out, ste_out)

        mean_out[1].sum().backward()
        ste_out[1].sum().backward()
        assert mean_input.grad[1:].abs().sum() > 0
        assert torch.allclose(ste_input.grad[1:], 4 * mean_input.grad[1:], atol=1e-6)


class TestConstructorValidation:
    @pytest.mark.parametrize("bad", [0, -1, 0.5])
    def test_pool_factor_must_be_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="pool_factor must be >= 1"):
            HierarchicalTokenPooling(pool_factor=bad)

    def test_num_protected_tokens_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="num_protected_tokens must be >= 0"):
            HierarchicalTokenPooling(num_protected_tokens=-1)

    def test_pool_factor_and_max_tokens_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            HierarchicalTokenPooling(pool_factor=2, max_tokens=32)

    def test_max_tokens_must_exceed_protected_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must exceed num_protected_tokens"):
            HierarchicalTokenPooling(max_tokens=1, num_protected_tokens=1)

    def test_tasks_accepts_a_bare_string(self) -> None:
        # Without the coercion, list("query") would silently split into characters.
        assert HierarchicalTokenPooling(tasks="query").tasks == ["query"]

    def test_tasks_round_trip_through_config(self, tmp_path) -> None:
        HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"]).save(str(tmp_path))
        restored = HierarchicalTokenPooling.load(str(tmp_path))
        assert restored.tasks == ["query", "document"]
        assert restored.pool_factor == 2


class TestForwardEdgeCases:
    def test_forward_empty_batch(self) -> None:
        # A features dict with a 0-batch must round-trip without collapsing to 1D. Cover both a
        # subclass that routes through super().forward (Hierarchical at pool_factor>1) and one that
        # reaches BaseTokenPooling.forward directly (Lambda).
        for pooling in (HierarchicalTokenPooling(pool_factor=2), LambdaTokenPooling(pool_fn=lambda x: x)):
            features = {
                "token_embeddings": torch.empty((0, 5, 4)),
                "attention_mask": torch.empty((0, 5), dtype=torch.bool),
            }
            out = pooling.forward(features, task="document")
            assert out["token_embeddings"].shape == (0, 5, 4)
            assert out["attention_mask"].shape == (0, 5)

    def test_forward_fully_padded_row_in_batch(self) -> None:
        # A fully-padded (all-masked) row pools to zero tokens. The other rows pool normally and the
        # batch right-pads to the max pooled length.
        pooling = LambdaTokenPooling(pool_fn=lambda emb: emb[: max(emb.size(0) // 2, 1)] if emb.size(0) else emb)
        embeddings = torch.randn(3, 10, 4)
        attention_mask = torch.zeros(3, 10, dtype=torch.bool)
        # Row 0: fully padded (all False). Row 1 and 2: right-padded.
        attention_mask[1, :6] = True
        attention_mask[2, :4] = True
        out = pooling.forward({"token_embeddings": embeddings, "attention_mask": attention_mask}, task="document")
        # Row 0 → 0 tokens (identity of empty), row 1 → 3, row 2 → 2. Max len = 3, right-padded.
        assert out["attention_mask"].tolist() == [
            [False, False, False],
            [True, True, True],
            [True, True, False],
        ]


class TestBaseForwardVariableLength:
    def test_base_forward_via_lambda_pooling_variable_length(self) -> None:
        # Exercise BaseTokenPooling.forward via LambdaTokenPooling (HierarchicalTokenPooling.forward
        # short-circuits with a fast path, so the base's re-pad + mask reconstruction is untested
        # unless we route through Lambda).
        pooling = LambdaTokenPooling(pool_fn=lambda emb: emb[: max(emb.size(0) // 2, 1)])
        # Batch of two documents with different real lengths (mask=1 for the first 8 / 6 rows).
        embeddings = torch.randn(2, 10, 4)
        attention_mask = torch.zeros(2, 10, dtype=torch.bool)
        attention_mask[0, :8] = True
        attention_mask[1, :6] = True
        out = pooling.forward({"token_embeddings": embeddings, "attention_mask": attention_mask}, task="document")
        # Halving lengths: doc 0 -> 4 tokens, doc 1 -> 3 tokens. Padded to max=4.
        assert out["token_embeddings"].shape == (2, 4, 4)
        assert out["attention_mask"].tolist() == [
            [True, True, True, True],
            [True, True, True, False],
        ]


class TestUnbindAndPadEdge:
    def test_3d_input_left_padded_with_mask(self) -> None:
        # Left-padded 3D input WITH mask (mask should short-circuit padding_side detection).
        docs = [_normed((6, 4)), _normed((10, 4))]
        max_len = max(len(d) for d in docs)
        padded = torch.stack([torch.cat([torch.zeros(max_len - len(d), 4), d], dim=0) for d in docs])
        mask = torch.zeros(2, max_len, dtype=torch.bool)
        mask[0, -6:] = True
        mask[1, -10:] = True
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(padded, attention_mask=mask, padding_side="left")
        assert isinstance(out, Tensor)
        # Output re-padded on the specified side (left). Each doc pooled to ceil(t / 2).
        # Doc 0: 6 tokens -> 3 pooled. Doc 1: 10 tokens -> 5 pooled. Padded to max=5.
        assert out.shape == (2, 5, 4)
        # Left-padding preserved on the shorter row.
        assert torch.all(out[0, :2] == 0)

    def test_3d_input_empty_batch(self) -> None:
        # Batch=0 3D input must round-trip as a 3D empty tensor (not collapse to 1D).
        empty = torch.empty((0, 5, 4))
        pooling = HierarchicalTokenPooling(pool_factor=2)
        out = pooling.pool(empty)
        assert isinstance(out, Tensor)
        assert out.shape == (0, 0, 4)

    def test_3d_bad_padding_side_raises(self) -> None:
        padded = torch.randn(2, 6, 4)
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="padding_side must be"):
            pooling.pool(padded, padding_side="middle")

    def test_3d_bad_padding_side_raises_even_with_mask(self) -> None:
        # `_unbind_padded` short-circuits its own padding_side validation when a mask is passed,
        # so the pool() entry point must own the validation.
        padded = torch.randn(2, 6, 4)
        mask = torch.ones(2, 6, dtype=torch.bool)
        pooling = HierarchicalTokenPooling(pool_factor=2)
        with pytest.raises(ValueError, match="padding_side must be"):
            pooling.pool(padded, attention_mask=mask, padding_side="middle")

    def test_left_pad_middle_zero_preserved_no_mask(self) -> None:
        # Symmetric to test_3d_unbind_preserves_middle_zero_row but for left padding: the leading
        # zero row is padding (must drop). The middle zero row is a real all-zero embedding (must keep).
        emb = torch.stack(
            [
                torch.tensor([0.0, 0.0]),  # padding at the left
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.0, 0.0]),  # zero-valued but "real"
                torch.tensor([0.5, 0.5]),
            ]
        )
        padded = emb.unsqueeze(0)  # (1, 4, 2)
        out = LambdaTokenPooling(pool_fn=lambda x: x).pool(padded, padding_side="left")
        # Expected: 3 real rows kept, left-padded so leading position stays zero.
        assert out.shape == (1, 3, 2)


def test_hierarchical_output_normalization_includes_protected_tokens(tmp_path):
    embeddings = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    embeddings[0] *= 0.5
    raw = HierarchicalTokenPooling(3, normalize_embeddings=False).pool([embeddings])[0]
    pooling = HierarchicalTokenPooling(3)
    normalized = pooling.pool([embeddings])[0]
    torch.testing.assert_close(normalized, torch.nn.functional.normalize(raw, dim=-1))
    torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(len(normalized)))
    pooling.save(str(tmp_path))
    restored = HierarchicalTokenPooling.load(str(tmp_path))
    torch.testing.assert_close(restored.pool([embeddings])[0], normalized)
    assert restored.normalize_embeddings


@pytest.mark.parametrize("normalize_embeddings", [None, False, True])
def test_hierarchical_normalization_config_compatibility(tmp_path, normalize_embeddings: bool | None) -> None:
    config = {"pool_factor": 3, "num_protected_tokens": 1, "tasks": ["document"]}
    if normalize_embeddings is not None:
        config["normalize_embeddings"] = normalize_embeddings
    config_path = tmp_path / HierarchicalTokenPooling.config_file_name
    config_path.write_text(json.dumps(config), encoding="utf-8")
    restored = HierarchicalTokenPooling.load(str(tmp_path))
    assert restored.normalize_embeddings is bool(normalize_embeddings)
    embeddings = _normed((12, 8))
    expected = HierarchicalTokenPooling(3, normalize_embeddings=bool(normalize_embeddings)).pool([embeddings])[0]
    torch.testing.assert_close(restored.pool([embeddings])[0], expected, rtol=0, atol=0)
    restored.save(str(tmp_path))
    assert json.loads(config_path.read_text())["normalize_embeddings"] is bool(normalize_embeddings)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pool_factor,num_tokens,num_protected_tokens", [(1, 4, 1), (2, 4, 4), (2, 2, 1), (2, 0, 1)])
def test_hierarchical_normalization_without_merging(
    dtype: torch.dtype, pool_factor: float, num_tokens: int, num_protected_tokens: int
) -> None:
    embeddings = torch.randn(num_tokens, 8, dtype=dtype)
    if num_tokens:
        embeddings[0] = 0
    original = embeddings.clone()
    pooling = HierarchicalTokenPooling(pool_factor, num_protected_tokens=num_protected_tokens)
    reference = embeddings.float() if dtype in (torch.float16, torch.bfloat16) else embeddings
    expected = torch.nn.functional.normalize(reference, dim=-1).to(dtype)
    torch.testing.assert_close(pooling.pool([embeddings])[0], expected, rtol=0, atol=0)
    features = {
        "token_embeddings": embeddings.unsqueeze(0),
        "attention_mask": torch.ones(1, num_tokens, dtype=torch.bool),
    }
    output = pooling(features, task="document")
    torch.testing.assert_close(output["token_embeddings"][0], expected, rtol=0, atol=0)
    assert output["attention_mask"].shape == (1, num_tokens)
    torch.testing.assert_close(embeddings, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_hierarchical_normalization_zero_centroid(dtype: torch.dtype) -> None:
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=dtype)
    pooling = HierarchicalTokenPooling(4, num_protected_tokens=0, normalize_embeddings=True)
    torch.testing.assert_close(pooling.pool([embeddings])[0], torch.zeros(1, 2, dtype=dtype))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty_batch", [False, True])
def test_hierarchical_batched_normalization_without_pooling(monkeypatch, device, dtype, empty_batch) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    embeddings = torch.randn(3, 5, 8, device=device, dtype=dtype)
    embeddings[0, 0] = 0
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]], device=device, dtype=torch.bool)
    if empty_batch:
        embeddings, mask = embeddings[:0], mask[:0]
    original = embeddings.clone()
    pooling = HierarchicalTokenPooling(1)
    expected = pooling.pool([embedding[m] for embedding, m in zip(embeddings, mask)])

    def fail_per_document_pooling(*args, **kwargs):
        pytest.fail("No-pooling forward must normalize the batch without per-document pooling")

    monkeypatch.setattr(pooling, "pool_one", fail_per_document_pooling)
    features = {"token_embeddings": embeddings, "attention_mask": mask}
    output = pooling(features, task="document")
    assert output is features
    assert output["attention_mask"] is mask
    assert output["token_embeddings"].shape == embeddings.shape
    assert output["token_embeddings"].dtype == dtype
    for actual, m, reference in zip(output["token_embeddings"], mask, expected):
        torch.testing.assert_close(actual[m], reference)
    torch.testing.assert_close(embeddings, original, rtol=0, atol=0)


@pytest.mark.parametrize("normalize_embeddings", [None, False, True])
def test_pooling_encode_and_saved_pipeline(tmp_path, normalize_embeddings: bool | None) -> None:
    pooling = (
        HierarchicalTokenPooling(3)
        if normalize_embeddings
        else HierarchicalTokenPooling(3, normalize_embeddings=False)
    )
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    texts = ["A long enough document with several tokens to cluster into a smaller representation."]
    query = model.encode_query(texts)
    expected = model.encode_document(texts, token_pooling=pooling)
    if normalize_embeddings:
        torch.testing.assert_close(expected[0].norm(dim=-1), expected[0].new_ones(len(expected[0])))
    model.append(pooling)
    torch.testing.assert_close(model.encode_document(texts)[0], expected[0])
    torch.testing.assert_close(model.encode_query(texts)[0], query[0])
    model.save_pretrained(str(tmp_path))
    if normalize_embeddings is None:
        modules = json.loads((tmp_path / "modules.json").read_text())
        config_path = tmp_path / modules[-1]["path"] / HierarchicalTokenPooling.config_file_name
        config = json.loads(config_path.read_text())
        del config["normalize_embeddings"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
    restored = MultiVectorEncoder(str(tmp_path))
    assert restored[-1].normalize_embeddings is bool(normalize_embeddings)
    torch.testing.assert_close(restored.encode_document(texts)[0], expected[0])
