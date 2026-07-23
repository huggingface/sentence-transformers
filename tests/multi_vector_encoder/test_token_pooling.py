from __future__ import annotations

import pickle
from itertools import combinations

import pytest
import torch
from torch import Tensor

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import (
    HierarchicalTokenPooling,
    LambdaTokenPooling,
)


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
        out = LambdaTokenPooling(pool_func=lambda x: x).pool(padded, padding_side="right")
        # Expected: real rows [0..2] preserved (including middle zero). Trailing pad row dropped.
        assert out.shape == (1, 3, 2)


class TestHierarchicalTokenPooling:
    def test_class_pool_matches_module_helper_directly(self) -> None:
        # The pool() class method and the internal _hierarchical_pool_one produce identical output
        # when given the same input (this pins the wiring between the public API and the math).
        from sentence_transformers.multi_vector_encoder.modules.token_pooling import _hierarchical_pool_one

        emb = _normed((15, 8))
        direct = _hierarchical_pool_one(emb, pool_factor=2, protected_tokens=1)
        via_pool = HierarchicalTokenPooling(pool_factor=2, protected_tokens=1).pool([emb])[0]
        assert torch.allclose(direct, via_pool)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_pooling_supports_low_precision_dtypes(self, dtype: torch.dtype) -> None:
        # The ColPali/ColQwen2 family runs bf16 by default, and numpy has no bfloat16: the scipy
        # distance step must run in fp32 while the pooled output keeps the input dtype.
        emb = _normed((15, 8)).to(dtype)
        out = HierarchicalTokenPooling(pool_factor=2, protected_tokens=1).pool([emb])[0]
        assert out.dtype == dtype
        # Same values in fp32 must yield the same clustering: only the mean accumulation dtype differs.
        reference = HierarchicalTokenPooling(pool_factor=2, protected_tokens=1).pool([emb.float()])[0]
        assert out.shape == reference.shape
        assert torch.allclose(out.float(), reference, atol=1e-2)

    def test_no_op_when_pool_factor_1(self) -> None:
        docs = [_normed((10, 4))]
        pooling = HierarchicalTokenPooling(pool_factor=1)
        # Pipeline forward path is a fast no-op.
        features = {
            "token_embeddings": docs[0].unsqueeze(0),
            "attention_mask": torch.ones(1, 10, dtype=torch.bool),
        }
        out = pooling.forward(features, task="document")
        assert out is features  # returned unchanged (fast path)

    def test_protected_tokens_untouched(self) -> None:
        emb = _normed((12, 8))
        out = HierarchicalTokenPooling(pool_factor=2, protected_tokens=2).pool([emb])[0]
        # First 2 rows are the protected tokens verbatim.
        assert torch.allclose(out[:2], emb[:2])

    def test_pooled_rows_are_cluster_means(self) -> None:
        # Every non-protected row of the output must equal the mean of some non-empty subset of the
        # non-protected input rows, and the subsets must partition the non-protected input rows.
        # Verifies the scatter_add + dense-remap math.
        torch.manual_seed(1)
        emb = _normed((7, 4))  # 6 non-protected rows -> brute-force over 2^6 subsets is trivial.
        protected_tokens = 1
        pool_factor = 3
        pooling = HierarchicalTokenPooling(pool_factor=pool_factor, protected_tokens=protected_tokens)
        out = pooling.pool([emb])[0]
        pooled_rows = out[protected_tokens:]
        source_rows = emb[protected_tokens:]
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
    def test_applies_pool_func_per_sample(self) -> None:
        def halve(emb: Tensor) -> Tensor:
            n = emb.size(0)
            return emb[: n - n % 2].view(-1, 2, emb.size(-1)).mean(dim=1)

        docs = [_normed((10, 4)), _normed((6, 4))]
        out = LambdaTokenPooling(pool_func=halve).pool(docs)
        assert [e.shape for e in out] == [(5, 4), (3, 4)]

    def test_save_raises_not_implemented(self, tmp_path) -> None:
        pooling = LambdaTokenPooling(pool_func=lambda x: x)
        with pytest.raises(NotImplementedError, match="cannot be saved"):
            pooling.save(str(tmp_path))


def _keep_first_two(emb: Tensor) -> Tensor:
    """Module-level (pickle-friendly) pool function used by the pickle smoke test."""
    return emb[:2]


class TestPicklability:
    def test_hierarchical_pooling_round_trips_through_pickle(self) -> None:
        # Ensures `pool={...}` (multi-process encode) works with a saveable pooling.
        pooling = HierarchicalTokenPooling(pool_factor=3, protected_tokens=1)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, HierarchicalTokenPooling)
        assert restored.pool_factor == 3 and restored.protected_tokens == 1
        emb = _normed((12, 8))
        assert torch.allclose(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_top_level_func_pickles(self) -> None:
        # LambdaTokenPooling survives pickle as long as `pool_func` is picklable itself.
        pooling = LambdaTokenPooling(pool_func=_keep_first_two)
        restored = pickle.loads(pickle.dumps(pooling))
        assert isinstance(restored, LambdaTokenPooling)
        emb = _normed((10, 4))
        assert torch.equal(pooling.pool([emb])[0], restored.pool([emb])[0])

    def test_lambda_pooling_with_lambda_does_not_pickle(self) -> None:
        # Anonymous lambdas / nested functions won't pickle. Confirms the docstring caveat.
        pooling = LambdaTokenPooling(pool_func=lambda x: x)
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

    def test_forward_skips_queries(self) -> None:
        pooling = HierarchicalTokenPooling(pool_factor=2)
        embeddings = _normed((2, 12, 8))
        features = {
            "token_embeddings": embeddings,
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"] is embeddings  # unchanged reference

    def test_forward_pools_queries_when_opted_in(self) -> None:
        # CRISP-style strategies compress queries too: pool_queries=True lifts the query gate.
        pooling = HierarchicalTokenPooling(pool_factor=2, pool_queries=True)
        features = {
            "token_embeddings": _normed((2, 12, 8)),
            "attention_mask": torch.ones(2, 12, dtype=torch.bool),
        }
        out = pooling.forward(features, task="query")
        assert out["token_embeddings"].shape[1] == 6


class TestEncodeWithPooling:
    def test_encode_document_with_pooling_kwarg(self) -> None:
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long document with plenty of distinct tokens for clustering here"
        without = model.encode_document([text], convert_to_tensor=True)[0]
        with_pooling = model.encode_document(
            [text],
            pooling=HierarchicalTokenPooling(pool_factor=2),
            convert_to_tensor=True,
        )[0]
        assert with_pooling.shape[0] < without.shape[0]

    def test_encode_document_with_lambda_pooling(self) -> None:
        def keep_first_two(emb: Tensor) -> Tensor:
            return emb[:2]

        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "some longer text with multiple tokens"
        out = model.encode_document(
            [text],
            pooling=LambdaTokenPooling(pool_func=keep_first_two),
            convert_to_tensor=True,
        )[0]
        assert out.shape[0] == 2

    def test_pooling_skips_queries(self) -> None:
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long query text with many tokens"
        without = model.encode_query([text], convert_to_tensor=True)[0]
        with_pooling = model.encode_query(
            [text],
            pooling=HierarchicalTokenPooling(pool_factor=2),
            convert_to_tensor=True,
        )[0]
        # Queries pass through unchanged: shape AND values must match (guards against silent mutation).
        assert with_pooling.shape == without.shape
        assert torch.equal(with_pooling, without)

    def test_pooling_pools_queries_when_opted_in(self) -> None:
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        text = "a fairly long query text with many tokens"
        without = model.encode_query([text], convert_to_tensor=True)[0]
        pooled = model.encode_query(
            [text],
            pooling=HierarchicalTokenPooling(pool_factor=2, pool_queries=True),
            convert_to_tensor=True,
        )[0]
        assert pooled.shape[0] < without.shape[0]

    def test_encode_document_with_pooling_and_padded_tensor(self) -> None:
        # Compose the pooling kwarg with convert_to_padded_tensor=True so the batch is re-padded
        # to 3D after pooling. Verifies the two independent code paths cooperate.
        model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        texts = [
            "a fairly long document with plenty of distinct tokens for clustering here",
            "a much shorter text",
        ]
        without = model.encode_document(texts, convert_to_padded_tensor=True)
        with_pooling = model.encode_document(
            texts,
            pooling=HierarchicalTokenPooling(pool_factor=2),
            convert_to_padded_tensor=True,
        )
        assert with_pooling.dim() == 3 and with_pooling.shape[0] == 2
        # Pooled output has strictly fewer tokens per doc than the un-pooled version.
        assert with_pooling.shape[1] < without.shape[1]


class TestConstructorValidation:
    @pytest.mark.parametrize("bad", [0, -1])
    def test_pool_factor_must_be_positive(self, bad: int) -> None:
        with pytest.raises(ValueError, match="pool_factor must be >= 1"):
            HierarchicalTokenPooling(pool_factor=bad)

    def test_protected_tokens_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="protected_tokens must be >= 0"):
            HierarchicalTokenPooling(protected_tokens=-1)

    def test_pool_queries_round_trips_through_config(self, tmp_path) -> None:
        HierarchicalTokenPooling(pool_factor=2, pool_queries=True).save(str(tmp_path))
        restored = HierarchicalTokenPooling.load(str(tmp_path))
        assert restored.pool_queries is True
        assert restored.pool_factor == 2


class TestForwardEdgeCases:
    def test_forward_empty_batch(self) -> None:
        # A features dict with a 0-batch must round-trip without collapsing to 1D. Cover both a
        # subclass that routes through super().forward (Hierarchical at pool_factor>1) and one that
        # reaches BaseTokenPooling.forward directly (Lambda).
        for pooling in (HierarchicalTokenPooling(pool_factor=2), LambdaTokenPooling(pool_func=lambda x: x)):
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
        pooling = LambdaTokenPooling(pool_func=lambda emb: emb[: max(emb.size(0) // 2, 1)] if emb.size(0) else emb)
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
        pooling = LambdaTokenPooling(pool_func=lambda emb: emb[: max(emb.size(0) // 2, 1)])
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
        out = LambdaTokenPooling(pool_func=lambda x: x).pool(padded, padding_side="left")
        # Expected: 3 real rows kept, left-padded so leading position stays zero.
        assert out.shape == (1, 3, 2)
