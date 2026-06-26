import contextlib

import compiled
import pytest
import torch

from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal

BUCKET_EXCEEDING_MAX_SEQ_LENGTH = 999_999


def _load(model_name: str, **kwargs) -> compiled.SentenceTransformer:
    """
    Load a compiled model on CPU, without calling compile_and_warm_up.
    """
    kwargs.setdefault(
        "compiled_token_buckets", (*compiled.DEFAULT_COMPILED_TOKEN_BUCKETS, BUCKET_EXCEEDING_MAX_SEQ_LENGTH)
    )
    return compiled.SentenceTransformer(model_name, device="cpu", compiled_batch_size=1, **kwargs)


def _nearest_bucket(buckets: tuple[int, ...], num_tokens: int) -> int:
    """
    The bucket _pad_to_bucket would pad to, or num_tokens itself when it exceeds every bucket.
    """
    return next((bucket for bucket in buckets if bucket >= num_tokens), num_tokens)


def _encodings(num_tokens: int, *, batch_size: int = 1, **extra) -> dict:
    return {
        "input_ids": torch.zeros(batch_size, num_tokens, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, num_tokens, dtype=torch.long),
        **extra,
    }


@contextlib.contextmanager
def _training_mode(model: compiled.SentenceTransformer, training: bool):
    was_training = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(was_training)


@pytest.fixture(
    scope="module",
    params=[
        "sentence-transformers/all-MiniLM-L6-v2",  # WordPiece (BERT), has token_type_ids, max_seq 256
        "Alibaba-NLP/gte-modernbert-base",  # BPE (ModernBERT), no token_type_ids, max_seq 8192
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # SentencePiece (XLM-R), max_seq 128
    ],
)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_name: str) -> compiled.SentenceTransformer:
    """
    Shared, read-only compiled model (not warmed up). Tests that mutate it should _load a fresh one.
    """
    return _load(model_name)


@pytest.fixture
def baseline_matmul_precision():
    """
    Set the global precision to something different from _COMPILED_MATMUL_PRECISION so tests can verify the
    compiled paths flip it and restore it on exit.
    """
    baseline: compiled.MatmulPrecision = "highest" if compiled._COMPILED_MATMUL_PRECISION != "highest" else "medium"
    with compiled._set_float32_matmul_precision(baseline):
        yield baseline


class TestInit:
    def test_clamps_buckets_to_max_seq_length(self, model: compiled.SentenceTransformer) -> None:
        input_buckets = (*compiled.DEFAULT_COMPILED_TOKEN_BUCKETS, BUCKET_EXCEEDING_MAX_SEQ_LENGTH)
        expected = tuple(sorted({min(bucket, model.max_seq_length) for bucket in input_buckets}))
        assert model._compiled_token_buckets == expected
        assert BUCKET_EXCEEDING_MAX_SEQ_LENGTH not in model._compiled_token_buckets
        assert all(bucket <= model.max_seq_length for bucket in model._compiled_token_buckets)
        # The oversized bucket gets clamped in (not dropped), so max_seq_length itself becomes a bucket.
        assert max(model._compiled_token_buckets) == model.max_seq_length

    def test_raises_when_no_buckets(self, model_name: str) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _load(model_name, compiled_token_buckets=())


class TestPadToBucket:
    @pytest.mark.parametrize("num_tokens", [50, 100, 200])
    def test_pads_to_nearest_bucket(self, model: compiled.SentenceTransformer, num_tokens: int) -> None:
        expected = _nearest_bucket(model._compiled_token_buckets, num_tokens)
        result = model._pad_to_bucket(_encodings(num_tokens))
        assert result["input_ids"].shape[1] == expected
        assert result["attention_mask"].shape[1] == expected

    def test_pads_token_type_ids_when_present(self, model: compiled.SentenceTransformer) -> None:
        expected = _nearest_bucket(model._compiled_token_buckets, 50)
        result = model._pad_to_bucket(_encodings(50, token_type_ids=torch.zeros(1, 50, dtype=torch.long)))
        assert result["token_type_ids"].shape[1] == expected

    def test_no_padding_when_exceeding_all_buckets(self, model: compiled.SentenceTransformer) -> None:
        num_tokens = BUCKET_EXCEEDING_MAX_SEQ_LENGTH
        result = model._pad_to_bucket(_encodings(num_tokens))
        assert result["input_ids"].shape[1] == num_tokens

    def test_batch_size_mismatch_skips_padding(self, model: compiled.SentenceTransformer) -> None:
        result = model._pad_to_bucket(_encodings(50, batch_size=2))
        assert result["input_ids"].shape[1] == 50

    def test_skips_when_attention_mask_missing(self, model: compiled.SentenceTransformer) -> None:
        # The flash-attention "flattened" path drops the 2D attention_mask; we must not try to bucket it.
        result = model._pad_to_bucket({"input_ids": torch.zeros(1, 50, dtype=torch.long)})
        assert result["input_ids"].shape[1] == 50

    def test_skips_when_input_ids_not_2d(self, model: compiled.SentenceTransformer) -> None:
        result = model._pad_to_bucket(
            {"input_ids": torch.zeros(50, dtype=torch.long), "attention_mask": torch.zeros(50, dtype=torch.long)}
        )
        assert result["input_ids"].dim() == 1

    def test_passes_through_non_tensor_metadata(self, model: compiled.SentenceTransformer) -> None:
        expected = _nearest_bucket(model._compiled_token_buckets, 50)
        result = model._pad_to_bucket(_encodings(50, modality="text", prompt_length=None))
        assert result["input_ids"].shape[1] == expected
        assert result["modality"] == "text"
        assert result["prompt_length"] is None

    def test_raises_on_unexpected_key(self, model: compiled.SentenceTransformer) -> None:
        with pytest.raises(ValueError, match="Unexpected encoding keys"):
            model._pad_to_bucket(_encodings(50, surprise=torch.zeros(1, 50)))


class TestTokenize:
    def test_pads_real_text_to_nearest_bucket(self, model: compiled.SentenceTransformer) -> None:
        # encode() tokenizes via preprocess on ST >= 5.3 and tokenize before that; exercise whichever it uses.
        text = compiled._create_text_with_num_tokens(100, model._tokenize_unpadded)
        padded = model.preprocess([text]) if compiled._ENCODE_USES_PREPROCESS else model.tokenize([text])
        assert padded["input_ids"].shape[1] == _nearest_bucket(model._compiled_token_buckets, 100)


class TestCreateTextWithNumTokens:
    # Awkward non-round counts (to catch off-by-one / rounding bugs the round bucket sizes would miss) plus the
    # actual buckets (the values compile_and_warm_up feeds to _create_text_with_num_tokens).
    @pytest.mark.parametrize("target_num_tokens", [8, 13, 47, 199, *compiled.DEFAULT_COMPILED_TOKEN_BUCKETS])
    def test_lands_exactly_on_target(self, model: compiled.SentenceTransformer, target_num_tokens: int) -> None:
        if target_num_tokens > model.max_seq_length:
            pytest.skip(f"target {target_num_tokens} > max_seq {model.max_seq_length}")
        text = compiled._create_text_with_num_tokens(target_num_tokens, model._tokenize_unpadded)
        assert model._tokenize_unpadded([text])["input_ids"].shape[1] == target_num_tokens

    @pytest.mark.parametrize("offset", [-1, 0])
    def test_lands_at_max_seq_boundary(self, model: compiled.SentenceTransformer, offset: int) -> None:
        target_num_tokens = model.max_seq_length + offset
        text = compiled._create_text_with_num_tokens(target_num_tokens, model._tokenize_unpadded)
        assert model._tokenize_unpadded([text])["input_ids"].shape[1] == target_num_tokens


class TestCompileAndWarmUp:
    def test_uses_compiled_precision_and_restores_global(
        self, model_name: str, monkeypatch: pytest.MonkeyPatch, baseline_matmul_precision: str
    ) -> None:
        model = _load(model_name)
        observed_precision: list[str] = []
        monkeypatch.setattr(
            model, "encode", lambda *a, **k: observed_precision.append(torch.get_float32_matmul_precision())
        )
        monkeypatch.setattr(torch, "compile", lambda *a, **k: None)

        model.compile_and_warm_up()

        assert observed_precision
        assert all(p == compiled._COMPILED_MATMUL_PRECISION for p in observed_precision)
        assert torch.get_float32_matmul_precision() == baseline_matmul_precision

    @pytest.mark.parametrize("compile_fallback", [True, False])
    def test_compiles_fallback_only_when_enabled(
        self, model_name: str, compile_fallback: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _load(model_name, compile_fallback=compile_fallback)
        compile_calls: list[dict] = []
        monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compile_calls.append(kwargs) or fn)
        monkeypatch.setattr(model, "encode", lambda *a, **k: None)

        model.compile_and_warm_up()

        assert (model._compiled_forward_dynamic is not None) == compile_fallback
        # One compile for the bucketed CUDA-graph forward, plus one for the dynamic fallback when enabled.
        assert len(compile_calls) == (2 if compile_fallback else 1)


class TestForward:
    def test_raises_in_training_mode(self, model: compiled.SentenceTransformer) -> None:
        with _training_mode(model, training=True):
            with pytest.raises(ValueError, match="training"):
                model.forward(_encodings(64))

    def test_raises_when_compile_and_warm_up_not_called(self, model: compiled.SentenceTransformer) -> None:
        with _training_mode(model, training=False):
            # 64 is a bucket -> the compiled (CUDA-graph) path, which must be warmed up.
            with pytest.raises(ValueError, match="compile_and_warm_up"):
                model.forward(_encodings(64))

    def test_fallback_raises_when_not_warmed(self, model: compiled.SentenceTransformer) -> None:
        with _training_mode(model, training=False):
            # 100 is not a bucket -> the (dynamic-compiled) fallback, which also requires warm-up.
            with pytest.raises(ValueError, match="compile_and_warm_up"):
                model.forward(_encodings(100))

    def test_fallback_runs_eager_when_disabled(self, model_name: str) -> None:
        model = _load(model_name, compile_fallback=False)
        with _training_mode(model, training=False):
            # Out-of-bucket length with compile_fallback=False runs eager super().forward, no warm-up needed.
            out = model.forward(_encodings(100))
        assert "sentence_embedding" in out

    def test_encode_uses_compiled_precision_and_restores_global(
        self, model_name: str, monkeypatch: pytest.MonkeyPatch, baseline_matmul_precision: str
    ) -> None:
        model = _load(model_name)
        observed_precision: list[str] = []

        def forward_with_precision_check(input: dict[str, torch.Tensor], **kwargs):
            observed_precision.append(torch.get_float32_matmul_precision())
            return SentenceTransformerOriginal.forward(model, input, **kwargs)

        monkeypatch.setattr(model, "_compiled_forward", forward_with_precision_check)

        model.encode("hello", show_progress_bar=False)

        assert observed_precision == [compiled._COMPILED_MATMUL_PRECISION]
        assert torch.get_float32_matmul_precision() == baseline_matmul_precision


class TestEncode:
    @pytest.mark.parametrize(
        ("encode_kwargs", "expected_tokenize_batch_sizes"),
        [
            # No batch_size -> defaults to _compiled_batch_size=1, so 3 texts tokenize individually.
            ({}, [1, 1, 1]),
            # User-provided batch_size is respected, so all 3 texts tokenize together.
            ({"batch_size": 3}, [3]),
        ],
    )
    def test_batch_size_kwarg(
        self,
        model: compiled.SentenceTransformer,
        monkeypatch: pytest.MonkeyPatch,
        encode_kwargs: dict,
        expected_tokenize_batch_sizes: list[int],
    ) -> None:
        # encode() tokenizes via preprocess on ST >= 5.3 and tokenize before that; capture whichever it uses.
        capture_attr = "preprocess" if compiled._ENCODE_USES_PREPROCESS else "tokenize"
        original = getattr(model, capture_attr)
        tokenize_batch_sizes: list[int] = []

        def capture(inputs, *args, **kwargs):
            tokenize_batch_sizes.append(len(inputs))
            return original(inputs, *args, **kwargs)

        monkeypatch.setattr(model, capture_attr, capture)
        # Route both the bucket-match and fallback paths through eager so encode completes on CPU.
        eager = lambda input, **kwargs: SentenceTransformerOriginal.forward(model, input, **kwargs)  # noqa: E731
        monkeypatch.setattr(model, "_compiled_forward", eager)
        monkeypatch.setattr(model, "_compiled_forward_dynamic", eager)

        model.encode(["hello", "world", "foo"], show_progress_bar=False, **encode_kwargs)

        assert tokenize_batch_sizes == expected_tokenize_batch_sizes
