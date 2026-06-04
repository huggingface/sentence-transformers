from __future__ import annotations

import torch

from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoderTokenEncoder, SparseTokenPooling


def _make_token_encoder() -> SparseAutoEncoderTokenEncoder:
    torch.manual_seed(0)
    encoder = SparseAutoEncoderTokenEncoder(input_dim=4, hidden_dim=8, k=2)
    with torch.no_grad():
        encoder.W_enc.normal_()
        encoder.b_enc.normal_()
        encoder.b_pre.normal_()
        encoder.data_mean = torch.randn(4)
    return encoder


def _copy_encoder(
    encoder: SparseAutoEncoderTokenEncoder, *, token_batch_size: int | None = None
) -> SparseAutoEncoderTokenEncoder:
    copied = SparseAutoEncoderTokenEncoder(
        input_dim=encoder.input_dim,
        hidden_dim=encoder.hidden_dim,
        k=encoder.k,
        variant=encoder.variant,
        token_batch_size=token_batch_size,
    )
    if encoder.data_mean is not None:
        copied.data_mean = encoder.data_mean.detach().clone()
    copied.load_state_dict(encoder.state_dict())
    copied.rms_scale = encoder.rms_scale
    return copied


def _clone_features(features):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in features.items()}


def _dense_pool_rows(features) -> int:
    pooler = Pooling(4, pooling_mode="mean")
    out = pooler(_clone_features(features))
    return out["sentence_embedding"].shape[0]


def _pool_expected(
    encoder: SparseAutoEncoderTokenEncoder,
    hidden: torch.Tensor,
    pooling_strategy: str,
    token_mask: torch.Tensor | None = None,
    prompt_length: int = 0,
) -> torch.Tensor:
    if prompt_length:
        hidden = hidden[prompt_length:]
        if token_mask is not None:
            token_mask = token_mask[prompt_length:]
    values, indices = encoder.encode_tokens(hidden)
    flat_values = values.reshape(-1)
    flat_indices = indices.reshape(-1)
    active = flat_values > 0
    if token_mask is not None:
        token_active = token_mask.reshape(-1, 1).expand_as(values).reshape(-1).to(torch.bool)
        active = active & token_active
    pooled = torch.zeros(encoder.hidden_dim, dtype=flat_values.dtype, device=flat_values.device)
    if not active.any():
        return pooled
    if pooling_strategy == "max_log1p":
        pooled.scatter_reduce_(0, flat_indices[active], torch.log1p(flat_values[active]), reduce="amax")
    elif pooling_strategy == "max":
        pooled.scatter_reduce_(0, flat_indices[active], flat_values[active], reduce="amax")
    else:
        pooled.scatter_add_(0, flat_indices[active], flat_values[active])
    return pooled


def _run_modules(
    features, encoder: SparseAutoEncoderTokenEncoder, pooling_strategy: str, include_prompt: bool = True
) -> torch.Tensor:
    pooler = SparseTokenPooling(
        embedding_dimension=encoder.hidden_dim, pooling_strategy=pooling_strategy, include_prompt=include_prompt
    )
    out = encoder(features)
    out = pooler(out)
    return out["sentence_embedding"]


def test_sparse_auto_encoder_token_pooling_matches_direct_pooling_padded() -> None:
    encoder = _make_token_encoder()
    hidden = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    features = {
        "token_embeddings": hidden.clone(),
        "attention_mask": mask,
    }
    assert _dense_pool_rows(features) == 2

    for pooling_strategy in ("sum", "max", "max_log1p"):
        actual = _run_modules(features.copy(), encoder, pooling_strategy)
        expected = torch.stack(
            [
                _pool_expected(encoder, hidden[0], pooling_strategy=pooling_strategy, token_mask=mask[0]),
                _pool_expected(encoder, hidden[1], pooling_strategy=pooling_strategy, token_mask=mask[1]),
            ]
        ).to(actual.dtype)
        torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_pooling_matches_dense_contract_padded_without_attention_mask() -> None:
    encoder = _make_token_encoder()
    hidden = torch.randn(2, 4, 4)
    features = {"token_embeddings": hidden.clone()}
    assert _dense_pool_rows(features) == 2

    actual = _run_modules(features, encoder, "sum")
    expected = torch.stack(
        [
            _pool_expected(encoder, hidden[0], pooling_strategy="sum"),
            _pool_expected(encoder, hidden[1], pooling_strategy="sum"),
        ]
    ).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_pooling_matches_dense_contract_padded_without_prompt() -> None:
    encoder = _make_token_encoder()
    hidden = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    features = {
        "token_embeddings": hidden.clone(),
        "attention_mask": mask,
        "prompt_length": 1,
    }
    dense_pooler = Pooling(4, pooling_mode="mean", include_prompt=False)
    assert dense_pooler(_clone_features(features))["sentence_embedding"].shape[0] == 2

    actual = _run_modules(features, encoder, "sum", include_prompt=False)
    expected = torch.stack(
        [
            _pool_expected(encoder, hidden[0], pooling_strategy="sum", token_mask=mask[0], prompt_length=1),
            _pool_expected(encoder, hidden[1], pooling_strategy="sum", token_mask=mask[1], prompt_length=1),
        ]
    ).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_sparse_token_pooling_excludes_prompt_after_left_padding() -> None:
    token_values = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            [[6.0], [7.0], [8.0], [9.0], [10.0]],
        ]
    )
    token_indices = torch.tensor(
        [
            [[0], [1], [2], [3], [4]],
            [[0], [1], [2], [3], [4]],
        ]
    )
    features = {
        "token_sparse_values": token_values,
        "token_sparse_indices": token_indices,
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]]),
        "prompt_length": 1,
    }

    actual = SparseTokenPooling(embedding_dimension=5, include_prompt=False)(features)["sentence_embedding"]
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 5.0],
            [0.0, 0.0, 8.0, 9.0, 10.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_pooling_matches_direct_pooling_flattened_rank3() -> None:
    encoder = _make_token_encoder()
    doc_a = torch.randn(3, 4)
    doc_b = torch.randn(2, 4)
    packed = torch.cat([doc_a, doc_b], dim=0).unsqueeze(0)
    features = {
        "token_embeddings": packed,
        "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
        "seq_idx": torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.int32),
    }
    assert _dense_pool_rows(features) == 2

    for pooling_strategy in ("sum", "max", "max_log1p"):
        actual = _run_modules(features, encoder, pooling_strategy)
        expected = torch.stack(
            [
                _pool_expected(encoder, doc_a, pooling_strategy=pooling_strategy),
                _pool_expected(encoder, doc_b, pooling_strategy=pooling_strategy),
            ]
        ).to(actual.dtype)
        torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_pooling_matches_dense_contract_flattened_rank2() -> None:
    encoder = _make_token_encoder()
    doc_a = torch.randn(3, 4)
    doc_b = torch.randn(2, 4)
    packed = torch.cat([doc_a, doc_b], dim=0)
    features = {
        "token_embeddings": packed,
        "cu_seq_lens_q": torch.tensor([0, 3, 5], dtype=torch.int32),
        "seq_idx": torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
    }
    assert _dense_pool_rows(features) == 2

    actual = _run_modules(features, encoder, "sum")
    expected = torch.stack(
        [
            _pool_expected(encoder, doc_a, pooling_strategy="sum"),
            _pool_expected(encoder, doc_b, pooling_strategy="sum"),
        ]
    ).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_pooling_matches_dense_contract_flattened_without_prompt() -> None:
    encoder = _make_token_encoder()
    doc_a = torch.randn(4, 4)
    doc_b = torch.randn(3, 4)
    packed = torch.cat([doc_a, doc_b], dim=0).unsqueeze(0)
    features = {
        "token_embeddings": packed,
        "cu_seq_lens_q": torch.tensor([0, 4, 7], dtype=torch.int32),
        "seq_idx": torch.tensor([[0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32),
        "prompt_length": 1,
    }
    dense_pooler = Pooling(4, pooling_mode="mean", include_prompt=False)
    assert dense_pooler(_clone_features(features))["sentence_embedding"].shape[0] == 2

    actual = _run_modules(features, encoder, "sum", include_prompt=False)
    expected = torch.stack(
        [
            _pool_expected(encoder, doc_a, pooling_strategy="sum", prompt_length=1),
            _pool_expected(encoder, doc_b, pooling_strategy="sum", prompt_length=1),
        ]
    ).to(actual.dtype)
    torch.testing.assert_close(actual, expected)


def test_sparse_auto_encoder_token_encoder_respects_max_active_dims() -> None:
    encoder = _make_token_encoder()
    hidden = torch.randn(2, 4, 4)
    features = {
        "token_embeddings": hidden.clone(),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }

    out = encoder(_clone_features(features), max_active_dims=1)
    assert out["token_sparse_values"].shape == (2, 4, 1)
    assert out["token_sparse_indices"].shape == (2, 4, 1)


def test_sparse_auto_encoder_token_encoder_training_outputs_and_gradients() -> None:
    encoder = _make_token_encoder()
    encoder.train()
    for hidden, features in [
        (
            torch.randn(2, 4, 4, requires_grad=True),
            {
                "token_embeddings": torch.randn(2, 4, 4, requires_grad=True),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
            },
        ),
        (
            torch.randn(8, 4, requires_grad=True),
            {"token_embeddings": torch.randn(8, 4, requires_grad=True)},
        ),
    ]:
        features["token_embeddings"] = hidden
        out = encoder(features)
        assert out["token_embedding_backbone"].shape == (8, 4)
        assert out["decoded_token_embeddings"].shape == (8, 4)
        assert "token_reconstruction_loss" not in out

        loss = torch.nn.functional.mse_loss(out["decoded_token_embeddings"], out["token_embedding_backbone"])
        loss = loss + out["token_sparse_values"].sum()
        encoder.zero_grad()
        loss.backward()

        assert hidden.grad is not None
        assert encoder.W_enc.grad is not None
        assert encoder.W_dec is not None
        assert encoder.W_dec.grad is not None
        assert encoder.b_enc.grad is not None
        assert encoder.b_pre.grad is not None


def test_sparse_auto_encoder_token_encoder_from_checkpoint_is_frozen_by_default(tmp_path) -> None:
    module = _make_token_encoder()
    module.save(str(tmp_path))
    checkpoint_path = tmp_path / "sae.pt"
    torch.save(
        {
            "W_enc": module.W_enc.detach(),
            "b_enc": module.b_enc.detach(),
            "b_pre": module.b_pre.detach(),
            "W_dec": module.W_dec.detach(),
            "config": {
                "input_dim": module.input_dim,
                "hidden_dim": module.hidden_dim,
                "k": module.k,
                "has_decoder": True,
            },
        },
        checkpoint_path,
    )

    frozen = SparseAutoEncoderTokenEncoder.from_checkpoint(checkpoint_path)
    assert not frozen.W_enc.requires_grad
    assert not frozen.b_enc.requires_grad
    assert not frozen.b_pre.requires_grad
    assert frozen.W_dec is not None
    assert not frozen.W_dec.requires_grad

    trainable = SparseAutoEncoderTokenEncoder.from_checkpoint(checkpoint_path, frozen=False)
    assert trainable.W_enc.requires_grad
    assert trainable.W_dec is not None
    assert trainable.W_dec.requires_grad


def test_sparse_auto_encoder_token_batching_matches_unchunked_padded_and_flattened() -> None:
    encoder = _make_token_encoder()
    chunked = _copy_encoder(encoder, token_batch_size=2)
    hidden = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    padded = {
        "token_embeddings": hidden.clone(),
        "attention_mask": mask,
    }

    actual = chunked(_clone_features(padded))
    expected = encoder(_clone_features(padded))
    torch.testing.assert_close(actual["token_sparse_values"], expected["token_sparse_values"])
    torch.testing.assert_close(actual["token_sparse_indices"], expected["token_sparse_indices"])
    torch.testing.assert_close(
        SparseTokenPooling(chunked.hidden_dim)(actual)["sentence_embedding"],
        SparseTokenPooling(encoder.hidden_dim)(expected)["sentence_embedding"],
    )

    doc_a = torch.randn(4, 4)
    doc_b = torch.randn(3, 4)
    flattened = {
        "token_embeddings": torch.cat([doc_a, doc_b], dim=0).unsqueeze(0),
        "cu_seq_lens_q": torch.tensor([0, 4, 7], dtype=torch.int32),
        "seq_idx": torch.tensor([[0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32),
    }
    actual = chunked(_clone_features(flattened))
    expected = encoder(_clone_features(flattened))
    torch.testing.assert_close(actual["token_sparse_values"], expected["token_sparse_values"])
    torch.testing.assert_close(actual["token_sparse_indices"], expected["token_sparse_indices"])
    torch.testing.assert_close(
        SparseTokenPooling(chunked.hidden_dim)(actual)["sentence_embedding"],
        SparseTokenPooling(encoder.hidden_dim)(expected)["sentence_embedding"],
    )


def test_sparse_auto_encoder_token_encoder_save_load_roundtrip(tmp_path) -> None:
    module = _make_token_encoder()
    module.save(str(tmp_path))

    loaded = SparseAutoEncoderTokenEncoder.load(str(tmp_path))
    hidden = torch.randn(1, 3, 4)
    features = {
        "token_embeddings": hidden,
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    actual = loaded(features.copy())
    expected = module(features.copy())
    torch.testing.assert_close(actual["token_sparse_values"], expected["token_sparse_values"])
    torch.testing.assert_close(actual["token_sparse_indices"], expected["token_sparse_indices"])
