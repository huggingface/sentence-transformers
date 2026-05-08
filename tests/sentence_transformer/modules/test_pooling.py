from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from packaging.version import parse as parse_version
from transformers import __version__ as transformers_version

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling
from sentence_transformers.sentence_transformer.modules.pooling import _convert_legacy_pooling_kwargs

requires_transformers_v5 = pytest.mark.skipif(
    parse_version(transformers_version) < parse_version("5.0.0"),
    reason="Flattened inputs require transformers >= 5.0",
)


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("prompt", ["", "query: ", "Summarize the following information: "])
def test_pooling_prompt_attention_mask_respects_include_prompt(
    stsb_bert_tiny_model: SentenceTransformer, padding_side: str, prompt: str
) -> None:
    model = stsb_bert_tiny_model
    model.tokenizer.padding_side = padding_side

    sentences = ["Text one", "Text two is a bit longer"]

    # First run with include_prompt=True (default behavior)
    model.set_pooling_include_prompt(True)
    outputs_with_prompt = model.encode(
        sentences,
        prompt=prompt,
        output_value=None,
    )

    # Then run with include_prompt=False (new behavior)
    model.set_pooling_include_prompt(False)
    outputs_without_prompt = model.encode(
        sentences,
        prompt=prompt,
        output_value=None,
    )

    assert len(outputs_with_prompt) == len(outputs_without_prompt) == len(sentences)

    for i, (out_with, out_without) in enumerate(zip(outputs_with_prompt, outputs_without_prompt)):
        attn_with = torch.as_tensor(out_with["attention_mask"])
        attn_without = torch.as_tensor(out_without["attention_mask"])

        # We never want to turn padding tokens back on
        assert torch.all(attn_without <= attn_with)

        if prompt == "":
            assert "prompt_length" not in out_without
            prompt_length = 0
        else:
            prompt_length = out_without["prompt_length"]
        if isinstance(prompt_length, torch.Tensor):
            prompt_length = int(prompt_length.item())
        else:
            prompt_length = int(prompt_length)

        # Positions that changed from 1 -> 0 correspond exactly to the prompt tokens
        removed = (attn_with == 1) & (attn_without == 0)
        assert int(removed.sum().item()) == prompt_length

        # If this is the short text, we should always see some 0's at the start for left padding
        # and at the end for right padding
        if i == 0:
            if padding_side == "left":
                assert attn_without[0] == 0
                assert attn_with[0] == 0
            else:
                assert attn_without[-1] == 0
                assert attn_with[-1] == 0


@pytest.mark.parametrize("pooling_mode", Pooling.POOLING_MODES)
def test_pooling_forward_all_strategies(pooling_mode: str) -> None:
    # Basic sanity check that all pooling strategies run and produce the
    # expected sentence embedding shape.
    embedding_dimension = 8
    pooling = Pooling(embedding_dimension=embedding_dimension, pooling_mode=pooling_mode)

    batch_size, seq_len = 3, 5
    token_embeddings = torch.randn(batch_size, seq_len, embedding_dimension)

    # Mix of left / right padding patterns, but always at least one non-pad token
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.int64,
    )

    features = {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
    }

    outputs = pooling(features)
    sentence_embedding = outputs["sentence_embedding"]

    assert sentence_embedding.shape == (
        batch_size,
        pooling.get_embedding_dimension(),
    )


@pytest.mark.parametrize("pooling_mode", Pooling.POOLING_MODES)
@pytest.mark.parametrize(
    "flattened",
    [False, pytest.param(True, marks=requires_transformers_v5)],
    ids=["padded", "flattened"],
)
def test_pooling_gradient_flow(pooling_mode: str, flattened: bool) -> None:
    """Verify that gradients flow through every pooling mode in both padded and flattened paths."""
    embedding_dimension = 8
    pooling = Pooling(embedding_dimension=embedding_dimension, pooling_mode=pooling_mode)

    token_embeddings = torch.randn(2, 6, embedding_dimension, requires_grad=True)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.int64)

    if flattened:
        features = _build_flattened_features(token_embeddings.detach(), attention_mask)
        # Replace with a fresh leaf tensor so we can check .grad
        flat_embeddings = features["token_embeddings"].detach().requires_grad_(True)
        features["token_embeddings"] = flat_embeddings
    else:
        features = {"token_embeddings": token_embeddings, "attention_mask": attention_mask}

    output = pooling(features)
    loss = output["sentence_embedding"].sum()
    loss.backward()

    grad_tensor = flat_embeddings if flattened else token_embeddings
    assert grad_tensor.grad is not None, f"No gradient computed for {pooling_mode}"
    assert grad_tensor.grad.abs().sum() > 0, f"Gradient is all zeros for {pooling_mode}"


def test_pooling_cls_uses_cls_token_embeddings() -> None:
    dim = 4
    pooling = Pooling(embedding_dimension=dim, pooling_mode="cls")

    batch_size, seq_len = 2, 3
    token_embeddings = torch.randn(batch_size, seq_len, dim)
    cls_token_embeddings = torch.randn(batch_size, dim)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)

    features = {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "cls_token_embeddings": cls_token_embeddings,
    }

    outputs = pooling(features)
    sentence_embedding = outputs["sentence_embedding"]

    assert torch.allclose(sentence_embedding, cls_token_embeddings)


@requires_transformers_v5
def test_pooling_flattened_cls_uses_cls_token_embeddings() -> None:
    dim = 4
    pooling = Pooling(embedding_dimension=dim, pooling_mode="cls")

    token_embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], [100.0, 200.0, 300.0, 400.0]],
            [[5.0, 6.0, 7.0, 8.0], [50.0, 60.0, 70.0, 80.0], [500.0, 600.0, 700.0, 800.0]],
        ]
    )
    attention_mask = torch.ones(2, 3, dtype=torch.int64)
    cls_token_embeddings = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )

    features = _build_flattened_features(token_embeddings, attention_mask)
    features["cls_token_embeddings"] = cls_token_embeddings

    outputs = pooling(features)
    sentence_embedding = outputs["sentence_embedding"]

    assert torch.allclose(sentence_embedding, cls_token_embeddings)


def test_pooling_max_respects_attention_mask() -> None:
    dim = 1
    pooling = Pooling(embedding_dimension=dim, pooling_mode="max")

    # Last position has the largest value but is masked out; max should
    # therefore come from the last unmasked token.
    token_embeddings = torch.tensor(
        [
            [[1.0], [3.0], [5.0], [10.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)

    outputs = pooling({"token_embeddings": token_embeddings, "attention_mask": attention_mask})
    sentence_embedding = outputs["sentence_embedding"]

    assert sentence_embedding.shape == (1, dim)
    assert torch.allclose(sentence_embedding, torch.tensor([[5.0]]))


def test_pooling_mean_and_mean_sqrt_len_tokens() -> None:
    dim = 1
    # Enable both mean and mean_sqrt_len at once to test that the
    # output dimension doubles and that both values are correct.
    pooling = Pooling(
        embedding_dimension=dim,
        pooling_mode=("mean", "mean_sqrt_len_tokens"),
    )

    token_embeddings = torch.tensor(
        [
            [[1.0], [3.0], [5.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)

    outputs = pooling({"token_embeddings": token_embeddings, "attention_mask": attention_mask})
    sentence_embedding = outputs["sentence_embedding"]

    # Implementation uses: sum_embeddings / sqrt(len) for the
    # `mean_sqrt_len_tokens` component, where len is the number of
    # attended tokens.
    expected_mean = (1.0 + 3.0) / 2.0
    expected_mean_sqrt = (1.0 + 3.0) / (2.0**0.5)

    assert sentence_embedding.shape == (1, 2 * dim)
    assert torch.allclose(
        sentence_embedding,
        torch.tensor([[expected_mean, expected_mean_sqrt]]),
        atol=1e-6,
    )


def test_pooling_weightedmean_respects_attention_mask() -> None:
    dim = 1
    pooling = Pooling(embedding_dimension=dim, pooling_mode="weightedmean")

    # With seq_len = 3, the weights are [1, 2, 3]. Only the first two
    # positions are attended to, so the weighted mean is:
    # (1*1 + 3*2) / (1 + 2) = 7/3
    token_embeddings = torch.tensor(
        [
            [[1.0], [3.0], [10.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)

    outputs = pooling({"token_embeddings": token_embeddings, "attention_mask": attention_mask})
    sentence_embedding = outputs["sentence_embedding"]

    expected = (1.0 * 1.0 + 3.0 * 2.0) / (1.0 + 2.0)
    assert sentence_embedding.shape == (1, dim)
    assert torch.allclose(sentence_embedding, torch.tensor([[expected]]), atol=1e-6)


def test_pooling_lasttoken_finds_last_attended_token() -> None:
    dim = 1
    pooling = Pooling(embedding_dimension=dim, pooling_mode="lasttoken")

    # Each row has a different pattern of attended tokens; the last
    # attended position should be selected.
    token_embeddings = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0]],  # last attended: idx 2 -> 2.0
            [[5.0], [6.0], [7.0], [8.0]],  # last attended: idx 1 -> 6.0
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.int64,
    )

    outputs = pooling({"token_embeddings": token_embeddings, "attention_mask": attention_mask})
    sentence_embedding = outputs["sentence_embedding"]

    assert sentence_embedding.shape == (2, dim)
    assert torch.allclose(sentence_embedding, torch.tensor([[2.0], [6.0]]), atol=1e-6)


def test_pooling_lasttoken_all_padding_returns_zero_vector() -> None:
    dim = 2
    pooling = Pooling(embedding_dimension=dim, pooling_mode="lasttoken")

    token_embeddings = torch.ones(1, 4, dim)
    attention_mask = torch.zeros(1, 4, dtype=torch.int64)

    outputs = pooling({"token_embeddings": token_embeddings, "attention_mask": attention_mask})
    sentence_embedding = outputs["sentence_embedding"]

    assert sentence_embedding.shape == (1, dim)
    assert torch.all(sentence_embedding == 0)


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize(
    "prompt_length_value",
    [
        2,
        torch.tensor([2]),
        torch.tensor([2, 2]),
    ],
)
def test_pooling_excludes_prompt_tokens_directly(padding_side: str, prompt_length_value) -> None:
    dim = 1
    pooling = Pooling(embedding_dimension=dim, pooling_mode="mean", include_prompt=False)

    batch_size, seq_len = 2, 5
    token_embeddings = torch.randn(batch_size, seq_len, dim)

    if padding_side == "right":
        # Right padding: [1, 1, 1, 0, 0]
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        )
    else:
        # Left padding: [0, 0, 1, 1, 1]
        attention_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=torch.int64,
        )

    # Clone so we can compare after the forward pass, since Pooling
    # mutates the attention mask in-place when include_prompt=False.
    original_attention_mask = attention_mask.clone()

    features = {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "prompt_length": prompt_length_value,
    }

    pooling(features)

    # For right padding, we expect the first `prompt_length` positions
    # to be set to 0. For left padding, we expect all padding tokens
    # plus the next `prompt_length` tokens to be set to 0.
    if isinstance(prompt_length_value, torch.Tensor):
        prompt_length_scalar = int(prompt_length_value[0].item())
    else:
        prompt_length_scalar = int(prompt_length_value)

    pad_lengths = original_attention_mask.to(torch.int32).argmax(dim=1)
    expected_zero_upto = pad_lengths + prompt_length_scalar

    for i in range(batch_size):
        # All positions strictly before expected_zero_upto[i] must be 0
        assert torch.all(attention_mask[i, : expected_zero_upto[i]] == 0)
        # All original padding positions must still be 0
        assert torch.all(attention_mask[i, original_attention_mask[i] == 0] == 0)


# Shared test fixtures: two sequences with different lengths and a mix of padding.
# seq 0: 3 real tokens + 1 pad, seq 1: 4 real tokens, no pad
_DIM = 2
_TOKEN_EMBEDDINGS = torch.tensor(
    [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [99.0, 99.0]],
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
    ]
)
_ATTENTION_MASK = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.int64)

# Pre-computed expected outputs per mode.
# seq 0 tokens: [1,2], [3,4], [5,6] (3 attended), seq 1 tokens: [10,20], [30,40], [50,60], [70,80] (4 attended)
_EXPECTED_BY_MODE: dict[str, torch.Tensor] = {
    "cls": torch.tensor([[1.0, 2.0], [10.0, 20.0]]),
    "max": torch.tensor([[5.0, 6.0], [70.0, 80.0]]),
    "mean": torch.tensor([[3.0, 4.0], [40.0, 50.0]]),
    "mean_sqrt_len_tokens": torch.tensor([[9.0, 12.0], [160.0, 200.0]])
    / torch.tensor([[3.0, 3.0], [4.0, 4.0]]).sqrt(),
    # weightedmean: sum(w_i * x_i) / sum(w_i) with weights [1, 2, 3, 4]
    # seq 0: (1*[1,2] + 2*[3,4] + 3*[5,6]) / 6 = [22,28]/6
    # seq 1: (1*[10,20] + 2*[30,40] + 3*[50,60] + 4*[70,80]) / 10 = [500,600]/10
    "weightedmean": torch.tensor([[22.0 / 6.0, 28.0 / 6.0], [500.0 / 10.0, 600.0 / 10.0]]),
    "lasttoken": torch.tensor([[5.0, 6.0], [70.0, 80.0]]),
}

_EXPECTED_BY_MODE_WITH_PROMPT_EXCLUDED: dict[str, torch.Tensor] = {
    "max": torch.tensor([[5.0, 6.0], [70.0, 80.0]]),
    "mean": torch.tensor([[4.0, 5.0], [50.0, 60.0]]),
    "mean_sqrt_len_tokens": torch.tensor([[8.0, 10.0], [150.0, 180.0]])
    / torch.tensor([[2.0, 2.0], [3.0, 3.0]]).sqrt(),
    "weightedmean": torch.tensor([[21.0 / 5.0, 26.0 / 5.0], [490.0 / 9.0, 580.0 / 9.0]]),
    "lasttoken": torch.tensor([[5.0, 6.0], [70.0, 80.0]]),
}


def _build_flattened_features(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert padded inputs to flattened format using DataCollatorWithFlattening."""
    from transformers import DataCollatorWithFlattening

    collator = DataCollatorWithFlattening(return_seq_idx=True, return_flash_attn_kwargs=True, return_position_ids=True)
    seq_lengths = attention_mask.sum(dim=1)
    samples = [{"input_ids": torch.zeros(int(length.item()), dtype=torch.long)} for length in seq_lengths]
    collated = collator(samples)

    flat = torch.cat([token_embeddings[i, : int(length.item())] for i, length in enumerate(seq_lengths)], dim=0)
    collated["token_embeddings"] = flat.unsqueeze(0)
    collated.pop("input_ids", None)
    collated.pop("labels", None)
    return collated


@pytest.mark.parametrize("pooling_mode", Pooling.POOLING_MODES)
@pytest.mark.parametrize(
    "flattened",
    [False, pytest.param(True, marks=requires_transformers_v5)],
    ids=["padded", "flattened"],
)
def test_pooling_exact_values(pooling_mode: str, flattened: bool) -> None:
    """Verify each pooling mode produces the expected exact values."""
    pooling = Pooling(embedding_dimension=_DIM, pooling_mode=pooling_mode)
    if flattened:
        features = _build_flattened_features(_TOKEN_EMBEDDINGS, _ATTENTION_MASK)
    else:
        features = {"token_embeddings": _TOKEN_EMBEDDINGS, "attention_mask": _ATTENTION_MASK}
    output = pooling(features)["sentence_embedding"]
    expected = _EXPECTED_BY_MODE[pooling_mode]
    assert output.shape == expected.shape, f"Shape mismatch: {output.shape} vs {expected.shape}"
    assert torch.allclose(output, expected, atol=1e-5), f"Value mismatch for {pooling_mode}:\n{output}\nvs\n{expected}"


@pytest.mark.parametrize(
    "modes",
    [
        ("cls", "mean"),
        ("mean", "mean_sqrt_len_tokens"),
        ("cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"),
    ],
)
@pytest.mark.parametrize(
    "flattened",
    [False, pytest.param(True, marks=requires_transformers_v5)],
    ids=["padded", "flattened"],
)
def test_pooling_multi_mode(modes: tuple[str, ...], flattened: bool) -> None:
    """Verify multiple pooling modes are concatenated in the correct order."""
    pooling = Pooling(embedding_dimension=_DIM, pooling_mode=modes)
    if flattened:
        features = _build_flattened_features(_TOKEN_EMBEDDINGS, _ATTENTION_MASK)
    else:
        features = {"token_embeddings": _TOKEN_EMBEDDINGS, "attention_mask": _ATTENTION_MASK}
    output = pooling(features)["sentence_embedding"]
    expected = torch.cat([_EXPECTED_BY_MODE[m] for m in modes], dim=-1)
    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-5)


@pytest.mark.parametrize(
    "pooling_mode",
    ["max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"],
)
@pytest.mark.parametrize(
    "flattened",
    [False, pytest.param(True, marks=requires_transformers_v5)],
    ids=["padded", "flattened"],
)
def test_pooling_excludes_prompt_tokens_for_padded_and_flattened_inputs(pooling_mode: str, flattened: bool) -> None:
    pooling = Pooling(embedding_dimension=_DIM, pooling_mode=pooling_mode, include_prompt=False)

    features: dict[str, Any]
    if flattened:
        features = cast(dict[str, Any], _build_flattened_features(_TOKEN_EMBEDDINGS, _ATTENTION_MASK))
    else:
        features = {
            "token_embeddings": _TOKEN_EMBEDDINGS,
            "attention_mask": _ATTENTION_MASK.clone(),
        }
    features["prompt_length"] = 1

    output = pooling(features)["sentence_embedding"]
    expected = _EXPECTED_BY_MODE_WITH_PROMPT_EXCLUDED[pooling_mode]

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, atol=1e-5)


def test_pooling_legacy_bool_kwargs_with_deprecation_warning() -> None:
    """Legacy ``pooling_mode_*`` bool kwargs should still work but emit a FutureWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pooling = Pooling(embedding_dimension=8, pooling_mode_cls_token=True)

    assert any(issubclass(warning.category, FutureWarning) for warning in w)
    assert pooling.pooling_mode == "cls"
    assert pooling.pooling_output_dimension == 8


def test_pooling_legacy_multiple_bool_kwargs() -> None:
    """Multiple legacy bool kwargs should produce a tuple of modes in forward order."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        pooling = Pooling(
            embedding_dimension=8,
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=True,
        )
    assert pooling.pooling_mode == ("cls", "mean")
    assert pooling.pooling_output_dimension == 16


def test_pooling_legacy_config_conversion() -> None:
    """Verify that old-style saved configs are silently converted when loading."""
    old_config = {
        "embedding_dimension": 384,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    _convert_legacy_pooling_kwargs(old_config)
    assert old_config == {"embedding_dimension": 384, "pooling_mode": "mean", "include_prompt": True}


def test_pooling_legacy_config_conversion_multi_mode() -> None:
    """Verify legacy config with multiple active modes converts to a tuple."""
    old_config = {
        "embedding_dimension": 384,
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }
    _convert_legacy_pooling_kwargs(old_config)
    assert old_config == {"embedding_dimension": 384, "pooling_mode": ("cls", "mean"), "include_prompt": True}


def test_pooling_config_round_trip(tmp_path: Path) -> None:
    """Verify config survives a save/load round-trip through JSON."""
    for mode in ["cls", ("cls", "mean")]:
        pooling = Pooling(embedding_dimension=8, pooling_mode=mode)
        pooling.save(str(tmp_path))
        loaded = Pooling.load(str(tmp_path))
        assert loaded.pooling_mode == pooling.pooling_mode
        assert loaded.embedding_dimension == pooling.embedding_dimension
        assert loaded.include_prompt == pooling.include_prompt


def test_pooling_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid pooling mode"):
        Pooling(embedding_dimension=8, pooling_mode="nonexistent")


@requires_transformers_v5
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("pooling_mode", Pooling.POOLING_MODES)
def test_pooling_flattened_live_flash_attention(pooling_mode: str) -> None:
    """End-to-end test comparing flattened (flash attention) vs padded pooling on a real model.

    Loads a real SentenceTransformer with flash_attention_2, runs inference, and compares
    the sentence embeddings against the same model without flash attention.
    """
    try:
        import kernels  # noqa: F401
    except ImportError:
        pytest.skip("kernels library not available")

    sentences = ["This is a test sentence.", "Another one that is a bit longer for variety."]

    # Reference model: standard padded inference in float16
    ref_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        model_kwargs={"torch_dtype": torch.float16},
    )
    for module in ref_model:
        if isinstance(module, Pooling):
            module.pooling_mode = pooling_mode
            module.pooling_output_dimension = module.embedding_dimension
            break
    ref_embeddings = ref_model.encode(sentences, convert_to_tensor=True)

    # Flash attention model: flattened inference in float16
    flash_model = SentenceTransformer(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16},
    )
    for module in flash_model:
        if isinstance(module, Pooling):
            module.pooling_mode = pooling_mode
            module.pooling_output_dimension = module.embedding_dimension
            break
    flash_embeddings = flash_model.encode(sentences, convert_to_tensor=True)

    # float16 has limited precision, so use a generous tolerance
    assert ref_embeddings.shape == flash_embeddings.shape
    assert torch.allclose(ref_embeddings.float(), flash_embeddings.float(), atol=1e-2), (
        f"Embeddings differ for pooling_mode={pooling_mode!r}, max absolute difference: {(ref_embeddings - flash_embeddings).abs().max().item()}"
    )
