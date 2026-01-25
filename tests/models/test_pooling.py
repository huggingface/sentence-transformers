from __future__ import annotations

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("prompt", ["", "query: ", "Summarize the following information: "])
def test_pooling_prompt_attention_mask_respects_include_prompt(
    stsb_bert_tiny_model: SentenceTransformer,
    padding_side: str,
    prompt: str,
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
    word_embedding_dimension = 8
    pooling = Pooling(word_embedding_dimension=word_embedding_dimension, pooling_mode=pooling_mode)

    batch_size, seq_len = 3, 5
    token_embeddings = torch.randn(batch_size, seq_len, word_embedding_dimension)

    # Mix of left / right padding patterns, but always at least one non-pad token
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
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
        pooling.get_sentence_embedding_dimension(),
    )


def test_pooling_cls_uses_cls_token_embeddings() -> None:
    dim = 4
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="cls")

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


def test_pooling_max_respects_attention_mask() -> None:
    dim = 1
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="max")

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
        word_embedding_dimension=dim,
        pooling_mode_mean_tokens=True,
        pooling_mode_mean_sqrt_len_tokens=True,
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
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="weightedmean")

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
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="lasttoken")

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
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="lasttoken")

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
    pooling = Pooling(word_embedding_dimension=dim, pooling_mode="mean", include_prompt=False)

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
