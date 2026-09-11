from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import CNN, LSTM, Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


def test_cnn_save_load_preserves_strides(tmp_path: Path) -> None:
    model = CNN(2, out_channels=3, kernel_sizes=[1, 3], stride_sizes=[2, 2])
    inputs = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)

    with torch.no_grad():
        expected = model({"token_embeddings": inputs.clone()})["token_embeddings"]
        model.save(str(tmp_path))
        restored = CNN.load(str(tmp_path), local_files_only=True)
        actual = restored({"token_embeddings": inputs.clone()})["token_embeddings"]

    torch.testing.assert_close(actual, expected)


def _anchor_cnn(kernel_sizes: list[int], stride_sizes: list[int]) -> CNN:
    cnn = CNN(1, out_channels=1, kernel_sizes=kernel_sizes, stride_sizes=stride_sizes)
    with torch.no_grad():
        for conv in cnn.convs:
            conv.weight.zero_()
            conv.weight[:, :, (conv.kernel_size[0] - 1) // 2] = 1
            conv.bias.fill_(0.25)
    return cnn


@pytest.mark.parametrize(
    ("kernel_sizes", "stride_sizes", "short_text", "expected_tokens"),
    [
        ([1, 3], [2, 2], "a b", [1.25]),
        ([2, 4], [1, 1], "a b c", [1.25, 2.25]),
        ([2, 4], [2, 2], "a b c", [1.25]),
        ([1, 2], [2, 2], "a b", [1.25]),
    ],
    ids=["strided", "even-kernel", "strided-even-kernel", "concatenating-mixed-parity"],
)
def test_cnn_encode_ignores_batch_padding(
    kernel_sizes: list[int], stride_sizes: list[int], short_text: str, expected_tokens: list[float]
) -> None:
    vocab = ["<pad>", "a", "b", "c", "d", "e", "f"]
    word_embeddings = WordEmbeddings(
        WhitespaceTokenizer(vocab=vocab, stop_words=[]),
        torch.arange(len(vocab), dtype=torch.float32).unsqueeze(1),
    )
    cnn = _anchor_cnn(kernel_sizes, stride_sizes)
    model = SentenceTransformer(modules=[word_embeddings, cnn, Pooling(2, "mean")], device="cpu")
    texts = [short_text, "a b c d e"]

    alone = model.encode(texts[:1], convert_to_tensor=True, show_progress_bar=False)
    batched = model.encode(texts, batch_size=2, convert_to_tensor=True, show_progress_bar=False)
    torch.testing.assert_close(alone[0], batched[0])
    expected = torch.tensor(expected_tokens).unsqueeze(1).expand(-1, 2)
    torch.testing.assert_close(alone[0], expected.mean(dim=0))

    alone_tokens = model.encode(texts[:1], output_value="token_embeddings", show_progress_bar=False)
    batched_tokens = model.encode(texts, batch_size=2, output_value="token_embeddings", show_progress_bar=False)
    torch.testing.assert_close(alone_tokens[0], expected)
    torch.testing.assert_close(batched_tokens[0], expected)


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_cnn_excludes_downsampled_prompt_without_changing_replayed_features(padding_side: str) -> None:
    cnn = _anchor_cnn([1, 3], [2, 2])
    pooling = Pooling(2, "mean", include_prompt=False)
    if padding_side == "right":
        embeddings = [[100, 101, 102, 3, 4, 99], [100, 101, 102, 3, 4, 5]]
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    else:
        embeddings = [[99, 100, 101, 102, 3, 4], [100, 101, 102, 3, 4, 5]]
        attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    features = {
        "token_embeddings": torch.tensor(embeddings, dtype=torch.float32).unsqueeze(-1),
        "attention_mask": attention_mask,
        "prompt_length": 3,
    }
    single = {
        "token_embeddings": torch.tensor([[[100.0], [101.0], [102.0], [3.0], [4.0]]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "prompt_length": 3,
    }

    with torch.no_grad():
        expected = pooling(cnn(single))["sentence_embedding"].expand(2, -1)
        first = pooling(cnn(features))["sentence_embedding"]
        replayed = pooling(cnn(features))["sentence_embedding"]

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(replayed, expected)


def test_cnn_concatenating_different_strides_require_valid_tokens_in_every_branch() -> None:
    cnn = _anchor_cnn([1, 2], [2, 1])
    features = {
        "token_embeddings": torch.tensor([[[1.0], [99.0]], [[2.0], [3.0]]]),
        "attention_mask": torch.tensor([[1, 0], [1, 1]]),
    }
    output = Pooling(2, "mean")(cnn(features))["sentence_embedding"]
    # The even branch has no valid output for the single-token row.
    torch.testing.assert_close(output, torch.tensor([[0.0, 0.0], [2.25, 2.25]]))


def test_cnn_preserves_prompt_exclusion_through_stacked_layers() -> None:
    first_cnn = _anchor_cnn([1], [2])
    second_cnn = _anchor_cnn([1], [2])
    pooling = Pooling(1, "mean", include_prompt=False)
    features = {
        "token_embeddings": torch.tensor(
            [
                [99, 100, 101, 102, 3, 4, 5, 6, 7, 8],
                [100, 101, 102, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.float32,
        ).unsqueeze(-1),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        "prompt_length": torch.tensor([3]),
    }
    output = pooling(second_cnn(first_cnn(features)))["sentence_embedding"]
    for row in range(2):
        single = {
            "token_embeddings": features["token_embeddings"][row, features["attention_mask"][row].bool()].unsqueeze(0),
            "attention_mask": torch.ones(1, int(features["attention_mask"][row].sum()), dtype=torch.long),
            "prompt_length": torch.tensor([3]),
        }
        expected = pooling(second_cnn(first_cnn(single)))["sentence_embedding"][0]
        torch.testing.assert_close(output[row], expected)


@pytest.mark.parametrize(("with_attention_mask", "padding_side"), [(False, "right"), (True, "right"), (True, "left")])
def test_cnn_preserves_packed_lstm_outputs(with_attention_mask: bool, padding_side: str) -> None:
    cnn = _anchor_cnn([1, 3], [2, 2])
    lstm = LSTM(2, hidden_dim=2)
    features = {
        "token_embeddings": torch.arange(12, dtype=torch.float32).reshape(2, 6, 1),
        "sentence_lengths": torch.tensor([2, 6]),
    }
    if with_attention_mask:
        features["attention_mask"] = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    if padding_side == "left":
        features["token_embeddings"][0, :, 0] = torch.tensor([99, 99, 99, 99, 0, 1])
        features["attention_mask"][0] = torch.tensor([0, 0, 0, 0, 1, 1])

    with torch.no_grad():
        batched = lstm(cnn(features))["token_embeddings"]
        for row, length in enumerate([2, 6]):
            tokens = features["token_embeddings"][row]
            tokens = tokens[features["attention_mask"][row].bool()] if with_attention_mask else tokens[:length]
            single = {
                "token_embeddings": tokens.unsqueeze(0),
                "sentence_lengths": torch.tensor([length]),
            }
            if with_attention_mask:
                single["attention_mask"] = torch.ones(1, length, dtype=torch.long)
            expected = lstm(cnn(single))["token_embeddings"][0]
            torch.testing.assert_close(batched[row, : len(expected)], expected)


@pytest.mark.parametrize("with_attention_mask", [False, True])
def test_cnn_ignores_nonzero_embeddings_at_padded_positions(with_attention_mask: bool) -> None:
    cnn = CNN(1, out_channels=1, kernel_sizes=[3], stride_sizes=[2])
    with torch.no_grad():
        cnn.convs[0].weight.fill_(1)
        cnn.convs[0].bias.zero_()
    single = {
        "token_embeddings": torch.tensor([[[1.0], [2.0], [3.0]]]),
        "sentence_lengths": torch.tensor([3]),
    }
    batch = {
        "token_embeddings": torch.tensor([[[1.0], [2.0], [3.0], [99.0], [99.0]], [[1.0], [2.0], [3.0], [4.0], [5.0]]]),
        "sentence_lengths": torch.tensor([3, 5]),
    }
    if with_attention_mask:
        single["attention_mask"] = torch.ones(1, 3, dtype=torch.long)
        batch["attention_mask"] = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        expected = cnn(single)
        actual = cnn(batch)
        torch.testing.assert_close(
            actual["token_embeddings"][0, : expected["token_embeddings"].size(1)],
            expected["token_embeddings"][0],
        )
        pooling = Pooling(1, "mean")
        torch.testing.assert_close(
            pooling(actual)["sentence_embedding"][0],
            pooling(expected)["sentence_embedding"][0],
        )
