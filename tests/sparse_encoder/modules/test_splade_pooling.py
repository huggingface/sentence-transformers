from __future__ import annotations

import pytest
import torch

from sentence_transformers.sparse_encoder.modules import SpladePooling


@pytest.mark.parametrize("pooling_strategy", ["max", "sum"])
@pytest.mark.parametrize("activation_function", ["relu", "log1p_relu"])
@pytest.mark.parametrize("training", [True, False])
def test_flattened_forward_matches_padded(pooling_strategy: str, activation_function: str, training: bool) -> None:
    # FA2 input unpadding feeds SpladePooling flat logits with cu_seq_lens_q instead of a padded
    # batch with an attention mask. Both paths must produce identical sparse embeddings. Both branch
    # on training to pick in-place transforms, so inference runs the other half of the code.
    torch.manual_seed(42)
    vocab_size = 97
    seq_lengths = [5, 12, 1, 8]
    segments = [torch.randn(length, vocab_size) for length in seq_lengths]

    padded_logits = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0.0)
    attention_mask = torch.zeros(len(seq_lengths), max(seq_lengths), dtype=torch.long)
    for i, length in enumerate(seq_lengths):
        attention_mask[i, :length] = 1

    flat_logits = torch.cat(segments, dim=0).unsqueeze(0)
    cu_seq_lens = torch.tensor([0, *torch.tensor(seq_lengths).cumsum(0).tolist()])

    pooling = SpladePooling(pooling_strategy=pooling_strategy, activation_function=activation_function)
    pooling.train(training)
    flat_logits_before = flat_logits.clone()
    padded_out = pooling({"token_embeddings": padded_logits, "attention_mask": attention_mask})
    flat_out = pooling({"token_embeddings": flat_logits, "cu_seq_lens_q": cu_seq_lens})

    assert flat_out["sentence_embedding"].shape == padded_out["sentence_embedding"].shape
    assert torch.allclose(flat_out["sentence_embedding"], padded_out["sentence_embedding"], atol=1e-6)
    # The flat path slices views of the caller's tensor, so its transforms must not be in-place.
    assert torch.equal(flat_logits, flat_logits_before)


def test_flattened_forward_backward() -> None:
    # The flat path must stay autograd-safe in training mode (no in-place ops on graph tensors).
    vocab_size = 13
    flat_logits = torch.randn(1, 9, vocab_size, requires_grad=True)
    cu_seq_lens = torch.tensor([0, 4, 9])

    pooling = SpladePooling(pooling_strategy="max")
    pooling.train()
    out = pooling({"token_embeddings": flat_logits, "cu_seq_lens_q": cu_seq_lens})
    out["sentence_embedding"].sum().backward()
    assert flat_logits.grad is not None
    assert flat_logits.grad.shape == flat_logits.shape
