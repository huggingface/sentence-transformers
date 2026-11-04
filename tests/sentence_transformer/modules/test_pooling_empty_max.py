from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.modules import Pooling


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("flattened", [False, True])
@pytest.mark.parametrize("prompt_length", [0, 2, 5])
def test_max_pooling_empty_sequences_return_zero(padding_side, flattened, prompt_length):
    sequences = [
        torch.tensor([[-9.0, -8.0], [-7.0, -6.0]]) if prompt_length else torch.empty(0, 2),
        torch.tensor([[-9.0, -8.0], [-7.0, -6.0], [-5.0, -4.0], [-3.0, -2.0]]),
        torch.tensor([[-4.0, -5.0], [-2.0, -3.0], [-6.0, -1.0]]),
    ]
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    if flattened:
        embeddings = torch.cat(sequences).unsqueeze(0).requires_grad_()
        features = {"token_embeddings": embeddings, "cu_seq_lens_q": torch.cat([torch.tensor([0]), lengths.cumsum(0)])}
    else:
        embeddings = torch.full((3, 4, 2), 100.0)
        mask = torch.zeros(3, 4, dtype=torch.long)
        for row, sequence in enumerate(sequences):
            if len(sequence):
                start = 4 - len(sequence) if padding_side == "left" else 0
                embeddings[row, start : start + len(sequence)] = sequence
                mask[row, start : start + len(sequence)] = 1
        embeddings.requires_grad_()
        features = {"token_embeddings": embeddings, "attention_mask": mask}
    if prompt_length:
        features["prompt_length"] = prompt_length
    model = Pooling(2, pooling_mode="max", include_prompt=False)

    result = model(features)["sentence_embedding"]

    expected = torch.stack(
        [
            sequence[prompt_length:].max(dim=0).values if len(sequence) > prompt_length else torch.zeros(2)
            for sequence in sequences
        ]
    )
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(torch.nn.functional.normalize(result, dim=1)).all()
    result.sum().backward()
    assert torch.isfinite(embeddings.grad).all()
    if prompt_length >= lengths.max():
        torch.testing.assert_close(embeddings.grad, torch.zeros_like(embeddings))
    if not flattened:
        torch.testing.assert_close(embeddings.grad[mask == 0], torch.zeros_like(embeddings.grad[mask == 0]))
