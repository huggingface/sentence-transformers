from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask


class PackHiddenStates(torch.nn.Module):
    """Reproduce the packed hidden-state layout without requiring a flash-attention GPU kernel."""

    def forward(self, features):
        mask = features["attention_mask"].bool()
        lengths = mask.sum(dim=-1)
        features["cu_seq_lens_q"] = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])
        features["all_layer_embeddings"] = tuple(
            layer[mask].unsqueeze(0) for layer in features["all_layer_embeddings"]
        )
        features["token_embeddings"] = features["token_embeddings"][mask].unsqueeze(0)
        features["input_ids"] = features["input_ids"][mask].unsqueeze(0)
        features.pop("attention_mask")
        return features


@pytest.mark.parametrize("model_class", [SentenceTransformer, MultiVectorEncoder])
@pytest.mark.parametrize("method", ["encode", "encode_query", "encode_document"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("layout", ["padded", "packed", "repadded"])
def test_encode_returns_per_input_hidden_states(stsb_bert_tiny_model, model_class, method, batch_size, layout):
    model = stsb_bert_tiny_model.cpu()
    if model_class is MultiVectorEncoder:
        model = MultiVectorEncoder(modules=[model[0]], device="cpu")
    if layout != "padded":
        model.append(PackHiddenStates())
    if layout == "repadded":
        model.append(MultiVectorMask(skiplist_words=[]))
    model[0].model.config.output_hidden_states = True
    model.eval()
    texts = ["Short.", "This is the longest sentence in this batch.", "A middle length sentence."]

    outputs = getattr(model, method)(texts, output_value=None, batch_size=batch_size)
    assert len(outputs) == len(texts)
    for text, output in zip(texts, outputs):
        with torch.inference_mode():
            expected = model(model.preprocess([text]))
        layers = output["all_layer_embeddings"]
        assert isinstance(layers, tuple)
        assert len(layers) == model[0].model.config.num_hidden_layers + 1
        for layer, expected_layer in zip(layers, expected["all_layer_embeddings"]):
            assert layer.ndim == 2
            if layout == "packed":
                assert layer.shape == expected_layer.shape[1:]
                actual = layer
            else:
                assert layer.shape == output["token_embeddings"].shape
                actual = layer[output["attention_mask"].bool()]
            torch.testing.assert_close(
                actual,
                expected_layer[0],
                atol=1e-5,
                rtol=1e-4,
            )
        assert output["modality"] == "text"

    single = getattr(model, method)(texts[0], output_value=None)
    assert single["all_layer_embeddings"][0].ndim == 2
