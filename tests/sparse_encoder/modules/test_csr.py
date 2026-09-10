from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.modules import SparseAutoEncoder


# Create a wrapper to measure outputs of the forward method
class ForwardMethodWrapper:
    def __init__(self, model, is_inference: bool = True):
        self.model = model
        self.original_forward = model.forward
        self.is_inference = is_inference
        self.outputs = []

    def __call__(self, *args, **kwargs):
        # Set the model to training mode if is_train is True
        with torch.inference_mode() if self.is_inference else nullcontext():
            output = self.original_forward(*args, **kwargs)
        self.outputs.append(output)
        return output

    def reset(self):
        self.outputs = []


@pytest.mark.parametrize(
    ["is_inference", "expected_keys"],
    [
        (
            False,
            {
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "token_embeddings",
                "sentence_embedding",
                "sentence_embedding_backbone",
                "sentence_embedding_encoded",
                "sentence_embedding_encoded_4k",
                "auxiliary_embedding",
                "decoded_embedding_k",
                "decoded_embedding_4k",
                "decoded_embedding_aux",
                "decoded_embedding_k_pre_bias",
                "modality",
            },
        ),
        (
            True,
            {"input_ids", "attention_mask", "token_type_ids", "token_embeddings", "sentence_embedding", "modality"},
        ),
    ],
)
def test_csr_outputs(csr_bert_tiny_model: SparseEncoder, is_inference: bool, expected_keys: set) -> None:
    model = csr_bert_tiny_model

    # Create the wrapper and replace the forward method
    wrapper = ForwardMethodWrapper(model, is_inference=is_inference)
    model.forward = wrapper

    # Run the encode method which should call forward internally
    inputs = model.preprocess(["This is a test sentence."])
    inputs = {
        key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
    }
    model(inputs)

    # Check that the model was called in the correct mode, and that the outputs contain the expected keys
    assert set(wrapper.outputs[0].keys()) == expected_keys
    # We don't have to restore the original forward method, as the model will not be reused


@pytest.mark.parametrize("mode", ["inference_mode", "eval", "no_grad"])
def test_csr_inference_does_not_update_dead_feature_stats(mode: str) -> None:
    module = SparseAutoEncoder(input_dim=4, hidden_dim=8, k=2, k_aux=2)
    features = {"sentence_embedding": torch.randn(2, 4)}

    if mode == "inference_mode":
        # forward() writes an inference tensor back into the dict, so the input must not be reused
        with torch.inference_mode():
            module({key: value.clone() for key, value in features.items()})
    elif mode == "eval":
        module.eval()
        module(dict(features))
    else:
        with torch.no_grad():
            module(dict(features))

    assert torch.equal(module.stats_last_nonzero, torch.zeros_like(module.stats_last_nonzero))


def test_csr_training_updates_dead_feature_stats() -> None:
    module = SparseAutoEncoder(input_dim=4, hidden_dim=8, k=2, k_aux=2)
    module.train()

    module({"sentence_embedding": torch.randn(2, 4)})

    assert torch.all(module.stats_last_nonzero > 0)


def test_csr_encode_does_not_update_dead_feature_stats(csr_bert_tiny_model: SparseEncoder) -> None:
    sparse_auto_encoder = csr_bert_tiny_model[-1]
    sparse_auto_encoder.stats_last_nonzero.zero_()

    csr_bert_tiny_model.encode(["This is a test sentence."] * 4, batch_size=1)

    assert torch.equal(
        sparse_auto_encoder.stats_last_nonzero, torch.zeros_like(sparse_auto_encoder.stats_last_nonzero)
    )
