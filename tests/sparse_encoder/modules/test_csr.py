from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from sentence_transformers import SparseEncoder


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
