"""
Computes embeddings
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import DataCollatorWithFlattening

from sentence_transformers import SentenceTransformer


def test_encode_token_embeddings(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """
    Test that encode(output_value='token_embeddings') works
    """
    model = paraphrase_distilroberta_base_v1_model
    sent = [
        "Hello Word, a test sentence",
        "Here comes another sentence",
        "My final sentence",
        "Sentences",
        "Sentence five five five five five five five",
    ]
    emb = model.encode(sent, output_value="token_embeddings", batch_size=2)
    assert len(emb) == len(sent)

    for s, e in zip(sent, emb):
        assert len(model.preprocess([s])["input_ids"][0]) == e.shape[0]


def test_encode_single_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Single sentence
    emb = model.encode("Hello Word, a test sentence")
    assert emb.shape == (768,)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Single sentence as list
    emb = model.encode(["Hello Word, a test sentence"])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 7.9811716) < 0.002

    # Sentence list
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 22.968266) < 0.007


def test_encode_normalize(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    emb = model.encode(
        [
            "Hello Word, a test sentence",
            "Here comes another sentence",
            "My final sentence",
        ],
        normalize_embeddings=True,
    )
    assert emb.shape == (3, 768)
    for norm in np.linalg.norm(emb, axis=1):
        assert abs(norm - 1) < 0.001


def test_encode_tuple_sentences(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    model = paraphrase_distilroberta_base_v1_model
    # Input a sentence tuple
    emb = model.encode([("Hello Word, a test sentence", "Second input for model")])
    assert emb.shape == (1, 768)
    assert abs(np.sum(emb) - 9.503508) < 0.002

    # List of sentence tuples
    emb = model.encode(
        [
            ("Hello Word, a test sentence", "Second input for model"),
            ("My second tuple", "With two inputs"),
            ("Final tuple", "final test"),
        ]
    )
    assert emb.shape == (3, 768)
    assert abs(np.sum(emb) - 32.14627) < 0.002


def test_encode_routes_through_module_call(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """encode() must run the forward pass via __call__ so that model.compile() applies to inference."""
    model = stsb_bert_tiny_model
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.encode("Hello world")
    finally:
        handle.remove()
    assert calls, "encode() should invoke the model via __call__, not call forward() directly"


def _flatten_text_inputs(model: SentenceTransformer) -> None:
    """Flatten text inputs as Transformer does with Flash Attention 2, which is not available on CPU."""
    transformer = model[0]
    transformer.can_flatten_inputs = True
    transformer.data_collator = DataCollatorWithFlattening(
        return_seq_idx=True, return_flash_attn_kwargs=True, return_position_ids=True
    )
    transformer._flatten_position_offset = transformer._infer_flatten_position_offset()


FLATTENING_SENTENCES = [
    "Hello Word, a test sentence",
    "Here comes another sentence",
    "My final sentence",
    "Sentences",
    "Sentence five five five five five five five",
]


def test_encode_token_embeddings_with_flattened_inputs(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    padded = model.encode(FLATTENING_SENTENCES, output_value="token_embeddings", batch_size=1)
    _flatten_text_inputs(model)

    # A flattened batch of one input holds exactly that input, so it must give the same token embeddings.
    flattened = model.encode(FLATTENING_SENTENCES, output_value="token_embeddings", batch_size=1)
    assert len(flattened) == len(FLATTENING_SENTENCES)
    for padded_embeddings, flattened_embeddings in zip(padded, flattened):
        torch.testing.assert_close(flattened_embeddings, padded_embeddings, rtol=1e-4, atol=1e-5)

    # Larger flattened batches pack several inputs into one row: each input must still get its own tokens.
    # The values can differ here, as the inputs attend to each other without a variable-length attention kernel.
    flattened = model.encode(FLATTENING_SENTENCES, output_value="token_embeddings", batch_size=2)
    assert [embeddings.shape for embeddings in flattened] == [embeddings.shape for embeddings in padded]


def test_encode_all_outputs_with_flattened_inputs(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    padded = model.encode(FLATTENING_SENTENCES, output_value=None, batch_size=2)
    _flatten_text_inputs(model)

    flattened = model.encode(FLATTENING_SENTENCES, output_value=None, batch_size=2)
    assert len(flattened) == len(FLATTENING_SENTENCES)
    for padded_output, flattened_output in zip(padded, flattened):
        padded_mask = padded_output["attention_mask"].bool()
        flattened_mask = flattened_output["attention_mask"].bool()
        assert torch.equal(flattened_output["input_ids"][flattened_mask], padded_output["input_ids"][padded_mask])
        assert flattened_output["token_embeddings"].shape[0] == flattened_mask.shape[0]
        assert flattened_output["sentence_embedding"].shape == padded_output["sentence_embedding"].shape
