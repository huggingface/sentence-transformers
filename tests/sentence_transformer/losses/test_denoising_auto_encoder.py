from __future__ import annotations

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss

TINY_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


def _base_params(loss: DenoisingAutoEncoderLoss) -> tuple[dict, dict]:
    """Return ``(encoder_params, decoder_base_params)`` keyed by name, for the shared base models."""
    encoder = loss.encoder.transformers_model
    decoder_base = loss.decoder._modules[loss.decoder.base_model_prefix]
    return dict(encoder.named_parameters()), dict(decoder_base.named_parameters())


def test_tie_encoder_decoder_does_not_raise(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Regression for #3737: tying raised a RuntimeError on transformers>=5.0.0, which removed
    the ``PreTrainedModel._tie_encoder_decoder_weights`` helper this loss relied on."""
    DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)


def test_tied_weights_share_storage(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """With tying, every parameter shared by name between encoder and decoder is the *same object*."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)
    encoder_params, decoder_params = _base_params(loss)

    # The word embeddings are the canonical tied weight: assert true aliasing, not just equal values.
    name = "embeddings.word_embeddings.weight"
    assert encoder_params[name] is decoder_params[name]

    # Every parameter the two share by name is tied to the same storage. The encoder's pooler is
    # encoder-only (the decoder's base model is built with add_pooling_layer=False), so it is
    # correctly excluded here rather than asserted as tied.
    common = encoder_params.keys() & decoder_params.keys()
    assert common, "expected the encoder and decoder to share parameters by name"
    untied = [n for n in common if encoder_params[n].data_ptr() != decoder_params[n].data_ptr()]
    assert not untied, f"these shared-name weights were not tied: {untied}"


def test_decoder_only_weights_are_not_tied(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Cross-attention is decoder-only, so it has no encoder counterpart and must be skipped."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)
    encoder_params, decoder_params = _base_params(loss)

    cross_attention = [n for n in decoder_params if "crossattention" in n]
    assert cross_attention, "expected the decoder to carry cross-attention parameters"
    assert all(n not in encoder_params for n in cross_attention)


def test_tying_preserves_decoder_lm_head(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The decoder's weight-tied LM head must stay bound to the shared input embeddings. Tying the
    encoder *to* the decoder keeps all three as one tensor; reversing the direction would rebind the
    decoder's input embeddings to the encoder's and orphan the LM head (silently untying it)."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)

    encoder_in = loss.encoder.transformers_model.embeddings.word_embeddings.weight
    decoder_base = loss.decoder._modules[loss.decoder.base_model_prefix]
    decoder_in = decoder_base.embeddings.word_embeddings.weight
    lm_head = loss.decoder.get_output_embeddings().weight

    # The test model has tie_word_embeddings=True, so input embeddings and LM head share one tensor.
    assert encoder_in is decoder_in is lm_head


def test_untied_weights_are_independent(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Without tying, the encoder and decoder keep independent storage."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, decoder_name_or_path=TINY_MODEL, tie_encoder_decoder=False)
    encoder_params, decoder_params = _base_params(loss)

    name = "embeddings.word_embeddings.weight"
    assert encoder_params[name].data_ptr() != decoder_params[name].data_ptr()


def test_forward_backward_flows_grad_to_tied_weight(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A train step produces a finite scalar and flows gradient into the tied embedding."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)
    loss = loss.to(stsb_bert_tiny_model.device)  # the trainer normally handles device placement

    features = stsb_bert_tiny_model.preprocess(["A noisy input sentence for tsdae"])
    features = {
        k: v.to(stsb_bert_tiny_model.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()
    }

    value = loss([features, features], labels=None)
    assert value.dim() == 0
    assert torch.isfinite(value)

    value.backward()
    shared = loss.encoder.transformers_model.embeddings.word_embeddings.weight
    assert shared.grad is not None


def test_tie_without_decoder_name_does_not_require_it(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """When tying, ``decoder_name_or_path`` may be omitted: it defaults to the encoder's checkpoint."""
    loss = DenoisingAutoEncoderLoss(stsb_bert_tiny_model, tie_encoder_decoder=True)
    assert loss.decoder is not None
