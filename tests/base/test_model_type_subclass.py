from __future__ import annotations

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder


def test_model_type_inherited_for_sentence_transformer_subclass(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """A subclass of SentenceTransformer should still report `model_type == "SentenceTransformer"`,
    so `_load_modules` takes the config path (not the conversion path) for pre-saved ST models.
    Reproduces #3536."""
    assert stsb_bert_tiny_model.model_type == "SentenceTransformer"

    converted_called = False

    class MySentenceTransformer(SentenceTransformer):
        def _load_converted_modules(self, *args, **kwargs):
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MySentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert subclassed.model_type == "SentenceTransformer"
    assert not converted_called, "Plain ST subclass should not hit the conversion loader path"


def test_model_type_inherited_for_sparse_encoder_subclass(splade_bert_tiny_model: SparseEncoder) -> None:
    """Same inheritance and routing for SparseEncoder subclasses."""
    assert splade_bert_tiny_model.model_type == "SparseEncoder"

    converted_called = False

    class MySparseEncoder(SparseEncoder):
        def _load_converted_modules(self, *args, **kwargs):
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MySparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")
    assert subclassed.model_type == "SparseEncoder"
    assert not converted_called, "Plain SparseEncoder subclass should not hit the conversion loader path"


def test_model_type_inherited_for_cross_encoder_subclass(reranker_bert_tiny_model: CrossEncoder) -> None:
    """Same inheritance and routing for CrossEncoder subclasses."""
    assert reranker_bert_tiny_model.model_type == "CrossEncoder"

    converted_called = False

    class MyCrossEncoder(CrossEncoder):
        def _load_converted_modules(self, *args, **kwargs):
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    subclassed = MyCrossEncoder("cross-encoder-testing/reranker-bert-tiny-gooaq-bce")
    assert subclassed.model_type == "CrossEncoder"
    assert not converted_called, "Plain CrossEncoder subclass should not hit the conversion loader path"


def test_model_type_override_takes_precedence() -> None:
    """A subclass that explicitly declares its own `model_type` opts out of the archetype
    identity, restoring the pre-#3536 conversion-path behavior on archetype-typed models."""
    converted_called = False

    class OverrideSentenceTransformer(SentenceTransformer):
        model_type = "OverrideSentenceTransformer"

        def _load_converted_modules(self, *args, **kwargs):
            nonlocal converted_called
            converted_called = True
            return super()._load_converted_modules(*args, **kwargs)

    overridden = OverrideSentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    assert overridden.model_type == "OverrideSentenceTransformer"
    assert converted_called, "Overriding `model_type` should route through the conversion loader path"
