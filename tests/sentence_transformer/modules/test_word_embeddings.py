from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import sentence_transformers.sentence_transformer.modules.word_embeddings as word_embeddings_module
from sentence_transformers.sentence_transformer.modules import WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_word_embeddings_load_routes_tokenizer_class_through_trust_gate(monkeypatch, trust_remote_code):
    """`WordEmbeddings.load` must resolve the config-controlled `tokenizer_class` through the
    `import_module_class` trust gate, forwarding `model_name_or_path` and the caller's `trust_remote_code`,
    rather than a raw `import_from_string`. Otherwise an untrusted Hub model could point `tokenizer_class` at
    an arbitrary dotted path and have it imported with no trust check (the same ungated-import class as the
    module `type` field, see #3801). Passing both trust values pins that the caller's value is forwarded, not
    a hardcoded default.
    """
    captured = {}

    class FakeTokenizerClass:
        @classmethod
        def load(cls, path):
            return WhitespaceTokenizer()

    def fake_import_module_class(class_ref, **kwargs):
        captured["class_ref"] = class_ref
        captured["kwargs"] = kwargs
        return FakeTokenizerClass

    monkeypatch.setattr(word_embeddings_module, "import_module_class", fake_import_module_class)
    monkeypatch.setattr(
        WordEmbeddings,
        "load_config",
        classmethod(
            lambda cls, **kwargs: {
                "tokenizer_class": "attacker.evil.Tokenizer",
                "update_embeddings": False,
                "max_seq_length": 100,
            }
        ),
    )
    monkeypatch.setattr(WordEmbeddings, "load_dir_path", classmethod(lambda cls, **kwargs: "/fake/local/path"))
    monkeypatch.setattr(
        WordEmbeddings,
        "load_torch_weights",
        classmethod(lambda cls, **kwargs: {"emb_layer.weight": torch.zeros(3, 4)}),
    )

    model = WordEmbeddings.load("attacker/evil-word-embeddings", trust_remote_code=trust_remote_code)

    assert isinstance(model, WordEmbeddings)
    assert captured["class_ref"] == "attacker.evil.Tokenizer"
    assert captured["kwargs"]["model_name_or_path"] == "attacker/evil-word-embeddings"
    assert captured["kwargs"]["trust_remote_code"] is trust_remote_code


def test_word_embeddings_save_load_round_trip_does_not_warn(tmp_path):
    """A genuine WordEmbeddings model stores a `sentence_transformers.*` tokenizer class, so a real
    save/load round-trip resolves through the gate's namespace allowlist (Path A) with no FutureWarning. This
    guards against the trust gate regressing normal loading.
    """
    vocab = ["hello", "world", "foo"]
    model = WordEmbeddings(
        tokenizer=WhitespaceTokenizer(vocab=vocab),
        embedding_weights=np.random.rand(len(vocab), 4).astype(np.float32),
    )
    model.save(str(tmp_path))

    # A genuine model stores its tokenizer as a `sentence_transformers.*` ref, which is what routes loading
    # through the gate's namespace allowlist (Path A) rather than the trust-gated fallback.
    assert model.get_config_dict()["tokenizer_class"].startswith("sentence_transformers.")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        loaded = WordEmbeddings.load(str(tmp_path))

    assert isinstance(loaded, WordEmbeddings)
    assert isinstance(loaded.tokenizer, WhitespaceTokenizer)
    assert list(loaded.tokenizer.get_vocab()) == vocab
