"""
Test for the AutoProcessor -> AutoTokenizer fallback in Transformer.

Validates that a text-only model whose tokenizer is registered via
``auto_map -> AutoTokenizer`` (with ``trust_remote_code=True``) but that
ships no ``processor_config.json`` / ``preprocessor_config.json`` loads
correctly via ``SentenceTransformer(...)``.
"""

from __future__ import annotations

import pytest
from transformers import PreTrainedTokenizerBase

from sentence_transformers import SentenceTransformer


# Public NeoAraBERT-based sentence-embedding model whose custom Arabic
# morphological tokenizer is exposed only via ``auto_map -> AutoTokenizer``.
# Before the fix, this raises:
#   ValueError: Unrecognized processing class in <repo>. Can't instantiate ...
TEST_MODEL = "Omartificial-Intelligence-Space/NeoAraBERT-MSA-Synonym-Matryoshka-V1"


@pytest.mark.slow
def test_autotokenizer_fallback_loads_text_only_custom_tokenizer() -> None:
    model = SentenceTransformer(TEST_MODEL, trust_remote_code=True)

    # The tokenizer must come back as a real tokenizer, not None.
    transformer_module = model[0]
    assert transformer_module.tokenizer is not None
    assert isinstance(transformer_module.tokenizer, PreTrainedTokenizerBase)

    # Encoding should work end-to-end.
    sentences = [
        "صلاة الجمعة في المسجد",  # anchor
        "الصلاة في الجامع",  # synonym
        "السباحة في البحر",  # irrelevant
    ]
    emb = model.encode(sentences, normalize_embeddings=True)
    assert emb.shape == (3, model.get_embedding_dimension())

    # Sanity: synonym should be closer to the anchor than the irrelevant.
    sim = emb @ emb.T
    assert sim[0, 1] > sim[0, 2], (
        f"anchor-vs-synonym ({sim[0, 1]:.3f}) should exceed "
        f"anchor-vs-irrelevant ({sim[0, 2]:.3f})"
    )
