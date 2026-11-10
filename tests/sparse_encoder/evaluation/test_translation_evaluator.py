from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseTranslationEvaluator
from sentence_transformers.sparse_encoder.modules import SparseStaticEmbedding


@pytest.mark.parametrize("num_sentences", [17, 1025])
@pytest.mark.parametrize("similarity_fn_name", ["dot", "cosine"])
def test_sparse_translation_matches_full_matrix(num_sentences, similarity_fn_name):
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3, "gamma": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    )
    model = SparseEncoder(
        modules=[SparseStaticEmbedding(tokenizer, weight=torch.tensor([0.0, 0.0, 2.0, 1.0, 2.0]))],
        similarity_fn_name=similarity_fn_name,
        device="cpu",
    )
    model.model_card_data.generate_widget_examples = False
    sources = ["alpha beta"] * (num_sentences - 1) + ["gamma"]
    targets = ["alpha beta gamma"] * (num_sentences - 1) + ["alpha"]
    source_embeddings = model.encode(sources, convert_to_tensor=False, convert_to_sparse_tensor=True)
    target_embeddings = model.encode(targets, convert_to_tensor=False, convert_to_sparse_tensor=True)
    assert all(embedding.is_sparse for embedding in source_embeddings + target_embeddings)
    scores = model.similarity(torch.stack(source_embeddings), torch.stack(target_embeddings))
    expected_forward = (
        sum(targets[i] == targets[j] for i, j in enumerate(scores.argmax(dim=1).tolist())) / num_sentences
    )
    expected_reverse = (
        sum(sources[i] == sources[j] for i, j in enumerate(scores.argmax(dim=0).tolist())) / num_sentences
    )

    metrics = SparseTranslationEvaluator(sources, targets)(model)

    assert metrics["src2trg_accuracy"] == expected_forward
    assert metrics["trg2src_accuracy"] == expected_reverse
    assert metrics["src2trg_accuracy"] == ((num_sentences - 1) / num_sentences if similarity_fn_name == "dot" else 0.0)
    assert metrics["active_dims"] > 0
