"""
Compute image embeddings
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from sentence_transformers import SentenceTransformer, util


def test_simple_encode(clip_vit_b_32_model: SentenceTransformer) -> None:
    model = clip_vit_b_32_model
    # Encode an image:
    image_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../examples/sentence_transformer/applications/image-search/two_dogs_in_snow.jpg",
    )
    img_emb = model.encode(Image.open(image_filepath))

    # Encode text descriptions
    text_emb = model.encode(["Two dogs in the snow", "A cat on a table", "A picture of London at night"])

    # Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)[0]
    assert abs(cos_scores[0] - 0.3069) < 0.01
    assert abs(cos_scores[1] - 0.1010) < 0.01
    assert abs(cos_scores[2] - 0.1086) < 0.01


@pytest.fixture()
def tiny_clip_model(tmp_path) -> SentenceTransformer:
    """A randomly initialized CLIP checkpoint saved to a local directory.

    Building the config, model, image processor, and tokenizer from scratch keeps this
    test fully offline while still exercising the real ``Transformer`` loading path.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        CLIPConfig,
        CLIPImageProcessor,
        CLIPModel,
        CLIPTextConfig,
        CLIPVisionConfig,
        PreTrainedTokenizerFast,
    )

    config = CLIPConfig(
        text_config=CLIPTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=2,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
        ),
        projection_dim=32,
    )
    model_dir = str(tmp_path / "tiny_clip")
    CLIPModel(config).save_pretrained(model_dir)
    CLIPImageProcessor(
        do_resize=True,
        size={"shortest_edge": 8},
        do_center_crop=True,
        crop_size={"height": 8, "width": 8},
    ).save_pretrained(model_dir)

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for i, word in enumerate(["product", "small", "outdoor", "balcony", "set", "cat", "dog"], start=3):
        vocab[word] = i
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<unk>",
        model_max_length=32,
    ).save_pretrained(model_dir)

    return SentenceTransformer(model_dir, local_files_only=True)


def test_encode_mixed_modalities(tiny_clip_model: SentenceTransformer) -> None:
    """CLIP-style models encode mixed text+image batches by splitting per modality."""
    model = tiny_clip_model
    rng = np.random.RandomState(0)
    images = [Image.fromarray(rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)) for _ in range(3)]
    texts = ["product", "small outdoor balcony set", "cat"]

    # Interleave the modalities so the mixed batch cannot be a single contiguous slice.
    inputs = [images[0], texts[0], images[1], texts[1], images[2], texts[2]]
    embeddings = model.encode(inputs, batch_size=2)
    assert embeddings.shape == (6, 32)

    image_embeddings = model.encode(images)
    text_embeddings = model.encode(texts)
    np.testing.assert_allclose(embeddings[0::2], image_embeddings, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(embeddings[1::2], text_embeddings, rtol=1e-5, atol=1e-6)
