"""
Compute image embeddings
"""

from __future__ import annotations

import os

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
