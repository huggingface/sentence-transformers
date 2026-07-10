"""Render a per-query-token MaxSim heatmap onto an image-document for a ColPali-style model.

This is the standard ColPali interpretability viz: for a given query and an image-document,
overlay a heatmap on the image showing which patch positions contribute most to the MaxSim
ranking. Useful for spot-checking why a retrieval ranking surfaced (or didn't surface) a page.

By default this script:

1. Loads a ColPali (PaliGemma backbone) model from the Hub.
2. Enables :class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask`'s
   ``keep_only_token_ids`` filter so the document embedding only contains image-patch tokens.
3. Encodes one query and one image-document.
4. Computes per-query-token similarity over the 32x32 PaliGemma image grid.
5. Saves a sum-aggregated heatmap overlay (per-patch contribution to the MaxSim ranking score) as
   ``heatmap_reranker.png``, plus one overlay per real query token.

The patch grid shape is inferred from the model's processor via
:func:`~sentence_transformers.multi_vector_encoder.interpretability.get_n_patches`:

* **PaliGemma at 448x448**: fixed ``(32, 32)`` grid = ``image_seq_length=1024`` image tokens.
* **Qwen2-VL family**: dynamic, depends on ``smart_resize`` of each input image's size.
* **Idefics3 / SmolVLM**: not supported yet (the split-image token order is not a plain grid).
"""

from __future__ import annotations

from transformers.image_utils import load_image

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.interpretability import (
    get_n_patches,
    maxsim_heatmap,
    real_query_token_slice,
)
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask


def main() -> None:
    # Any PaliGemma-based ColPali checkpoint.
    model = MultiVectorEncoder("tomaarsen/colpali-v1.3-merged-st", trust_remote_code=True)

    # Restrict the document mask to image-patch tokens so the doc embedding lines up 1:1 with
    # the n_patches grid (no text-prefix tokens to filter out via image_mask later).
    mask_module = next(m for m in model if isinstance(m, MultiVectorMask))
    mask_module.keep_only_token_ids = [model.processor.image_token_id]

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_embeddinggemma-300m.png"
    image = load_image(image_url)
    query = "What's the title of the Figure"

    # Infer the (n_patches_x, n_patches_y) grid from the model's processor. Fixed 32x32 for
    # PaliGemma, computed from the image size for dynamic-grid models like ColQwen2.
    n_patches = get_n_patches(model, image.size)

    # Encode the query with output_value=None to get the raw per-input features: both the token
    # embeddings and the input ids that produced them, for labeling the per-token heatmaps below.
    query_outputs = model.encode_query([query], output_value=None)[0]
    document_embeddings = model.encode_document([image], convert_to_tensor=True)

    # Drop the bos prefix and trailing expansion tokens (both carry strong attention-sink signals).
    query_slice = real_query_token_slice(model, query)
    real_query_embedding = query_outputs["token_embeddings"][query_slice]

    overlay = maxsim_heatmap(
        image=image,
        query_embedding=real_query_embedding,
        image_embedding=document_embeddings[0],
        n_patches=n_patches,
    )
    overlay.save("heatmap_reranker.png")
    print(f"Saved heatmap_reranker.png ({overlay.size[0]}x{overlay.size[1]}, RGBA)")

    # Per-query-token heatmaps: one PIL Image per real query token, e.g. for tooling that
    # animates token-by-token attention.
    per_token = maxsim_heatmap(
        image=image,
        query_embedding=real_query_embedding,
        image_embedding=document_embeddings[0],
        n_patches=n_patches,
        aggregate="none",
    )
    print(f"Per-query-token: {len(per_token)} heatmaps available.")

    # Label each map with its token: the ids come from the same encode call that produced the
    # embeddings, sliced the same way, so label i matches heatmap i exactly.
    tokens = model.tokenizer.convert_ids_to_tokens(query_outputs["input_ids"][query_slice])
    for i, (tok, overlay) in enumerate(zip(tokens, per_token)):
        overlay.save(f"heatmap_token{i:02d}_{tok}.png")


if __name__ == "__main__":
    main()
