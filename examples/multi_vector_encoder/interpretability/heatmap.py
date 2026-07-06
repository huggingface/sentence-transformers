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
5. Saves a max-aggregated heatmap overlay as ``heatmap.png``.

Per-family ``n_patches``:

* **PaliGemma at 448x448**: ``(32, 32)`` = ``image_seq_length=1024`` image tokens. Read it from
  ``processor.image_seq_length`` if unsure: ``n = isqrt(processor.image_seq_length); n_patches=(n, n)``.
* **Qwen2-VL family**: depends on ``smart_resize`` per image; call
  ``processor.get_n_patches(image.size, spatial_merge_size=2)`` from ``colpali_engine`` for the
  matching processor and pass the result.
* **Idefics3 / SmolVLM**: same processor pattern.
"""

from __future__ import annotations

from math import isqrt

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.interpretability import (
    maxsim_heatmap,
    real_query_token_slice,
)
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask


def main() -> None:
    # Any PaliGemma-based ColPali checkpoint. The ``-st`` variants are saved with the right
    # sentence-transformers config; the ``-hf`` variants also work with a slight load tweak.
    model = MultiVectorEncoder("tomaarsen/colpali-v1.3-merged-st", trust_remote_code=True)

    # Restrict the document mask to image-patch tokens so the doc embedding lines up 1:1 with
    # the n_patches grid (no text-prefix tokens to filter out via image_mask later).
    mask_module = next(m for m in model if isinstance(m, MultiVectorMask))
    mask_module.keep_only_token_ids = [model.processor.image_token_id]

    # PaliGemma 3B at 448x448 has 1024 image patches arranged 32x32.
    side = isqrt(model.processor.image_seq_length)
    n_patches = (side, side)

    image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ettin-reranker/mteb_ndcg10_embeddinggemma-300m.png"
    query = "What's the title of the Figure"

    query_embeddings = model.encode_query([query], convert_to_tensor=True)
    document_embeddings = model.encode_document([image], convert_to_tensor=True)

    # Drop the bos prefix and trailing expansion tokens (both carry strong attention-sink signals).
    real_query_embedding = query_embeddings[0][real_query_token_slice(model, query)]

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

    prompt = (model.prompts or {}).get("query", "") or ""
    tokens = model.processor.tokenizer.convert_ids_to_tokens(
        model.processor.tokenizer(prompt + query, return_tensors="pt")["input_ids"][0]
    )
    for i, (tok, overlay) in enumerate(zip(tokens, per_token)):
        overlay.save(f"heatmap_token{i:02d}_{tok}.png")


if __name__ == "__main__":
    main()
