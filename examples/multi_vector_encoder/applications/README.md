# Applications

MultiVectorEncoder models score with the MaxSim late-interaction operator, which keeps token-level matching information that single-vector models discard. In this folder you find example scripts that show how to use that in practice.

## Semantic Search

[semantic_search.py](semantic_search.py) encodes a corpus once, then scores queries against it with MaxSim. Because every document is stored as a sequence of token vectors rather than one vector, this trades a larger index footprint for stronger retrieval, and it is the simplest way to see the ranking quality on your own data.

## Retrieve & Rerank

[retrieve_rerank.py](retrieve_rerank.py) combines a fast bi-encoder first stage with a MultiVectorEncoder second stage: the bi-encoder narrows a large corpus down to a handful of candidates, and the multi-vector model rescores only those. This keeps the index small while still paying for late interaction where it matters, and the script prints the timings of both stages so you can see the tradeoff.

## Interpretability

Because MaxSim scores are sums of per-query-token maxima, a multi-vector ranking can be traced back to the exact tokens (or image patches) that produced it. [heatmap.py](../interpretability/heatmap.py) shows the standard ColPali visualization: for a query and a page image, it overlays heatmaps showing which patches contribute most to the ranking score, summed and per query token. Useful for spot-checking why a retrieval ranking surfaced (or missed) a page. [text_similarity_map.py](../interpretability/text_similarity_map.py) is the text-only counterpart: it ranks a corpus the way semantic_search.py does, then splits the top document's score across the query tokens that earned it and prints the document with those tokens highlighted in place. The underlying utilities live in `sentence_transformers.multi_vector_encoder.interpretability`.

## Compression

[token_pooling.py](../compression/token_pooling.py) shrinks the document index with `HierarchicalTokenPooling`, which clusters each document's token vectors and mean-pools each cluster. The script compares pool factors and prints the token-count reduction next to the MaxSim score drift, and shows the three ways to apply a pooling: per encode call, standalone on cached embeddings, or baked into the model as a pipeline module.

For visual document retrieval (matching text queries against page images directly, skipping OCR), see the [multimodal training examples](../training/multimodal/README.md).
