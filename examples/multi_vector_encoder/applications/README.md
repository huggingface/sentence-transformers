# Applications

MultiVectorEncoder models score with the MaxSim late-interaction operator, which keeps token-level matching information that single-vector models discard. In this folder you find example scripts that show how to use that in practice.

## Semantic Search

[semantic_search.py](semantic_search.py) encodes a corpus once, then scores queries against it with MaxSim. Because every document is stored as a sequence of token vectors rather than one vector, this trades a larger index footprint for stronger retrieval, and it is the simplest way to see the ranking quality on your own data.

## Retrieve & Rerank

[retrieve_rerank.py](retrieve_rerank.py) combines a fast bi-encoder first stage with a MultiVectorEncoder second stage: the bi-encoder narrows a large corpus down to a handful of candidates, and the multi-vector model rescores only those. This keeps the index small while still paying for late interaction where it matters, and the script prints the timings of both stages so you can see the tradeoff.

For visual document retrieval (matching text queries against page images directly, skipping OCR), see the [multimodal training examples](../training/multimodal/README.md).
