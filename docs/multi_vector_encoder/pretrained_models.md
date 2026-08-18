# Pretrained Models

```{eval-rst}
The `sentence-transformers tag <https://huggingface.co/models?library=sentence-transformers&other=multi-vector>`_
on the Hugging Face Hub is the list that stays current, and we are working to get it onto every model that works
with :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`. The tables below are what we test against directly, so
treat them as a starting point rather than the full set. For text retrieval in particular, any PyLate or
Stanford-NLP ColBERT checkpoint loads whether or not it carries the tag yet.

Models integrate seamlessly with this simple interface:
```

```python
from sentence_transformers import MultiVectorEncoder

# Download from the 🤗 Hub
model = MultiVectorEncoder("lightonai/LateOn")

# Run inference
queries = ["What is the capital of France?"]
documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# (10, 128) (9, 128) - one 128-dimensional vector per token

# Get the late-interaction (MaxSim) similarity scores for the embeddings
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[9.1129, 8.8769]])
```

## Text Retrieval Models

These load with their trained prefix tokens, query expansion, and punctuation skiplist recovered from the saved configuration. Where a `revision` is listed, pass it until the pull request on that repository is merged, after which the plain model name is enough.

The NanoBEIR column reports the mean NDCG@10 (higher is better) across the 13 [NanoBEIR datasets](https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en), each a 50-query subsample of a BEIR dataset, as a fast proxy for English text retrieval quality. We used the `MultiVectorNanoBEIREvaluator` to compute the scores for the primarily-English models. A `-` means the model was not evaluated on it. Note that NanoBEIR is a small benchmark, and its scores aren't a substitute for evaluating on your own data, which is always the right way to pick a model.

| Model | Parameters | Dimensionality | NanoBEIR | Notes |
| --- | :---: | :---: | :---: | --- |
| [lightonai/LateOn-regularized](https://huggingface.co/lightonai/LateOn-regularized) | 149M | 128 | 0.6897 | - |
| [lightonai/LateOn-hpool-regularized](https://huggingface.co/lightonai/LateOn-hpool-regularized) | 149M | 128 | 0.6876 | - |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | 128 | 0.6868 | - |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M | 128 | 0.6864 | needs `trust_remote_code=True` |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 307M | 128 | 0.6851 | - |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 149M | 128 | 0.6720 | - |
| [topk-io/Iso-ModernColBERT](https://huggingface.co/topk-io/Iso-ModernColBERT) | 149M | 128 | 0.6687 | - |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M | 128 | 0.6662 | needs `trust_remote_code=True` |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | 149M | 128 | 0.6569 | - |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M | 96 | 0.6550 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 32M | 64 | 0.6524 | - |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 353M | 128 | 0.6441 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 17M | 48 | 0.6407 | - |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | 110M | 128 | 0.6201 | - |
| [lightonai/LateOn-Code](https://huggingface.co/lightonai/LateOn-Code) | 149M | 128 | 0.6169 | - |
| [lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT) | 149M | 128 | 0.6164 | - |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | 149M | 128 | 0.6078 | - |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 110M | 128 | 0.6053 | - |
| [VAGOsolutions/SauerkrautLM-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-EuroColBERT) | 212M | 128 | 0.5982 | - |
| [antoinelouis/colbert-xm](https://huggingface.co/antoinelouis/colbert-xm) | 853M | 128 | 0.5915 | - |
| [VAGOsolutions/SauerkrautLM-Multi-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT) | 149M | 128 | 0.5886 | - |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | 335M | 128 | 0.5733 | `revision="refs/pr/4"` |
| [lightonai/LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) | 17M | 48 | 0.5274 | - |
| [VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT) | 149M | 128 | 0.5267 | - |
| [VAGOsolutions/SauerkrautLM-Reason-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Reason-EuroColBERT) | 212M | 128 | 0.4479 | - |
| [NeuML/biomedbert-base-colbert](https://huggingface.co/NeuML/biomedbert-base-colbert) | 110M | 128 | 0.4320 | - |
| [yjoonjang/colbert-ko-v1](https://huggingface.co/yjoonjang/colbert-ko-v1) | 149M | 128 | - | - |
| [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert) | 111M | 256 | - | - |
| [samheym/GerColBERT](https://huggingface.co/samheym/GerColBERT) | 110M | 128 | - | - |

## Visual Document Retrieval Models

ColPali-style models embed page images as documents and text as queries.

The NanoViDoRe column reports the mean NDCG@10 (higher is better) across [NanoViDoRe v3](https://huggingface.co/datasets/lightonai/NanoViDoRe_v3), a compact visual document retrieval benchmark spanning 8 subsets (computer science, energy, finance in English and French, HR, industrial, pharmaceuticals, and physics). Like with NanoBEIR, NanoViDoRe is a small benchmark which shouldn't replace evaluation on your own data.

| Model | Parameters | Dimensionality | NanoViDoRe | Notes |
| --- | :---: | :---: | :---: | --- |
| [webAI-Official/webAI-ColVec1.1-8b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-8b) | 8.4B | 640 | 0.6580 | needs `trust_remote_code=True` |
| [webAI-Official/webAI-ColVec1.1-4b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-4b) | 4.5B | 640 | 0.6520 | needs `trust_remote_code=True` |
| [tencent/EVIE-Preview-4.5B](https://huggingface.co/tencent/EVIE-Preview-4.5B) | 4.54B | 128 | 0.6405 | - |
| [TomoroAI/tomoro-colqwen3-embed-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b) | 8.8B | 320 | 0.6206 | needs `trust_remote_code=True` |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 4.4B | 320 | 0.6019 | needs `trust_remote_code=True` |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 3.8B | 128 | 0.5402 | - |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 3.8B | 128 | 0.5395 | - |
| [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) | 4.4B | 128 | 0.5309 | - |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 2.9B | 128 | 0.4802 | - |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | 2.9B | 128 | 0.4793 | - |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 2.9B | 128 | 0.4691 | - |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 2.2B | 128 | 0.4685 | - |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 2.2B | 128 | 0.4526 | - |
| [vidore/colpali](https://huggingface.co/vidore/colpali) | 2.9B | 128 | 0.4516 | - |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 2.9B | 128 | 0.4314 | - |
| [vidore/colsmolvlm-v0.1](https://huggingface.co/vidore/colsmolvlm-v0.1) | 2.1B | 128 | 0.4054 | - |
| [vidore/colpali-hard-v1.1](https://huggingface.co/vidore/colpali-hard-v1.1) | 2.9B | 128 | 0.3949 | - |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M) | 507M | 128 | 0.3459 | - |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M) | 256M | 128 | 0.2673 | - |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 252M | 128 | 0.2632 | - |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | 2.9B | 128 | - | - |
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | 2.2B | 128 | - | - |

Most of these are LoRA adapter repositories, with the adapter applied directly onto its base at load time. Some also have a `-merged` sibling on the Hub (e.g. [vidore/colpali-v1.3-merged](https://huggingface.co/vidore/colpali-v1.3-merged)) with the adapter already folded into the weights.

The three `-hf` entries are the transformers-native `*ForRetrieval` ports. They load without any configuration, but use more modeling from `transformers` and less from `sentence_transformers`. Generally, it's preferable to use the original models instead, as the ports score approximately the same.
