# Pretrained Models

```{eval-rst}
The `sentence-transformers tag <https://huggingface.co/models?library=sentence-transformers&other=multi-vector>`_
on the Hugging Face Hub is the list that stays current, and we are working to get it onto every model that works
with :class:`~sentence_transformers.MultiVectorEncoder`. The tables below are what we test against directly, so
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

| Model | Parameters | Backbone | Notes |
| --- | :---: | --- | --- |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 149M | gte-modernbert-base | - |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | 149M | ModernBERT-base | - |
| [lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT) | 149M | ModernBERT-base | - |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | 149M | ModernBERT-base | - |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | ModernBERT-base | - |
| [lightonai/LateOn-hpool-regularized](https://huggingface.co/lightonai/LateOn-hpool-regularized) | 149M | ModernBERT-base | - |
| [lightonai/LateOn-Code](https://huggingface.co/lightonai/LateOn-Code) | 149M | ModernBERT-base (code) | - |
| [lightonai/LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) | 17M | ModernBERT (17M, code, 48-dim output) | - |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 307M | ModernBERT (multilingual) | - |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 353M | LFM2 (350M) | - |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M | LFM2.5 (350M) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 32M | ModernBERT (32M) | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 17M | ModernBERT (17M) | - |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | 335M | bert-large-uncased | `revision="refs/pr/N"` |
| [VAGOsolutions/SauerkrautLM-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-EuroColBERT) | 212M | EuroBERT-210m | - |
| [VAGOsolutions/SauerkrautLM-Reason-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Reason-EuroColBERT) | 212M | EuroBERT-210m | - |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 559M | XLM-RoBERTa (multilingual) | Needs `trust_remote_code=True` |
| [antoinelouis/colbert-xm](https://huggingface.co/antoinelouis/colbert-xm) | 853M | X-MOD (multilingual) | - |
| [yjoonjang/colbert-ko-v1](https://huggingface.co/yjoonjang/colbert-ko-v1) | 149M | ModernBERT (Korean) | - |
| [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert) | 111M | BERT (Turkish, 256-dim output) | - |
| [samheym/GerColBERT](https://huggingface.co/samheym/GerColBERT) | 110M | BERT (German) | - |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M | BERT (33M) | - |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 110M | bert-base-uncased | - |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | 110M | bert-base-uncased | - |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M | Qwen3-0.6B (bidirectional) | Needs `trust_remote_code=True` |

## Visual Document Retrieval Models

ColPali-style models embed page images as documents and text as queries, skipping OCR entirely (see the [ViDoRe benchmark](https://huggingface.co/vidore) family). Each one needs a small Sentence Transformers configuration in its repository, and most of those are open pull requests at the time of writing. Where a `revision` is listed, pass it until the pull request is merged, after which the plain model name is enough:

```python
model = MultiVectorEncoder("vidore/colqwen2.5-v0.2", revision="refs/pr/N")
```

<!-- TODO (v6.0): Replace all "refs/pr/N" before release -->

| Model | Parameters | Backbone | Notes |
| --- | :---: | --- | --- |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"` |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"` |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"` |
| [vidore/colpali](https://huggingface.co/vidore/colpali) | 2.9B | PaliGemma-3B | `revision="refs/pr/N"` |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 2.2B | Qwen2-VL-2B | `revision="refs/pr/N"` |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 2.2B | Qwen2-VL-2B | `revision="refs/pr/N"` |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 3.8B | Qwen2.5-VL-3B | `revision="refs/pr/N"` |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 3.8B | Qwen2.5-VL-3B | `revision="refs/pr/N"` |
| [vidore/colsmolvlm-v0.1](https://huggingface.co/vidore/colsmolvlm-v0.1) | 2.1B | SmolVLM-Instruct | `revision="refs/pr/N"` |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M) | 460M | SmolVLM-500M | `revision="refs/pr/N"` |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M) | 228M | SmolVLM-256M | `revision="refs/pr/N"` |
| [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) | 4.4B | Qwen2.5-Omni-3B (also audio and video) | `revision="refs/pr/N"` |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 252M | ModernVBERT (250M) | `revision="refs/pr/N"` |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 4.4B | Qwen3-VL-4B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [TomoroAI/tomoro-colqwen3-embed-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b) | 8.8B | Qwen3-VL-8B | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [webAI-Official/webAI-ColVec1.1-4b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-4b) | 4.5B | Qwen3.5-4B (bidirectional) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [webAI-Official/webAI-ColVec1.1-8b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-8b) | 8.4B | Qwen3.5-9B (bidirectional) | `revision="refs/pr/N"`, needs `trust_remote_code=True` |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | 2.9B | PaliGemma-3B | - |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | 2.9B | PaliGemma-3B | - |
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | 2.2B | Qwen2-VL-2B | - |

Most of these are LoRA adapter repositories. On `transformers>=5.15.0` they load with the stock `Transformer` and no `trust_remote_code`, since the adapter is applied directly onto its base at load time. Some also have a `-merged` sibling on the Hub (e.g. [vidore/colpali-v1.3-merged](https://huggingface.co/vidore/colpali-v1.3-merged)) with the adapter already folded into the weights.

```{eval-rst}
The three ``-hf`` entries at the bottom are the transformers-native ``*ForRetrieval`` ports, auto-detected on
load. They need no configuration, since the projection and normalization live inside the model, but they are
separate uploads that lag the originals, so prefer the checkpoint the authors publish and maintain.
```
