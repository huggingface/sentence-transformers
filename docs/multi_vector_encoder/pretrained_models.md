# Pretrained Models

```{eval-rst}
Several Multi-Vector Encoder models have been publicly released on the Hugging Face Hub:

* **Community models**: `All Multi-Vector Encoder models on Hugging Face <https://huggingface.co/models?library=sentence-transformers&other=multi-vector>`_.

.. note::
    The ``sentence-transformers`` library tag is shared by all four model types, so pair it with the ``multi-vector`` tag to find only the models that load with :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`. ``cross-encoder`` and ``sparse`` are the equivalents for Cross Encoder and Sparse Encoder models, while dense Sentence Transformer models make up the bulk of what the library tag returns on its own.

That community list is the one that stays current, and we are working to get the ``multi-vector`` tag onto every model that works with :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`. The tables below are what we test against directly, so treat them as a starting point rather than the full set. For text retrieval in particular, any PyLate or Stanford-NLP ColBERT checkpoint loads whether or not it carries the tag yet.

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
| [chungimungi/GLInt](https://huggingface.co/chungimungi/GLInt) | 149M | 128 | 0.6914 | - |
| [lightonai/LateOn-regularized](https://huggingface.co/lightonai/LateOn-regularized) | 149M | 128 | 0.6897 | - |
| [lightonai/LateOn-hpool-regularized](https://huggingface.co/lightonai/LateOn-hpool-regularized) | 149M | 128 | 0.6876 | - |
| [lightonai/LateOn](https://huggingface.co/lightonai/LateOn) | 149M | 128 | 0.6868 | - |
| [LiquidAI/LFM2.5-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2.5-ColBERT-350M) | 353M | 128 | 0.6864 | needs `trust_remote_code=True` |
| [lightonai/mLateOn](https://huggingface.co/lightonai/mLateOn) | 307M | 128 | 0.6851 | - |
| [lightonai/ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) | 149M | 128 | 0.6824 | - |
| [VAGOsolutions/SauerkrautLM-Multi-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-ModernColBERT) | 149M | 128 | 0.6741 | - |
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | 149M | 128 | 0.6720 | - |
| [topk-io/Iso-ModernColBERT](https://huggingface.co/topk-io/Iso-ModernColBERT) | 149M | 128 | 0.6687 | - |
| [perplexity-ai/pplx-embed-v1-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-late-0.6b) | 596M | 128 | 0.6662 | needs `trust_remote_code=True` |
| [VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT) | 149M | 128 | 0.6616 | - |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | 33M | 96 | 0.6550 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | 32M | 64 | 0.6524 | - |
| [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 559M | 128 | 0.6517 | needs `trust_remote_code=True` |
| [DataScience-UIBK/Reason-mxbai-colbert-v0.1-32m](https://huggingface.co/DataScience-UIBK/Reason-mxbai-colbert-v0.1-32m) | 32M | 128 | 0.6486 | - |
| [LiquidAI/LFM2-ColBERT-350M](https://huggingface.co/LiquidAI/LFM2-ColBERT-350M) | 353M | 128 | 0.6441 | - |
| [lightonai/ColBERT-Zero-unsupervised-noprompts](https://huggingface.co/lightonai/ColBERT-Zero-unsupervised-noprompts) | 149M | 128 | 0.6430 | - |
| [mixedbread-ai/mxbai-edge-colbert-v0-17m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-17m) | 17M | 48 | 0.6407 | - |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | 110M | 128 | 0.6201 | - |
| [lightonai/LateOn-Code](https://huggingface.co/lightonai/LateOn-Code) | 149M | 128 | 0.6169 | - |
| [lightonai/Agent-ModernColBERT](https://huggingface.co/lightonai/Agent-ModernColBERT) | 149M | 128 | 0.6164 | - |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | 149M | 128 | 0.6078 | - |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | 110M | 128 | 0.6053 | - |
| [VAGOsolutions/SauerkrautLM-Reason-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-Reason-EuroColBERT) | 212M | 128 | 0.6039 | - |
| [VAGOsolutions/SauerkrautLM-EuroColBERT](https://huggingface.co/VAGOsolutions/SauerkrautLM-EuroColBERT) | 212M | 128 | 0.5965 | - |
| [antoinelouis/colbert-xm](https://huggingface.co/antoinelouis/colbert-xm) | 853M | 128 | 0.5915 | - |
| [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1) | 335M | 128 | 0.5733 | - |
| [lightonai/LateOn-Code-edge](https://huggingface.co/lightonai/LateOn-Code-edge) | 17M | 48 | 0.5274 | - |
| [NeuML/biomedbert-base-colbert](https://huggingface.co/NeuML/biomedbert-base-colbert) | 110M | 128 | 0.4320 | - |
| [NeuML/colbert-bert-tiny](https://huggingface.co/NeuML/colbert-bert-tiny) | 4M | 128 | 0.4035 | - |
| [yjoonjang/colbert-ko-v1](https://huggingface.co/yjoonjang/colbert-ko-v1) | 149M | 128 | - | - |
| [ytu-ce-cosmos/turkish-colbert](https://huggingface.co/ytu-ce-cosmos/turkish-colbert) | 111M | 256 | - | - |
| [samheym/GerColBERT](https://huggingface.co/samheym/GerColBERT) | 110M | 128 | - | - |
| [bclavie/JaColBERT](https://huggingface.co/bclavie/JaColBERT) | 111M | 128 | - | - |
| [bclavie/JaColBERTv2](https://huggingface.co/bclavie/JaColBERTv2) | 111M | 128 | - | - |
| [answerdotai/JaColBERTv2.4](https://huggingface.co/answerdotai/JaColBERTv2.4) | 111M | 128 | - | `revision="refs/pr/2"` |
| [answerdotai/JaColBERTv2.5](https://huggingface.co/answerdotai/JaColBERTv2.5) | 111M | 128 | - | `revision="refs/pr/5"` |

## Visual Document Retrieval Models

ColPali-style models embed page images as documents and text as queries.

The NanoViDoRe column reports the mean NDCG@10 (higher is better) across [NanoViDoRe v3](https://huggingface.co/datasets/lightonai/NanoViDoRe_v3), a compact visual document retrieval benchmark spanning 8 subsets (computer science, energy, finance in English and French, HR, industrial, pharmaceuticals, and physics). Like with NanoBEIR, NanoViDoRe is a small benchmark which shouldn't replace evaluation on your own data.

| Model | Parameters | Dimensionality | NanoViDoRe | Notes |
| --- | :---: | :---: | :---: | --- |
| [webAI-Official/webAI-ColVec1.1-8b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-8b) | 8.4B | 640 | 0.6580 | needs `trust_remote_code=True` |
| [webAI-Official/webAI-ColVec1.1-4b](https://huggingface.co/webAI-Official/webAI-ColVec1.1-4b) | 4.5B | 640 | 0.6520 | needs `trust_remote_code=True` |
| [vultr/VultronRetrieverPrime-Qwen3.5-8B](https://huggingface.co/vultr/VultronRetrieverPrime-Qwen3.5-8B) | 8.39B | 320 | 0.6423 | - |
| [vultr/VultronRetrieverCore-Qwen3.5-4.5B](https://huggingface.co/vultr/VultronRetrieverCore-Qwen3.5-4.5B) | 4.54B | 320 | 0.6410 | - |
| [tencent/EVIE-Preview-4.5B](https://huggingface.co/tencent/EVIE-Preview-4.5B) | 4.54B | 128 | 0.6405 | - |
| [nvidia/nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | 8.77B | 4096 | 0.6374 | `revision="refs/pr/4"`, needs `trust_remote_code=True` |
| [athrael-soju/colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 4.54B | 320 | 0.6358 | - |
| [TomoroAI/tomoro-colqwen3-embed-8b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b) | 8.8B | 320 | 0.6206 | needs `trust_remote_code=True` |
| [nvidia/nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | 4.83B | 2560 | 0.6200 | `revision="refs/pr/7"`, needs `trust_remote_code=True` |
| [OpenSearch-AI/Ops-Colqwen3-4B](https://huggingface.co/OpenSearch-AI/Ops-Colqwen3-4B) | 4.44B | 2560 | 0.6150 | `revision="refs/pr/5"`, needs `trust_remote_code=True` |
| [nvidia/llama-nemotron-colembed-vl-3b-v2](https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2) | 4.41B | 3072 | 0.6112 | `revision="refs/pr/3"`, needs `trust_remote_code=True` |
| [tsystems/colqwen2.5-3b-multilingual-v1.0](https://huggingface.co/tsystems/colqwen2.5-3b-multilingual-v1.0) | 3.75B | 128 | 0.6039 | - |
| [tsystems/colqwen2.5-3b-multilingual-v1.0-merged](https://huggingface.co/tsystems/colqwen2.5-3b-multilingual-v1.0-merged) | 3.75B | 128 | 0.6027 | - |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | 4.4B | 320 | 0.6019 | needs `trust_remote_code=True` |
| [nomic-ai/colnomic-embed-multimodal-7b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b) | 8.29B | 128 | 0.5942 | `revision="refs/pr/3"` |
| [nomic-ai/colnomic-embed-multimodal-3b](https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b) | 3.75B | 128 | 0.5929 | `revision="refs/pr/6"` |
| [VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-8b-v0.1) | 8.15B | 128 | 0.5819 | - |
| [Metric-AI/ColQwen2.5-3b-multilingual-v1.0](https://huggingface.co/Metric-AI/ColQwen2.5-3b-multilingual-v1.0) | 3.75B | 128 | 0.5763 | `revision="refs/pr/2"` |
| [vultr/VultronRetrieverFlash-Qwen3.5-0.8B](https://huggingface.co/vultr/VultronRetrieverFlash-Qwen3.5-0.8B) | 853M | 320 | 0.5693 | - |
| [VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1) | 4.44B | 128 | 0.5656 | - |
| [VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1) | 2.13B | 128 | 0.5530 | - |
| [Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0](https://huggingface.co/Verm1ion/ColTurk-VDR-Qwen3VL-4B-v1.0) | 4.44B | 320 | 0.5434 | `revision="refs/pr/1"` |
| [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | 3.8B | 128 | 0.5402 | - |
| [vidore/colqwen2.5-v0.1](https://huggingface.co/vidore/colqwen2.5-v0.1) | 3.8B | 128 | 0.5395 | - |
| [vidore/colqwen-omni-v0.1](https://huggingface.co/vidore/colqwen-omni-v0.1) | 4.4B | 128 | 0.5309 | - |
| [VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColQwen3-1.7b-Turbo-v0.1) | 1.76B | 128 | 0.5035 | - |
| [vidore/colpali-v1.3](https://huggingface.co/vidore/colpali-v1.3) | 2.9B | 128 | 0.4802 | - |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | 2.9B | 128 | 0.4793 | - |
| [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2) | 2.9B | 128 | 0.4691 | - |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | 2.2B | 128 | 0.4685 | - |
| [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1) | 2.2B | 128 | 0.4526 | - |
| [vidore/colpali](https://huggingface.co/vidore/colpali) | 2.9B | 128 | 0.4516 | - |
| [vidore/colpali-v1.1](https://huggingface.co/vidore/colpali-v1.1) | 2.9B | 128 | 0.4314 | - |
| [VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1) | 451M | 128 | 0.4249 | - |
| [vidore/colsmolvlm-v0.1](https://huggingface.co/vidore/colsmolvlm-v0.1) | 2.1B | 128 | 0.4054 | - |
| [VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1](https://huggingface.co/VAGOsolutions/SauerkrautLM-ColMinistral3-3b-v0.1) | 4.25B | 128 | 0.3978 | - |
| [vidore/colpali-hard-v1.1](https://huggingface.co/vidore/colpali-hard-v1.1) | 2.9B | 128 | 0.3949 | - |
| [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M) | 507M | 128 | 0.3459 | - |
| [vidore/colSmol-256M](https://huggingface.co/vidore/colSmol-256M) | 256M | 128 | 0.2673 | - |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | 252M | 128 | 0.2632 | - |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | 2.9B | 128 | - | - |
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | 2.2B | 128 | - | - |

Most of these are LoRA adapter repositories, with the adapter applied directly onto its base at load time. Some also have a `-merged` sibling on the Hub (e.g. [vidore/colpali-v1.3-merged](https://huggingface.co/vidore/colpali-v1.3-merged)) with the adapter already folded into the weights.

The three `-hf` entries are the transformers-native `*ForRetrieval` ports. They load without any configuration, but use more modeling from `transformers` and less from `sentence_transformers`. Generally, it's preferable to use the original models instead, as the ports score approximately the same.
