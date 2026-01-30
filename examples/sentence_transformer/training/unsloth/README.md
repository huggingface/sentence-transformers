# Training with Unsloth

Sentence Transformers integrates with [Unsloth](https://unsloth.ai/docs), an open-source training framework that focuses on fast, memory-efficient fine-tuning for encoder-style models (including many Sentence Transformers checkpoints). Unsloth can train embedding, classifier, BERT, and reranker models with high throughput and long context while keeping VRAM usage low.

This directory contains examples of how to use Unsloth together with Sentence Transformers. If you are looking for the more general PEFT-based adapter workflow built directly on top of Sentence Transformers, see the [PEFT training README](../peft/README.md).

Unsloth builds on top of Hugging Face `sentence-transformers` to support most Sentence Transformers compatible models like Qwen3-Embedding, EmbeddingGemma, BERT, and more. Models like `google/embeddinggemma-300m` can be fine-tuned on GPUs with as little as 3 GB of VRAM and then used anywhere: Sentence Transformers, Transformers, LangChain, LlamaIndex, Haystack, Txtai, vLLM, llama.cpp, TEI, FAISS/vector DBs, and more.

## Examples in this repository

The following scripts demonstrate Unsloth-based fine-tuning for Sentence Transformers models:

- [training_gooaq_unsloth.py](training_gooaq_unsloth.py): Fine-tunes a BERT-style encoder on the GooAQ question-answer dataset using `FastSentenceTransformer` and a LoRA adapter, with `CachedMultipleNegativesRankingLoss` and NanoBEIR evaluation.
- [training_medical_unsloth.py](training_medical_unsloth.py): Fine-tunes `google/embeddinggemma-300m` on the MIRIAD medical retrieval dataset using LoRA, InformationRetrievalEvaluator, and GPU memory tracking for LoRA-specific overhead.

For PEFT-based training that does not rely on Unsloth, see:

- [PEFT training with adapters](../peft/README.md): Native PEFT integration in Sentence Transformers (LoRA and other adapter types) with examples such as `training_gooaq_lora.py`.

## Unsloth Colab notebooks

The Unsloth project maintains several Colab notebooks that combine their training pipeline with Sentence Transformers-compatible models:

- [EmbeddingGemma (300M)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/EmbeddingGemma_(300M).ipynb)
- [Qwen3-Embedding 4B](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(4B).ipynb) • [0.6B](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_Embedding_(0_6B).ipynb)
- [BGE M3](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/BGE_M3.ipynb)
- [ModernBERT-large](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ModernBert.ipynb)
- [All-MiniLM-L6-v2](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/All_MiniLM_L6_v2.ipynb)
- [GTE ModernBert](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ModernBert.ipynb)

## Fine-tuning via FastSentenceTransformer

The Unsloth integration for Sentence Transformers is centered around `FastSentenceTransformer`, which mirrors the usual Sentence Transformers API but adds optimized training paths. For models without `modules.json`, we fall back to the default Sentence Transformers pooling modules. If you use custom heads or nonstandard pooling, double-check the resulting embeddings.

Key save/push methods on `FastSentenceTransformer`:

- `save_pretrained()`: Saves LoRA adapters to a local folder.
- `save_pretrained_merged()`: Saves the merged model (base model + adapters) to a local folder.
- `push_to_hub()`: Pushes LoRA adapters to the Hugging Face Hub.
- `push_to_hub_merged()`: Pushes the merged model to the Hugging Face Hub.

One important detail: when loading a model for inference with `FastSentenceTransformer`, you must pass `for_inference=True`:

```python
from unsloth import FastSentenceTransformer

model = FastSentenceTransformer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    for_inference=True,
)
```

For Hugging Face Hub authorization, you can run:

```bash
hf auth login
```

inside the same environment before calling the Hub methods. After that, `push_to_hub()` and `push_to_hub_merged()` do not require an explicit token argument.

## Inference and deployment

Once you have fine-tuned a model with Unsloth, you can load it back into Sentence Transformers or any other compatible stack and run inference as usual. For example, when you have pushed a LoRA adapter to the Hub, you can:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("<your-unsloth-finetuned-model>")

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

query_embedding = model.encode_query(query)
document_embedding = model.encode_document(documents)
similarity = model.similarity(query_embedding, document_embedding)
```

Your Unsloth-finetuned model can then be deployed anywhere Sentence Transformers models work: custom APIs, vector databases, RAG pipelines, or inference servers like TEI and vLLM.

## Benchmarks

Unsloth benchmarks indicate 1.8–3.3× speedups for embedding fine-tuning over strong FlashAttention 2 baselines, across sequence lengths from 128 to 2048 and beyond. For example, `google/embeddinggemma-300m` can be trained with QLoRA on ~3GB VRAM and with LoRA on ~6GB VRAM.

For up-to-date benchmark visualizations and details, see the Unsloth documentation: [Embedding fine-tuning benchmarks](https://unsloth.ai/docs/new/embedding-finetuning#unsloth-benchmarks).

### Resources

For more information, see:

- [Unsloth embedding fine-tuning documentation](https://unsloth.ai/docs/new/embedding-finetuning)
- [Unsloth GitHub repository](https://github.com/unslothai/unsloth)
