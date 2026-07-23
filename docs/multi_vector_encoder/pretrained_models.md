# Pretrained Models

```{eval-rst}
Late-interaction models with the ``sentence-transformers`` tag on the Hugging Face Hub work out of
the box with :class:`~sentence_transformers.MultiVectorEncoder`:

* **Community models**: `Multi-vector models on Hugging Face <https://huggingface.co/models?library=sentence-transformers&other=multi-vector>`_.

Models integrate seamlessly with this simple interface:
```

```python
from sentence_transformers import MultiVectorEncoder

# Download from the 🤗 Hub
model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

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
# tensor([[9.6037, 9.4055]])
```

## Text Retrieval Models

These checkpoints load with their trained prefix tokens, query expansion, and punctuation skiplist
recovered from the saved configuration:

| Model Name | Backbone |
|---|---|
| [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | gte-modernbert-base |
| [lightonai/Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT) | ModernBERT-base |
| [mixedbread-ai/mxbai-edge-colbert-v0-32m](https://huggingface.co/mixedbread-ai/mxbai-edge-colbert-v0-32m) | ModernBERT (32M) |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | BERT (33M) |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | bert-base-uncased |
| [lightonai/colbertv2.0](https://huggingface.co/lightonai/colbertv2.0) | bert-base-uncased |

Many more community checkpoints are available on the Hub, e.g. under the
[multi-vector tag](https://huggingface.co/models?library=sentence-transformers&other=multi-vector).

## Visual Document Retrieval Models

```{eval-rst}
ColPali-style models embed page *images* as documents and text as queries, skipping OCR entirely
(see the `ViDoRe benchmark <https://huggingface.co/vidore>`_ family). The transformers-native
``*ForRetrieval`` checkpoints are auto-detected; their processor formats queries and image documents,
and the projection and normalisation run inside the model.
```

| Model Name | Backbone | Architecture |
|---|---|---|
| [vidore/colqwen2-v1.0-hf](https://huggingface.co/vidore/colqwen2-v1.0-hf) | Qwen2-VL-2B | ColQwen2ForRetrieval |
| [vidore/colpali-v1.3-hf](https://huggingface.co/vidore/colpali-v1.3-hf) | PaliGemma-3B | ColPaliForRetrieval |
| [vidore/colpali-v1.2-hf](https://huggingface.co/vidore/colpali-v1.2-hf) | PaliGemma-3B | ColPaliForRetrieval |
| [ModernVBERT/colmodernvbert](https://huggingface.co/ModernVBERT/colmodernvbert) | ModernVBERT (250M) | ColModernVBertForRetrieval |
