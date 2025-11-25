# quantization

`sentence_transformers.quantization` defines different helpful functions to perform embedding quantization.

```{eval-rst}
.. note::
   `Embedding Quantization <../../../examples/sentence_transformer/applications/embedding-quantization/README.html>`_ differs from model quantization. The former shrinks the size of embeddings such that semantic search/retrieval is faster and requires less memory and disk space. The latter refers to lowering the precision of the model weights to speed up inference. This page only shows documentation for the former.
```

```{eval-rst}
.. automodule:: sentence_transformers.quantization
   :members: quantize_embeddings, semantic_search_faiss, semantic_search_usearch
```
