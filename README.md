<!--- BADGES: START --->

[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/models?library=sentence-transformers)
[![GitHub - License](https://img.shields.io/github/license/UKPLab/sentence-transformers?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentence-transformers?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/sentence-transformers?logo=pypi&style=flat&color=orange)][#pypi-package]
[![Docs - GitHub.io](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=sentence-transformers)][#docs-package]

<!-- [![PyPI - Downloads](https://img.shields.io/pypi/dm/sentence-transformers?logo=pypi&style=flat&color=green)][#pypi-package] -->

<!--- BADGES: END --->

# Sentence Transformers: Embeddings, Retrieval, and Reranking

This framework provides an easy method to compute embeddings for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models ([quickstart](https://sbert.net/docs/quickstart.html#sentence-transformer)), to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models ([quickstart](https://sbert.net/docs/quickstart.html#cross-encoder)) or to generate sparse embeddings using Sparse Encoder models ([quickstart](https://sbert.net/docs/quickstart.html#sparse-encoder)). This unlocks a wide range of applications, including [semantic search](https://sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html), and [paraphrase mining](https://sbert.net/examples/applications/paraphrase-mining/README.html).

A wide selection of over [15,000 pre-trained Sentence Transformers models](https://huggingface.co/models?library=sentence-transformers) are available for immediate use on 🤗 Hugging Face, including many of the state-of-the-art models from the [Massive Text Embeddings Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Additionally, it is easy to train or finetune your own [embedding models](https://sbert.net/docs/sentence_transformer/training_overview.html), [reranker models](https://sbert.net/docs/cross_encoder/training_overview.html) or [sparse encoder models](https://sbert.net/docs/sparse_encoder/training_overview.html) using Sentence Transformers, enabling you to create custom models for your specific use cases.

For the **full documentation**, see **[www.SBERT.net](https://www.sbert.net)**.

## Installation

We recommend **Python 3.10+**, **[PyTorch 1.11.0+](https://pytorch.org/get-started/locally/)**, and **[transformers v4.34.0+](https://github.com/huggingface/transformers)**.

**Install with pip**

```
pip install -U sentence-transformers
```

**Install with conda**

```
conda install -c conda-forge sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/huggingface/sentence-transformers) and install it directly from the source code:

```
pip install -e .
```

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

## Getting Started

See [Quickstart](https://www.sbert.net/docs/quickstart.html) in our documentation.

### Embedding Models

First download a pretrained embedding a.k.a. Sentence Transformer model.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# => (3, 384)
```

And that's already it. We now have numpy arrays with the embeddings, one for each text. We can use these to compute similarities.

```python
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

### Reranker Models

First download a pretrained reranker a.k.a. Cross Encoder model.

```python
from sentence_transformers import CrossEncoder

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

Then provide some texts to the model.

```python
# The texts for which to predict similarity scores
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
]

# 2a. predict scores for pairs of texts
scores = model.predict([(query, passage) for passage in passages])
print(scores)
# => [8.607139 5.506266 6.352977]
```

And we're good to go. You can also use [`model.rank`](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.rank) to avoid having to perform the reranking manually:

```python
# 2b. Rank a list of passages for a query
ranks = model.rank(query, passages, return_documents=True)

print("Query:", query)
for rank in ranks:
    print(f"- #{rank['corpus_id']} ({rank['score']:.2f}): {rank['text']}")
"""
Query: How many people live in Berlin?
- #0 (8.61): Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
- #2 (6.35): In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
- #1 (5.51): Berlin has a yearly total of about 135 million day visitors, making it one of the most-visited cities in the European Union.
"""
```

### Sparse Encoder Models

First download a pretrained sparse embedding a.k.a. Sparse Encoder model.

```python

from sentence_transformers import SparseEncoder

# 1. Load a pretrained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate sparse embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 30522] - sparse representation with vocabulary size dimensions

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[   35.629,     9.154,     0.098],
#         [    9.154,    27.478,     0.019],
#         [    0.098,     0.019,    29.553]])

# 4. Check sparsity stats
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity: {stats['sparsity_ratio']:.2%}")
# Sparsity: 99.84%
```

## Pre-Trained Models

We provide a large list of pretrained models for more than 100 languages. Some models are general purpose models, while others produce embeddings for specific use cases.

- [Pretrained Sentence Transformer (Embedding) Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
- [Pretrained Cross Encoder (Reranker) Models](https://sbert.net/docs/cross_encoder/pretrained_models.html)
- [Pretrained Sparse Encoder (Sparse Embeddings) Models](https://sbert.net/docs/sparse_encoder/pretrained_models.html)

## Training

This framework allows you to fine-tune your own sentence embedding methods, so that you get task-specific sentence embeddings. You have various options to choose from in order to get perfect sentence embeddings for your specific task.

- Embedding Models
  - [Sentence Transformer > Training Overview](https://www.sbert.net/docs/sentence_transformer/training_overview.html)
  - [Sentence Transformer > Training Examples](https://www.sbert.net/docs/sentence_transformer/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/master/examples/sentence_transformer/training).
- Reranker Models
  - [Cross Encoder > Training Overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
  - [Cross Encoder > Training Examples](https://www.sbert.net/docs/cross_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/master/examples/cross_encoder/training).
- Sparse Embedding Models
  - [Sparse Encoder > Training Overview](https://www.sbert.net/docs/sparse_encoder/training_overview.html)
  - [Sparse Encoder > Training Examples](https://www.sbert.net/docs/sparse_encoder/training/examples.html) or [training examples on GitHub](https://github.com/huggingface/sentence-transformers/tree/master/examples/sparse_encoder/training).

Some highlights across the different types of training are:

- Support of various transformer networks including BERT, RoBERTa, XLM-R, DistilBERT, Electra, BART, ...
- Multi-Lingual and multi-task learning
- Evaluation during training to find optimal model
- [20+ loss functions](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html) for embedding models, [10+ loss functions](https://www.sbert.net/docs/package_reference/cross_encoder/losses.html) for reranker models and [10+ loss functions](https://www.sbert.net/docs/package_reference/sparse_encoder/losses.html) for sparse embedding models, allowing you to tune models specifically for semantic search, paraphrase mining, semantic similarity comparison, clustering, triplet loss, contrastive loss, etc.

## Application Examples

You can use this framework for:

- **Computing Sentence Embeddings**

  - [Dense Embeddings](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
  - [Sparse Embeddings](https://www.sbert.net/examples/sparse_encoder/applications/computing_embeddings/README.html)

- **Semantic Textual Similarity**

  - [Dense STS](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)
  - [Sparse STS](https://www.sbert.net/examples/sparse_encoder/applications/semantic_textual_similarity/README.html)

- **Semantic Search**

  - [Dense Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
  - [Sparse Search](https://www.sbert.net/examples/sparse_encoder/applications/semantic_search/README.html)

- **Retrieve & Re-Rank**

  - [Dense only Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
  - [Sparse/Dense/Hybrid Retrieval](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)

- [Clustering](https://www.sbert.net/examples/sentence_transformer/applications/clustering/README.html)

- [Paraphrase Mining](https://www.sbert.net/examples/sentence_transformer/applications/paraphrase-mining/README.html)

- [Translated Sentence Mining](https://www.sbert.net/examples/sentence_transformer/applications/parallel-sentence-mining/README.html)

- [Multilingual Image Search, Clustering & Duplicate Detection](https://www.sbert.net/examples/sentence_transformer/applications/image-search/README.html)

and many more use-cases.

For all examples, see [examples/sentence_transformer/applications](https://github.com/huggingface/sentence-transformers/tree/master/examples/sentence_transformer/applications).

## Development setup

After cloning the repo (or a fork) to your machine, in a virtual environment, run:

```
python -m pip install -e ".[dev]"

pre-commit install
```

To test your changes, run:

```
pytest
```

## Citing & Authors

If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084):

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

If you use one of the multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813):

```bibtex
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
```

Please have a look at [Publications](https://www.sbert.net/docs/publications.html) for our different publications that are integrated into SentenceTransformers.

### Maintainers

Maintainer: [Tom Aarsen](https://github.com/tomaarsen), 🤗 Hugging Face

Don't hesitate to open an issue if something is broken (and it shouldn't be) or if you have further questions.

---

This project was originally developed by the [Ubiquitous Knowledge Processing (UKP) Lab](https://www.ukp.tu-darmstadt.de/) at TU Darmstadt. We're grateful for their foundational work and continued contributions to the field.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

[#docs-package]: https://www.sbert.net/
[#github-license]: https://github.com/huggingface/sentence-transformers/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/sentence-transformers/
