# Pretrained Models

```{eval-rst}
We have released various pre-trained Cross Encoder models via our Cross Encoder Hugging Face organization. Additionally, numerous community Cross Encoder models have been publicly released on the Hugging Face Hub.

* **Original models**: `Cross Encoder Hugging Face organization <https://huggingface.co/models?library=sentence-transformers&author=cross-encoder>`_.
* **Community models**: `All Cross Encoder models on Hugging Face <https://huggingface.co/models?library=sentence-transformers&pipeline_tag=text-ranking>`_.

Each of these models can be easily downloaded and used like so:
```

```python
from sentence_transformers import CrossEncoder
import torch

# Load https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", activation_fn=torch.nn.Sigmoid())
scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])
# => array([0.9998173 , 0.01312432], dtype=float32)
```

Cross-Encoders require text pairs as inputs and output a score 0...1 (if the Sigmoid activation function is used). They do not work for individual sentences and they don't compute embeddings for individual texts.

## MS MARCO

[MS MARCO Passage Retrieval](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset with real user queries from Bing search engine with annotated relevant text passages. Models trained on this dataset are very effective as rerankers for search systems.

```{eval-rst}
.. note::
    You can initialize these models with ``activation_fn=torch.nn.Sigmoid()`` to force the model to return scores between 0 and 1. Otherwise, the raw value can reasonably range between -10 and 10.
```

| Model Name | NDCG@10 (TREC DL 19) | MRR@10 (MS Marco Dev) | Docs / Sec |
| ------------- | :-------------: | :-----: | ---: |
| [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2) | 69.84 | 32.56 | 9000
| [cross-encoder/ms-marco-MiniLM-L2-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2) | 71.01 | 34.85 | 4100
| [cross-encoder/ms-marco-MiniLM-L4-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | 73.04 | 37.70 | 2500
| **[cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)** | 74.30 | 39.01 | 1800
| [cross-encoder/ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 74.31 | 39.02 | 960
| [cross-encoder/ms-marco-electra-base](https://huggingface.co/cross-encoder/ms-marco-electra-base) | 71.99 | 36.41 | 340 |

For details on the usage, see [Retrieve & Re-Rank](../../examples/sentence_transformer/applications/retrieve_rerank/README.md).

## SQuAD (QNLI)

QNLI is based on the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) ([HF](https://huggingface.co/datasets/rajpurkar/squad)) and was introduced by the [GLUE Benchmark](https://huggingface.co/papers/1804.07461) ([HF](https://huggingface.co/datasets/nyu-mll/glue)). Given a passage from Wikipedia, annotators created questions that are answerable by that passage. These models output higher scores if a passage answers a question.

| Model Name | Accuracy on QNLI dev set |
| ------------- | :----------------------------: |
| [cross-encoder/qnli-distilroberta-base](https://huggingface.co/cross-encoder/qnli-distilroberta-base) | 90.96 |
| [cross-encoder/qnli-electra-base](https://huggingface.co/cross-encoder/qnli-electra-base) | 93.21 |

## STSbenchmark

The following models can be used like this:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/stsb-roberta-base")
scores = model.predict([("It's a wonderful day outside.", "It's so sunny today!"), ("It's a wonderful day outside.", "He drove to work earlier.")])
# => array([0.60443085, 0.00240758], dtype=float32)
```

They return a score 0...1 indicating the semantic similarity of the given sentence pair.
| Model Name | STSbenchmark Test Performance |
| ------------- | :----------------------------: |
| [cross-encoder/stsb-TinyBERT-L4](https://huggingface.co/cross-encoder/stsb-TinyBERT-L4) | 85.50 |
| [cross-encoder/stsb-distilroberta-base](https://huggingface.co/cross-encoder/stsb-distilroberta-base) | 87.92 |
| [cross-encoder/stsb-roberta-base](https://huggingface.co/cross-encoder/stsb-roberta-base) | 90.17 |
| [cross-encoder/stsb-roberta-large](https://huggingface.co/cross-encoder/stsb-roberta-large) | 91.47 |

## Quora Duplicate Questions

These models have been trained on the [Quora duplicate questions dataset](https://huggingface.co/datasets/sentence-transformers/quora-duplicates). They can used like the STSb models and give a score 0...1 indicating the probability that two questions are duplicate questions.

| Model Name | Average Precision dev set |
| ------------- | :----------------------------: |
| [cross-encoder/quora-distilroberta-base](https://huggingface.co/cross-encoder/quora-distilroberta-base) | 87.48 |
| [cross-encoder/quora-roberta-base](https://huggingface.co/cross-encoder/quora-roberta-base) | 87.80 |
| [cross-encoder/quora-roberta-large](https://huggingface.co/cross-encoder/quora-roberta-large) | 87.91 |

```{eval-rst}
.. note::
    The model don't work for question similarity. The question "How to learn Java?" and "How to learn Python?" will get a low score, as these questions are not duplicates. For question similarity, a :class:`~sentence_transformers.SentenceTransformer` trained on the Quora dataset will yield much more meaningful results.
```

## NLI

Given two sentences, are these contradicting each other, entailing one the other or are these neutral? The following models were trained on the [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) datasets.
| Model Name | Accuracy on MNLI mismatched set |
| ------------- | :----------------------------: |
| [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | 90.04 |
| [cross-encoder/nli-deberta-base](https://huggingface.co/cross-encoder/nli-deberta-base) | 88.08 |
| [cross-encoder/nli-deberta-v3-xsmall](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall) | 87.77 |
| [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | 87.55 |
| [cross-encoder/nli-roberta-base](https://huggingface.co/cross-encoder/nli-roberta-base) | 87.47 |
| [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) | 86.89 |
| [cross-encoder/nli-distilroberta-base](https://huggingface.co/cross-encoder/nli-distilroberta-base) | 83.98 |

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
scores = model.predict([
    ("A man is eating pizza", "A man eats something"),
    ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
])

# Convert scores to labels
label_mapping = ["contradiction", "entailment", "neutral"]
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
# => ['entailment', 'contradiction']
```

## Community Models

Some notable models from the Community include:

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)
- [BAAI/bge-reranker-v2-minicpm-layerwise](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise)
- [jinaai/jina-reranker-v1-tiny-en](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en)
- [jinaai/jina-reranker-v1-turbo-en](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)
- [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
- [mixedbread-ai/mxbai-rerank-base-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1)
- [mixedbread-ai/mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1)
- [maidalun1020/bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)
- [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
- [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
