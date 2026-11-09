# Model Merging

Model merging combines the weights of compatible models into a single model without additional training. It can be useful when you have several fine-tunes of a common base model and want to combine their capabilities, or when you want to interpolate between a model and a further fine-tuned checkpoint.

Sentence Transformers uses [mergekit](https://github.com/arcee-ai/mergekit) to merge weights through its in-memory API. The result has the same architecture as the inputs and can be used, trained, and saved like any other Cross Encoder.

Merging can improve some tasks while making others worse. Evaluate the individual inputs and the merged model on the tasks you care about before choosing a merge.

## Installation

Install the `merge` extra, which includes `mergekit>=0.1.5`:

```bash
pip install "sentence-transformers[merge]"
```

## Merge Two Rerankers

This example combines BGE's multilingual reranker with a Korean fine-tune using equal weights:

```python
from sentence_transformers import CrossEncoder

merged = CrossEncoder.merge(
    models=[
        "BAAI/bge-reranker-v2-m3",
        "dragonkue/bge-reranker-v2-m3-ko",
    ],
    method="linear",
    parameters={"weight": [0.5, 0.5]},
)
merged.save_pretrained("merged-reranker-linear")

scores = merged.predict([
    ("What is model merging?", "Combining the weights of compatible models."),
    ("What is model merging?", "Pizza is great with cheese."),
])
print(scores)
```

See [model_merging.py](model_merging.py) for the runnable script.

## Choosing a Merge Method

Select a mergekit method with `method` and pass its options through `parameters`. See [mergekit's method documentation](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md) for descriptions, parameters, and defaults.

Start with `linear` as a baseline. You can also compare `slerp` for two models, or `ties`, `dare_linear`, and `dare_ties` when you have a compatible common base checkpoint. The [shared model merging guide](../../../sentence_transformer/training/merging/README.md#choosing-a-merge-method) shows how to pass parameters and a base model.

```{eval-rst}
See :meth:`CrossEncoder.merge <sentence_transformers.cross_encoder.model.CrossEncoder.merge>` for the supported method names, arguments, and return value.
```

## Choosing Inputs and Evaluating

The models must have compatible architectures, tokenizers, module configurations, and classification heads. For classification tasks, output labels must have the same meaning and order. Fine-tunes of the same reranker are a useful starting point.

The output inherits configuration and preprocessing from the first input. Compare both parents and the merged model on representative reranking data, including the languages you want to support. A merge with a language-specific fine-tune may improve one language while reducing performance on another.

Merging defaults to float32 on CPU and holds the input models and the result in memory. See [Memory and Precision](../../../sentence_transformer/training/merging/README.md#memory-and-precision) for details.

## Results on NanoBEIR

The evaluation below uses a different pair from the Korean fine-tune example: BGE-reranker-large and BGE-reranker-v2-m3, combined with a 50/50 linear merge. The table covers the 11 datasets used by `CrossEncoderNanoBEIREvaluator`. ArguAna and Touche2020 are excluded. Each model reranked the same BM25 top-100 candidates plus all relevant documents, with a 512-token limit per query-document pair. Scores are nDCG@10 multiplied by 100, averaged equally across datasets.

| Model | All 11 datasets | MS MARCO | Other 10 datasets |
| --- | ---: | ---: | ---: |
| [BGE-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 67.05 | **70.83** | 66.67 |
| [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 69.29 | 68.67 | 69.35 |
| [50/50 merge](https://huggingface.co/cross-encoder/merged-bge-reranker-large) | **69.81** | 70.28 | **69.76** |

The merge gained 0.52 points over BGE-reranker-v2-m3 on average, but scored 4.24 points lower on Quora. On the seven datasets evaluated after candidate selection, it beat the stronger parent's average by only 0.08 points.

### Use the Published Merge

Load [cross-encoder/merged-bge-reranker-large](https://huggingface.co/cross-encoder/merged-bge-reranker-large) directly with `CrossEncoder`:

```python
from sentence_transformers import CrossEncoder

merged = CrossEncoder("cross-encoder/merged-bge-reranker-large")
```

The [model card](https://huggingface.co/cross-encoder/merged-bge-reranker-large#evaluation) includes the results for each dataset and a model creation script. This pair requires cropping M3's position embeddings to match BGE-reranker-large before merging. The published merge supports 512-token inputs, compared with M3's native 8192-token limit.

## Evaluate Your Merge

Run the same evaluator on each parent and the merged model, using matching input limits. The default `CrossEncoderNanoBEIREvaluator` uses the 11-dataset protocol described above:

```python
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

evaluator = CrossEncoderNanoBEIREvaluator(rerank_k=100, batch_size=16)
results = evaluator(merged)
print(results[evaluator.primary_metric])
```

## Other Model Types

The same API is available for all four model types:

- [Sentence Transformer merging](../../../sentence_transformer/training/merging/README.md), with dense embedding examples and NanoBEIR results.
- [Sparse Encoder merging](../../../sparse_encoder/training/merging/README.md), with SPLADE examples and NanoBEIR results.
- [Multi-Vector Encoder merging](../../../multi_vector_encoder/training/merging/README.md), including token projection layers.
