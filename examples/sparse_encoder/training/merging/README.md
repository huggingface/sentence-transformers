# Model Merging

Model merging combines the weights of compatible models into a single model without additional training. It can be useful when you have several fine-tunes of a common base model and want to combine their capabilities, or when you want to interpolate between a model and a further fine-tuned checkpoint.

Sentence Transformers uses [mergekit](https://github.com/arcee-ai/mergekit) to merge weights through its in-memory API. The result has the same architecture as the inputs and can be used, trained, and saved like any other Sparse Encoder.

Merging can improve some tasks while making others worse. Evaluate the individual inputs and the merged model on the tasks you care about before choosing a merge.

## Installation

Install the `merge` extra, which includes `mergekit>=0.1.5`:

```bash
pip install "sentence-transformers[merge]"
```

## Merge Two SPLADE Models

The ensemble-distilled and self-distilled SPLADE co-condenser models share a compatible architecture and tokenizer:

```python
from sentence_transformers import SparseEncoder

merged = SparseEncoder.merge(
    models=[
        "naver/splade-cocondenser-ensembledistil",
        "naver/splade-cocondenser-selfdistil",
    ],
    method="linear",
    parameters={"weight": [0.5, 0.5]},
)
merged.save_pretrained("merged-splade-linear")

queries = merged.encode_query(["What is model merging?"])
documents = merged.encode_document(["Model merging combines compatible model weights."])
print(merged.similarity(queries, documents))
```

See [model_merging.py](model_merging.py) for the runnable script.

## Choosing a Merge Method

Select a mergekit method with `method` and pass its options through `parameters`. See [mergekit's method documentation](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md) for descriptions, parameters, and defaults.

Start with `linear` as a baseline. You can also compare `slerp` for two models, or `ties`, `dare_linear`, and `dare_ties` when you have a compatible common base checkpoint. For these [task-vector methods](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md#task-vector-methods), the co-condenser models above can use `Luyu/co-condenser-marco` as their shared pretrained base. The [shared model merging guide](../../../sentence_transformer/training/merging/README.md#choosing-a-merge-method) shows how to pass parameters and a base model.

```{eval-rst}
See :meth:`SparseEncoder.merge <sentence_transformers.sparse_encoder.model.SparseEncoder.merge>` for the supported method names, arguments, and return value.
```

## Compatible Inputs

Vocabulary IDs, model architecture, and pooling settings must be compatible. The result keeps the first input's pooling configuration and tokenizer.

Merging defaults to float32 on CPU and holds the inputs and the result in memory. See [Memory and Precision](../../../sentence_transformer/training/merging/README.md#memory-and-precision) for details.

## Results on NanoBEIR

Building on the co-condenser example above, the evaluation below combines SPLADE-v3 with each of those models in a 50/50 linear merge. The table compares all three parents and both merges on all 13 [NanoBEIR](https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en) datasets. Scores are nDCG@10 multiplied by 100, averaged equally across datasets.

| Model | All 13 datasets | MS MARCO | Other 12 datasets |
| --- | ---: | ---: | ---: |
| [EnsembleDistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | 62.18 | 67.80 | 61.71 |
| [SelfDistil](https://huggingface.co/naver/splade-cocondenser-selfdistil) | 62.70 | 65.92 | 62.43 |
| [SPLADE-v3](https://huggingface.co/naver/splade-v3) | 63.36 | **70.01** | 62.81 |
| SPLADE-v3 + EnsembleDistil | 63.84 | 69.69 | 63.35 |
| [SPLADE-v3 + SelfDistil](https://huggingface.co/sparse-encoder/merged-SPLADE-v3-selfdistil) | **64.01** | 67.46 | **63.72** |

The SPLADE-v3 + SelfDistil merge gained 0.65 points over SPLADE-v3 on average and outperformed both parents on 8 of 13 datasets. Its MS MARCO score fell by 2.55 points. The SPLADE-v3 + EnsembleDistil merge gave a smaller average improvement with a smaller MS MARCO loss.

### Use the Published Merge

The SPLADE-v3 + SelfDistil checkpoint is available as [sparse-encoder/merged-SPLADE-v3-selfdistil](https://huggingface.co/sparse-encoder/merged-SPLADE-v3-selfdistil). Load it directly with `SparseEncoder`:

```python
from sentence_transformers import SparseEncoder

merged = SparseEncoder("sparse-encoder/merged-SPLADE-v3-selfdistil")
```

The [model card](https://huggingface.co/sparse-encoder/merged-SPLADE-v3-selfdistil#evaluation) includes the results for each dataset and a model creation script. To reproduce the 50/50 merge yourself:

```python
merged = SparseEncoder.merge(
    ["naver/splade-v3", "naver/splade-cocondenser-selfdistil"],
    method="linear",
    parameters={"weight": [0.5, 0.5]},
)
merged.save_pretrained("merged-SPLADE-v3-selfdistil")
```

## Evaluate Your Merge

Run the same evaluator on each parent and the merged model, using matching prompts and input limits. The default `SparseNanoBEIREvaluator` covers all 13 NanoBEIR datasets:

```python
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

evaluator = SparseNanoBEIREvaluator(batch_size=8)
results = evaluator(merged)
print(results[evaluator.primary_metric])
```

Also compare sparsity and retrieval cost. A merge can change the number of active vocabulary dimensions, affecting index size and search speed even when retrieval quality improves.

## Other Model Types

The same API is available for all four model types:

- [Sentence Transformer merging](../../../sentence_transformer/training/merging/README.md), with dense embedding examples and NanoBEIR results.
- [Cross Encoder merging](../../../cross_encoder/training/merging/README.md), including classification heads.
- [Multi-Vector Encoder merging](../../../multi_vector_encoder/training/merging/README.md), including token projection layers.
