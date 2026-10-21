# Model Merging

Model merging combines the weights of compatible models into a single model without additional training. It can be useful when you have several fine-tunes of a common base model and want to combine their capabilities, or when you want to interpolate between a model and a further fine-tuned checkpoint.

Sentence Transformers uses [mergekit](https://github.com/arcee-ai/mergekit) to merge weights through its in-memory API. The result has the same architecture as the inputs and can be used, trained, and saved like any other Multi-Vector Encoder.

Merging can improve some tasks while making others worse. Evaluate the individual inputs and the merged model on the tasks you care about before choosing a merge.

## Installation

Install the `merge` extra, which includes `mergekit>=0.1.5`:

```bash
pip install "sentence-transformers[merge]"
```

## Merge Two LateOn Models

LateOn and LateOn-regularized share compatible architectures, tokenizers, and query/document length settings:

```python
from sentence_transformers import MultiVectorEncoder

merged = MultiVectorEncoder.merge(
    models=["lightonai/LateOn", "lightonai/LateOn-regularized"],
    method="linear",
    parameters={"weight": [0.6, 0.4]},
)
merged.save_pretrained("merged-lateon")

queries = merged.encode_query(["What is model merging?"])
documents = merged.encode_document(["Model merging combines fine-tuned weights."])
print(merged.similarity(queries, documents))
```

See [model_merging.py](model_merging.py) for the runnable script.

## Choosing a Merge Method

Select a mergekit method with `method` and pass its options through `parameters`. See [mergekit's method documentation](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md) for descriptions, parameters, and defaults.

Start with `linear` as a baseline. You can also compare `slerp` for two models, or `ties`, `dare_linear`, and `dare_ties` when you have a compatible common base checkpoint. The [shared model merging guide](../../../sentence_transformer/training/merging/README.md#choosing-a-merge-method) shows how to pass parameters and a base model.

```{eval-rst}
See :meth:`MultiVectorEncoder.merge <sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.merge>` for the supported method names, arguments, and return value.
```

## Choosing Inputs and Evaluating

Matching transformer weights alone is not enough. The projection layers, tokenizers, and module configurations must also be compatible, including query/document lengths and query expansion settings. Models with different token embedding dimensions cannot be merged with this API.

The output inherits configuration, prompts, and processors from the first input. Compare the individual models and the merge on representative retrieval data, using the same query and document processing settings.

Merging defaults to float32 on CPU and holds the inputs and the result in memory. See [Memory and Precision](../../../sentence_transformer/training/merging/README.md#memory-and-precision) for details.
