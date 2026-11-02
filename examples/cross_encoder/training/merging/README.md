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
