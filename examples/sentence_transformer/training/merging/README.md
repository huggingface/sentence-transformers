# Model Merging

Model merging combines the weights of compatible models into a single model without additional training. It can be useful when you have several fine-tunes of a common base model and want to combine their capabilities, or when you want to interpolate between a model and a further fine-tuned checkpoint.

Sentence Transformers uses [mergekit](https://github.com/arcee-ai/mergekit) to merge weights through its in-memory API. The result has the same architecture as the inputs and can be used, trained, and saved like any other Sentence Transformer.

Merging can improve some tasks while making others worse. Evaluate the individual inputs and the merged model on the tasks you care about before choosing a merge.

## Installation

Install the `merge` extra, which includes `mergekit>=0.1.5`:

```bash
pip install "sentence-transformers[merge]"
```

## Merge Two Models

The following example takes a weighted average of two compatible MiniLM models. The weights give the first model a contribution of 60% and the second 40%.

```python
from sentence_transformers import SentenceTransformer

inputs = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
]
merged = SentenceTransformer.merge(
    models=inputs,
    method="linear",
    parameters={"weight": [0.6, 0.4]},
)
merged.save_pretrained("merged-MiniLM-L6")

embeddings = merged.encode(["What is model merging?", "Combining compatible model weights."])
print(merged.similarity(embeddings, embeddings))
```

`models` accepts Hugging Face model IDs, local paths, and already loaded model instances. The returned model is in evaluation mode, and the input models are left unchanged. Configuration, prompts, and processors are inherited from the first input.

See [model_merging.py](model_merging.py) for a runnable example with Linear, SLERP, and TIES.

## Choosing a Merge Method

Sentence Transformers wraps mergekit's merge methods. See [mergekit's method documentation](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md) for descriptions of the methods, their parameters, and defaults. Select a method with `method` and pass its options through `parameters`.

A few starting points to compare on your evaluation data:

- `linear`: A weighted average, as shown above. A useful first baseline.
- `slerp`: An alternative for interpolating between two models.
- `ties`, `dare_linear`, and `dare_ties`: Options to try when you have a compatible common base checkpoint.

For example, to use SLERP:

```python
slerp_merged = SentenceTransformer.merge(
    inputs,
    method="slerp",
    parameters={"t": 0.5},
)
```

For methods requiring a base, pass it as `base_model`, ideally a common checkpoint from which the inputs were fine-tuned. One of the inputs can also serve as the base model, like so:

```python
ties_merged = SentenceTransformer.merge(
    inputs,
    method="ties",
    base_model=inputs[0],
    parameters={"density": 0.7},
)
```

For [task-vector methods](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md#task-vector-methods), `base_model` serves as the reference and is excluded from the inputs whose changes are combined. Per-input parameter lists therefore cover only the non-base inputs in their original order.

```{eval-rst}
The wrapper supplies ``weight=1.0`` for methods accepting it and ``t=0.5`` for SLERP. Other defaults come from mergekit. See :meth:`SentenceTransformer.merge <sentence_transformers.sentence_transformer.model.SentenceTransformer.merge>` for the supported method names, arguments, and return value. This method is available on all four model types.
```

## Compatible Inputs

Use models with matching architectures, module configurations, tokenizer vocabularies, tensor shapes, and tied weights. Matching embedding dimensions alone is not enough. For example, models using different pooling strategies cannot be merged with this API.

Validation checks these structural requirements, but does not verify common training ancestry or tokenizer preprocessing rules. Inputs should be fine-tunes of a common ancestor, and their tokenization and prompt conventions should be suitable for your task. Quantized models are not supported.

Models with attached PEFT adapters are also rejected. Merge the adapters into the base weights and unload them before calling `merge`.

## Memory and Precision

Merging runs on CPU in float32 by default. All input models, an optional separate base, and the merged result are held in memory, along with temporary working tensors. Account for this when merging large models.

Use `device="cuda"` to merge and return the model on a GPU, or move the result afterward with `merged.to("cuda")`. To calculate the merge in float32 and store the result in a smaller dtype, set `out_dtype`:

```python
merged = SentenceTransformer.merge(
    inputs,
    parameters={"weight": [0.6, 0.4]},
    dtype="float32",
    out_dtype="bfloat16",
)
```

`dtype` controls the floating-point inputs passed to the merge method. `out_dtype` casts the merged tensors afterward. Method-specific internal calculations may use a different precision.

## Other Model Types

The same API is available for all four model types:

- [Cross Encoder merging](../../../cross_encoder/training/merging/README.md), including classification heads.
- [Sparse Encoder merging](../../../sparse_encoder/training/merging/README.md), with SPLADE examples and NanoBEIR results.
- [Multi-Vector Encoder merging](../../../multi_vector_encoder/training/merging/README.md), including token projection layers.
