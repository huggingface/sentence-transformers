# Bucket-based compilation

The class in [`compiled.py`](./compiled.py) achieves up to a 3x speedup for
short sequences compared to no compilation by eliminating Python overhead. No
external dependencies required besides CUDA.


## Usage

```python
import torch

import compiled

assert torch.cuda.is_available(), "CUDA is required"

# Load the model with optional compilation kwargs:
model = compiled.SentenceTransformer(
    "lightonai/modernbert-embed-large",
    model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    # Compilation kwargs:
    compiled_batch_size=1,  # serve one text at a time
    compiled_token_buckets=(64, 128, 256, 512, 1024),  # tune to your distribution
    compile_fallback=True,  # trade off warm up time for speed after the largest bucket
)

# Warm up the model. Can take minutes for slightly larger models
model.compile_and_warm_up()

# Serve
x = model.encode("Hello, world!")
```


## Caution

`compiled.SentenceTransformer` is not always a drop-in replacement for
`SentenceTransformer`. For your model, you should consider running it on a
representative workload to:

- verify its tokenizer is compatible
- measure numerical drift in embeddings
- benchmark the speedup by tuning `compiled_token_buckets` and `compile_fallback`.

> [!WARNING]
> The CUDA-graph path (`mode="reduce-overhead"`) reuses its output buffers
> across calls, so copy or clone any embedding you need to keep past the next
> `encode()` call, e.g., pass `convert_to_numpy=True`.

> [!NOTE]
> If your model server is managed by k8s, you may need a startup probe
> to wait for the `model.compile_and_warm_up()` call to complete when loading
> the model.


## Benchmark

Takes ~7 min to run on an L4 GPU.

```bash
uv run examples/sentence_transformer/applications/compilation/benchmark.py
```

Per-model, per-bucket latency where:

- `base` refers to the `SentenceTransformer` built-in results
- `st_compiled` refers to the `SentenceTransformer` built-in `model[0].compile(dynamic=True)` results
- `compiled` refers to the compiled version: `compiled.SentenceTransformer` results.

| model_name                              | bucket   | base_ms_p50 | st_compiled_ms_p50 | compiled_ms_p50 | speedup_st_p50 | speedup_p50 |
|-----------------------------------------|----------|-------------|--------------------|-----------------|----------------|-------------|
| BAAI/bge-base-en-v1.5                   | <=64     | 11.27       | 8.45               | 5.13            | 1.33           | 2.2         |
| BAAI/bge-base-en-v1.5                   | 65-128   | 11.61       | 8.75               | 5.43            | 1.33           | 2.14        |
| BAAI/bge-base-en-v1.5                   | 129-256  | 11.81       | 8.81               | 5.93            | 1.34           | 1.99        |
| BAAI/bge-base-en-v1.5                   | 257-512  | 12.6        | 9.48               | 7.09            | 1.33           | 1.78        |
| BAAI/bge-small-en-v1.5                  | <=64     | 11.5        | 8.61               | 4.47            | 1.34           | 2.57        |
| BAAI/bge-small-en-v1.5                  | 65-128   | 11.56       | 8.58               | 4.77            | 1.35           | 2.42        |
| BAAI/bge-small-en-v1.5                  | 129-256  | 11.94       | 9.0                | 5.15            | 1.33           | 2.32        |
| BAAI/bge-small-en-v1.5                  | 257-512  | 12.71       | 9.28               | 6.2             | 1.37           | 2.05        |
| intfloat/e5-small-v2                    | <=64     | 11.28       | 8.2                | 4.51            | 1.38           | 2.5         |
| intfloat/e5-small-v2                    | 65-128   | 11.49       | 8.41               | 4.85            | 1.37           | 2.37        |
| intfloat/e5-small-v2                    | 129-256  | 11.8        | 8.68               | 5.29            | 1.36           | 2.23        |
| intfloat/e5-small-v2                    | 257-512  | 12.17       | 9.09               | 6.2             | 1.34           | 1.96        |
| lightonai/modernbert-embed-large        | <=64     | 25.58       | 13.66              | 9.42            | 1.87           | 2.72        |
| lightonai/modernbert-embed-large        | 65-128   | 27.5        | 13.78              | 9.94            | 2.0            | 2.77        |
| lightonai/modernbert-embed-large        | 129-256  | 28.14       | 14.25              | 10.71           | 1.97           | 2.63        |
| lightonai/modernbert-embed-large        | 257-512  | 28.66       | 14.78              | 15.64           | 1.94           | 1.83        |
| lightonai/modernbert-embed-large        | 513-1024 | 29.87       | 21.37              | 25.44           | 1.4            | 1.17        |
| lightonai/modernbert-embed-large        | >1024    | 215.63      | 199.17             | 193.49          | 1.08           | 1.11        |
| sentence-transformers/all-MiniLM-L6-v2  | <=64     | 6.79        | 5.05               | 3.08            | 1.35           | 2.21        |
| sentence-transformers/all-MiniLM-L6-v2  | 65-128   | 7.07        | 5.4                | 3.28            | 1.31           | 2.16        |
| sentence-transformers/all-MiniLM-L6-v2  | 129-256  | 7.4         | 5.71               | 3.59            | 1.3            | 2.06        |
| sentence-transformers/all-mpnet-base-v2 | <=64     | 12.48       | 8.48               | 4.97            | 1.47           | 2.51        |
| sentence-transformers/all-mpnet-base-v2 | 65-128   | 12.86       | 8.71               | 5.26            | 1.48           | 2.44        |
| sentence-transformers/all-mpnet-base-v2 | 129-256  | 13.64       | 8.85               | 5.91            | 1.54           | 2.31        |
| sentence-transformers/all-mpnet-base-v2 | 257-512  | 15.0        | 9.54               | 6.66            | 1.57           | 2.25        |


`lightonai/modernbert-embed-large` bucket `513-1024` demonstrates that the
padding that `compiled.SentenceTransformer` adds to reach the next bucket can
hurt performance compare to plain compilation.

Warmup time per model where:

- `st_warmup_sec` is the warmup time for the `SentenceTransformer`'s
  `model[0].compile(dynamic=True)` call
- `warmup_sec` is the warmup time for the `compiled.SentenceTransformer`'s
  `model.compile_and_warm_up()` call.

| model_name                              | st_warmup_sec | warmup_sec |
|-----------------------------------------|---------------|------------|
| BAAI/bge-base-en-v1.5                   | 11.9          | 15.4       |
| BAAI/bge-small-en-v1.5                  | 12.0          | 15.3       |
| intfloat/e5-small-v2                    | 11.8          | 15.3       |
| lightonai/modernbert-embed-large        | 31.3          | 98.2       |
| sentence-transformers/all-MiniLM-L6-v2  | 7.4           | 7.1        |
| sentence-transformers/all-mpnet-base-v2 | 15.8          | 14.8       |
