# Evaluation with MTEB

The [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a comprehensive benchmark suite for evaluating embedding models across diverse NLP tasks like retrieval, classification, clustering, reranking, and semantic similarity.

This guide walks you through using MTEB with SentenceTransformer models for post-training evaluation. This is *not* designed for use during training, as this risks overfitting on public benchmarks. For evaluation during training, please see the [Evaluator section in the Training Overview](../training_overview.md#evaluator). To fully integrate your model to MTEB, you can follow the [Adding a model to the Leaderboard](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) guide from the MTEB Documentation.

## Installation

Install MTEB and its dependencies:

```bash
pip install mteb>=2.0.0
```

## Evaluation

You can evaluate your SentenceTransformer model on individual tasks from the MTEB suite like so:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 1: Run a specific single task
tasks = mteb.get_tasks(tasks=["STS22.v2"], languages=["eng"])
results = mteb.evaluate(model, tasks)
```

.. note::

   If you are evaluating existings models the MTEB team recommends that you use `mteb.get_model("{model_name}")` instead of `SentenceTransformer`. This will load the model as it is implemented in MTEB, typically by the model developers. This ensures reproducible results, which might otherwise vary due to normalization, quantization, prompts or similar. If the model isn't implemented in `mteb` it will attempt to load the model using `SentenceTransformer`.

For the full list of available tasks, you can check the MTEB Tasks overview, e.g. for [STS22.v2](https://embeddings-benchmark.github.io/mteb/overview/available_tasks/sts#sts22v2).

You can also filter available MTEB tasks based on task type, domain, language, and more.
For example, the following snippet evaluates on English retrieval tasks in the medical domain:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 2: Run all English retrieval tasks in the medical domain
tasks = mteb.get_tasks(
    task_types=["Retrieval"],
    domains=["Medical"],
    languages=["eng"]
)
results = mteb.evaluate(model, tasks)
```

Lastly, it's often valuable to evaluate on predefined benchmarks. For example, to run all retrieval tasks in the `MTEB(eng, v2)` benchmark:

```python
import mteb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example 3: Run the MTEB benchmark for English tasks
benchmark = mteb.get_benchmark("MTEB(eng, v2)")
results = mteb.evaluate(model, benchmark)
```

For the full list of supported benchmarks, visit the [MTEB Benchmarks documentation](https://embeddings-benchmark.github.io/mteb/overview/available_benchmarks/).

## Additional Arguments

When running evaluations, you can pass arguments down to `model.encode()` using the `encode_kwargs` parameter on [`mteb.evaluate`](https://embeddings-benchmark.github.io/mteb/api/evaluation/#mteb.evaluate). This allows you to customize how embeddings are generated, such as setting `batch_size`, `truncate_dim`, or `normalize_embeddings`. For example:

```python
...

results = mteb.evaluate(
    model,
    tasks,
    encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
)
```

Additionally, your SentenceTransformer model may have been configured to use `prompts`. MTEB will automatically detect and use these prompts if they are defined in your model's configuration. For task-specific or document/query-specific prompts, you should read the MTEB Documentation on [Running SentenceTransformer models with prompts](https://embeddings-benchmark.github.io/mteb/usage/running_the_evaluation#running-sentencetransformer-model-with-prompts).

## Results Handling

MTEB caches all results to disk, so you can rerun `mteb.evaluate` without needing to redownload datasets or recomputing scores. By default these are stored in `~/.cache/mteb`, which is configurable using the environmental variable `MTEB_CACHE`. However you can also manage the cache using the `ResultCache` object:

```python
import mteb.cache import ResultCache
from sentence_transformers import SentenceTransformer

cache = ResultCache("my_mteb_results_folder")

model = SentenceTransformer("all-MiniLM-L6-v2")
tasks = mteb.get_tasks(tasks=["STS17", "STS22.v2"], languages=["eng"])
results = mteb.evaluate(model, tasks, cache=cache)

for task_results in results:
    # Print the aggregated main scores for each task
    print(f"{task_results.task_name}: {task_results.get_score():.4f} mean {task_results.task.metadata.main_score}")
    """
    STS17: 0.2881 mean cosine_spearman
    STS22.v2: 0.4925 mean cosine_spearman
    """

    # Or e.g. print the individual scores for each split or subset
    print(task_results.only_main_score().to_dict())
```

You can even avoid rerunning already existing result by running downloading existing result from the [results repository](https://github.com/embeddings-benchmark/results):

```py
import mteb.cache import ResultCache

cache = ResultCache("my_mteb_results_folder")
cache.download_from_remote() # will take a while the first time

# will only rerun missing results
results = mteb.evaluate(
    tasks, 
    model, 
    cache=cache,
    overwrite_strategy="only-missing" # default
)
```

To read more about how to load and work with results check out the [MTEB documentation](https://embeddings-benchmark.github.io/mteb/usage/loading_results/).

## Leaderboard Submission

To add your model to the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), you will need to follow the [Adding a Model](https://github.com/embeddings-benchmark/mteb/blob/main/docs/adding_a_model.md) MTEB Documentation.

For the process, you'll need to follow these steps:
1. Add your model metadata (name, languages, number of parameters, framework, training datasets, etc.) to the [MTEB Repository](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models).
2. Evaluate your model using MTEB on your desired tasks and save the results.
2. Submit your results to the [MTEB Results Repository](https://github.com/embeddings-benchmark/results).

Once both are merged, after a day you'll be able to find your model on the [official leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
