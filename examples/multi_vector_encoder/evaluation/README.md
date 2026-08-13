# Multi-Vector Encoder Evaluation

This directory contains examples demonstrating how to evaluate Multi-Vector Encoder (ColBERT-style, late-interaction) models.

Multi-vector models encode each input into a sequence of token vectors instead of a single vector, and compare a query with a document using MaxSim: for each query token, take the maximum similarity against any document token, then sum across the query tokens. The evaluators handle that end to end. They encode with `encode_query` and `encode_document` (so the `[Q]` and `[D]` prefixes, query expansion, and any document skiplist apply), score with the model's `similarity_fn_name` (`maxsim` or `meanmaxsim`), and compute the usual metrics on top of those scores. Note that `truncate_dim` is not supported by these evaluators, because token embeddings have no Matryoshka-style truncation.

To run any of these evaluation scripts, simply execute the Python script. Each script will:

1. Load a pretrained multi-vector model.
1. Prepare the evaluation dataset.
1. Configure the appropriate evaluator.
1. Run the evaluation.
1. Report the results.

```{eval-rst}
============================================================================================  ======================================================================================================================================
Evaluator                                                                                     Evaluation Script
============================================================================================  ======================================================================================================================================
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorNanoBEIREvaluator`  `nano_beir.py <https://github.com/huggingface/sentence-transformers/blob/main/examples/multi_vector_encoder/evaluation/nano_beir.py>`_
============================================================================================  ======================================================================================================================================
```

## Example with NanoBEIR Evaluation:

This script evaluates a pretrained multi-vector model on NanoBEIR ([`nano_beir.py`](nano_beir.py)). NanoBEIR is a downsized version of [BEIR](https://github.com/beir-cellar/beir) with roughly 50 queries and 5,000 documents per subset, which makes it a quick way to compare retrieval quality before committing to a full-scale BEIR run. No data preparation is required: the evaluator loads the Nano-\* subsets itself.

```python
"""Evaluate a pretrained multi-vector model on NanoBEIR.

NanoBEIR is a fast benchmarking suite of 13 small BEIR subsets, useful for quickly comparing models
without running the full BEIR evaluation. This script loads a model from the Hub and runs all 13
Nano-* IR datasets with MaxSim scoring.
"""

from __future__ import annotations

from pprint import pprint

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator


def main() -> None:
    model = MultiVectorEncoder("lightonai/LateOn")
    evaluator = MultiVectorNanoBEIREvaluator(batch_size=16)
    results = evaluator(model)
    print(f"Primary metric: {evaluator.primary_metric} = {results[evaluator.primary_metric]:.4f}")
    pprint({k: v for k, v in results.items() if "ndcg@10" in k})


if __name__ == "__main__":
    main()
```

The evaluator reports MRR@k, NDCG@k, Recall@k, Precision@k, Accuracy@k, and MAP@k for every subset, and aggregates those metrics across subsets at the end. A few options worth knowing about:

- `dataset_names` restricts the run to a handful of the 13 subsets, for example `["msmarco", "nq", "fiqa2018"]`. That is the common choice for evaluating during training, with the full suite reserved for the end of the run.
- `dataset_id` points the evaluator at another dataset with the same layout, such as a translated variant from the [NanoBEIR collection](https://huggingface.co/collections/sentence-transformers/nanobeir-datasets), for non-English evaluation.
- `batch_size` sets how many texts are encoded at a time, and `corpus_chunk_size` how many documents are encoded and scored per round-trip.
- `chunk_elements` caps the number of elements in the MaxSim scoring intermediate. Lower it to reduce the memory used while scoring.

## Other Evaluators

NanoBEIR is the only task with an example script in this directory, but the package ships MaxSim-scored evaluators for other tasks as well. Each of their class docstrings contains a runnable example:

```{eval-rst}
========================================================================================================  ==============================================================================================
Evaluator                                                                                                 Required Data
========================================================================================================  ==============================================================================================
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorInformationRetrievalEvaluator`  Queries (qid => question), corpus (cid => document), and relevant documents (qid => set[cid]).
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorRerankingEvaluator`             List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries.
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorTripletEvaluator`               (anchor, positive, negative) triplets.
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorDistillationEvaluator`          Queries with candidate documents and teacher scores.
========================================================================================================  ==============================================================================================
```

`MultiVectorInformationRetrievalEvaluator` is the evaluator that `MultiVectorNanoBEIREvaluator` runs per subset, so it accepts the same metric and memory options for your own corpus. `MultiVectorRerankingEvaluator` reports MAP, MRR@k, and NDCG@k over a fixed candidate list per query, which is the setup for using a multi-vector model as a second-stage reranker. `MultiVectorTripletEvaluator` checks how often the anchor scores its positive above its negative. `MultiVectorDistillationEvaluator` compares student scores against teacher scores with a KL divergence and a Spearman rank correlation, for tracking knowledge distillation runs.
