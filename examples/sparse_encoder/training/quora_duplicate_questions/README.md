# Quora Duplicate Questions

This folder contains a script that demonstrate how to train Sparse Encoders for **Information Retrieval**. As a simple example, we will use the [Quora Duplicate Questions dataset](https://huggingface.co/datasets/sentence-transformers/quora-duplicates). It contains over 500,000 sentences with over 400,000 pairwise annotations whether two questions are a duplicate or not.

Models trained on this dataset can be used for mining duplicate questions, i.e., given a large set of sentences (in this case questions), identify all pairs that are duplicates using sparse vector similarity.

## Training

```{eval-rst}
Choosing the right loss function is crucial for finetuning useful sparse encoder models. For the given task, the :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` loss functions is a good start. 
```

For the complete example, see **[training_splade_quora.py](training_splade_quora.py)** that leverage the loss to train a splade model on this dataset.

```{eval-rst}
:class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` is especially suitable for Information Retrieval / Semantic Search with sparse encoders. A nice advantage is that it only requires positive pairs, i.e., we only need examples of duplicate questions.
```

Using the loss is easy and does not require tuning of any hyperparameters:

```python
from datasets import load_dataset
from sentence_transformers import losses
# Assume 'model' is your SparseEncoder model

full_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train").select(
    range(100000)
)
dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
# => Dataset({
#     features: ['anchor', 'positive', 'negative],
#     num_rows: 99000
# })

loss = losses.SpladeLoss(
    model=model,
    loss=losses.SparseMultipleNegativesRankingLoss(model=model),
    query_regularizer_weight=query_regularizer_weight,  # Weight for query loss
    document_regularizer_weight=document_regularizer_weight,  # Weight for document loss
)
```

```{eval-rst}
.. note::
    Increasing the batch sizes usually yields better results, as the  task gets harder. It is more difficult to identify the correct duplicate question out of a set of 100 questions than out of a set of only 10 questions. So it is advisable to set the training batch size as large as possible. For sparse models, batch size might also be constrained by the memory required for gradient accumulation as the sparse representations are dense during backpropagation.

.. note::
    :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` only works if *(a_i, b_j)* with j != i is actually a negative, non-duplicate question pair. In few instances, this assumption is wrong. But in the majority of cases, if we sample two random questions, they are not duplicates. If your dataset cannot fulfil this property,  :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss` might not work well.
```
