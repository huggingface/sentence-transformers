# Semantic Textual Similarity

Semantic Textual Similarity (STS) assigns a score on the similarity of two texts. In this example, we use the [stsb](https://huggingface.co/datasets/sentence-transformers/stsb) dataset as training data to fine-tune our sparse encoder model. See the following example scripts how to tune a SparseEncoder, and espacially fine-tune a splade model on STS data:

- **[train_splade_stsbenchmark.py](train_splade_stsbenchmark.py)** - This example shows how to fine-tune a splade model already pre-trained by using a pre-trained splade model (e.g. [`splade-cocondenser-ensembledistil`](https://huggingface.co/naver/splade-cocondenser-ensembledistil)) and fine tuning it to get better result on this specific task.

## Training data

```{eval-rst}
In STS, we have sentence pairs annotated together with a score indicating the similarity. In the original STSbenchmark dataset, the scores range from 0 to 5. They have been normalized to range between 0 and 1 in `stsb <https://huggingface.co/datasets/sentence-transformers/stsb>`_, as that is required for :class:`~sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss` as you can see in the `Loss Overiew <../../../../docs/sparse_encoder/loss_overview.html>`_.
```

Here is a simplified version of our training data:

```python
from datasets import Dataset

sentence1_list = ["My first sentence", "Another pair"]
sentence2_list = ["My second sentence", "Unrelated sentence"]
labels_list = [0.8, 0.3]
train_dataset = Dataset.from_dict({
    "sentence1": sentence1_list,
    "sentence2": sentence2_list,
    "label": labels_list,
})
# => Dataset({
#     features: ['sentence1', 'sentence2', 'label'],
#     num_rows: 2
# })
print(train_dataset[0])
# => {'sentence1': 'My first sentence', 'sentence2': 'My second sentence', 'label': 0.8}
print(train_dataset[1])
# => {'sentence1': 'Another pair', 'sentence2': 'Unrelated sentence', 'label': 0.3}
```

In the aforementioned script, we directly load the [stsb](https://huggingface.co/datasets/sentence-transformers/stsb) dataset:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/stsb", split="train")
# => Dataset({
#     features: ['sentence1', 'sentence2', 'score'],
#     num_rows: 5749
# })
```

## Loss Function

```{eval-rst}
We use :class:`~sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss` as our loss function.
```

For each sentence pair, we pass sentence A and sentence B through the sparse encoder model, which yields the sparse embeddings *u* und *v*. The similarity of these embeddings is computed using cosine similarity and the result is compared to the gold similarity score. Note that the two sentences are fed through the same model rather than two separate models. In particular, the cosine similarity for similar texts is maximized and the cosine similarity for dissimilar texts is minimized. This allows our model to be fine-tuned and to recognize the similarity of sentences.

For more details, see [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://huggingface.co/papers/1908.10084).
