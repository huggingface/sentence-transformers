# Quora Duplicate Questions

This folder contains scripts that demonstrate how to train SentenceTransformers for **Information Retrieval**. As a simple example, we will use the [Quora Duplicate Questions dataset](https://huggingface.co/datasets/sentence-transformers/quora-duplicates). It contains over 500,000 sentences with over 400,000 pairwise annotations whether two questions are a duplicate or not.

Models trained on this dataset can be used for mining duplicate questions, i.e., given a large set of sentences (in this case questions), identify all pairs that are duplicates. See [Paraphrase Mining](../../applications/paraphrase-mining/README.md) for an example how to use sentence transformers to mine for duplicate questions / paraphrases. This approach can be scaled to hundred thousands of sentences.

```{eval-rst}
You can also train and use :class:`~sentence_transformers.cross_encoder.CrossEncoder` models for this task. See `Cross Encoder > Training Examples > Quora Duplicate Questions <../../../cross_encoder/training/quora_duplicate_questions/README.html>`_ for more details.
```

## Training

```{eval-rst}
Choosing the right loss function is crucial for finetuning useful models. For the given task, two loss functions are especially suitable: :class:`~sentence_transformers.losses.OnlineContrastiveLoss` and :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`.
```

### Contrastive Loss

For the complete training example, see [training_OnlineContrastiveLoss.py](training_OnlineContrastiveLoss.py).

```{eval-rst}
The Quora Duplicates dataset has a `pair-class subset <https://huggingface.co/datasets/sentence-transformers/quora-duplicates/viewer/pair-class>`_ which consists of question pairs and labels: 1 for duplicate and 0 for different.

As shown by our `Loss Overview <../../../../docs/sentence_transformer/loss_overview.html>`_, this allows us to use :class:`~sentence_transformers.losses.ContrastiveLoss`. Similar pairs with label 1 are pulled together, so that they are close in vector space, while dissimilar pairs that are closer than a defined margin are pushed away in vector space.

An improved version is :class:`~sentence_transformers.losses.OnlineContrastiveLoss`. This loss looks which negative pairs have a lower distance than the largest positive pair and which positive pairs have a higher distance than the lowest distance of negative pairs. I.e., this loss automatically detects the hard cases in a batch and computes the loss only for these cases.
```

The loss can be used like this:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
# => Dataset({
#     features: ['sentence1', 'sentence2', 'label'],
#     num_rows: 404290
# })
print(train_dataset[0])
# => {'sentence1': 'What is the step by step guide to invest in share market in india?', 'sentence2': 'What is the step by step guide to invest in share market?', 'label': 0}
train_loss = losses.OnlineContrastiveLoss(model=model, margin=0.5)
```

## MultipleNegativesRankingLoss

For the complete example, see [training_MultipleNegativesRankingLoss.py](training_MultipleNegativesRankingLoss.py).

```{eval-rst}
:class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` is especially suitable for Information Retrieval / Semantic Search. A nice advantage is that it only requires positive pairs, i.e., we only need examples of duplicate questions. See `NLI > MultipleNegativesRankingLoss <../nli/README.html#multiplenegativesrankingloss>`_ for more information on how the loss works.
```

Using the loss is easy and does not require tuning of any hyperparameters:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train")
# => Dataset({
#     features: ['anchor', 'positive'],
#     num_rows: 149263
# })
print(train_dataset[0])
# => {'anchor': 'Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?', 'positive': "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?"}
train_loss = losses.MultipleNegativesRankingLoss(model)
```

As 'is_duplicate' is a symmetric relation, we can use not just (anchor, positive) but also (positive, anchor) to our training sample set:

```python
from datasets import concatenate_datasets

train_dataset = concatenate_datasets([
    train_dataset,
    train_dataset.rename_columns({"anchor": "positive", "positive": "anchor"})
])
# Dataset({
#     features: ['anchor', 'positive'],
#     num_rows: 298526
# })
```

```{eval-rst}
.. note::
    Increasing the batch sizes usually yields better results, as the  task gets harder. It is more difficult to identify the correct duplicate question out of a set of 100 questions than out of a set of only 10 questions. So it is advisable to set the training batch size as large as possible. I trained it with a batch size of 350 on 32 GB GPU memory.

.. note::
    :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` only works if *(a_i, b_j)* with j != i is actually a negative, non-duplicate question pair. In few instances, this assumption is wrong. But in the majority of cases, if we sample two random questions, they are not duplicates. If your dataset cannot fulfil this property,  :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` might not work well.
```

### Multi-Task-Learning

```{eval-rst}
:class:`~sentence_transformers.losses.ContrastiveLoss` works well for pair classification, i.e., given two pairs, are these duplicates or not. It pushes negative pairs far away in vector space, so that the distinguishing between duplicate and non-duplicate pairs works good.

:class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` on the other sides mainly reduces the distance between positive pairs out of large set of possible candidates. However, the distance between  non-duplicate questions is not so large, so that this loss does not work that well for pair classification.
```

In [training_multi-task-learning.py](training_multi-task-learning.py) I demonstrate how we can train the network with both losses. The essential code is to define both losses and to pass it to the fit method.

```python
from datasets import load_dataset
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformer

model_name = "stsb-distilbert-base"
model = SentenceTransformer(model_name)

# https://huggingface.co/datasets/sentence-transformers/quora-duplicates
mnrl_dataset = load_dataset(
    "sentence-transformers/quora-duplicates", "triplet", split="train"
)  # The "pair" subset also works
mnrl_train_dataset = mnrl_dataset.select(range(100000))
mnrl_eval_dataset = mnrl_dataset.select(range(100000, 101000))

mnrl_train_loss = MultipleNegativesRankingLoss(model=model)

# https://huggingface.co/datasets/sentence-transformers/quora-duplicates
cl_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
cl_train_dataset = cl_dataset.select(range(100000))
cl_eval_dataset = cl_dataset.select(range(100000, 101000))

cl_train_loss = ContrastiveLoss(model=model, margin=0.5)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset={
        "mnrl": mnrl_train_dataset,
        "cl": cl_train_dataset,
    },
    eval_dataset={
        "mnrl": mnrl_eval_dataset,
        "cl": cl_eval_dataset,
    },
    loss={
        "mnrl": mnrl_train_loss,
        "cl": cl_train_loss,
    },
)
trainer.train()
```

## Pretrained Models

Currently the following models trained on Quora Duplicate Questions are available:

- [distilbert-base-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-quora-ranking): We extended the [distilbert-base-nli-stsb-mean-tokens](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens) model and trained it with *OnlineContrastiveLoss* and with *MultipleNegativesRankingLoss* on the Quora Duplicate questions dataset. For the code, see [training_multi-task-learning.py](training_multi-task-learning.py)
- [distilbert-multilingual-nli-stsb-quora-ranking](https://huggingface.co/sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking): Extension of *distilbert-base-nli-stsb-quora-ranking* to be multi-lingual. Trained on parallel data for 50 languages.

You can load & use pre-trained models like this:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("distilbert-base-nli-stsb-quora-ranking")
```
