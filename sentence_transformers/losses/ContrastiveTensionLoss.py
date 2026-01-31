from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from sentence_transformers.readers import InputExample
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import cos_sim


class ContrastiveTensionLoss(nn.Module):
    """
    This loss expects only single sentences, without any labels. Positive and negative pairs are automatically created via random sampling,
    such that a positive pair consists of two identical sentences and a negative pair consists of two different sentences. An independent
    copy of the encoder model is created, which is used for encoding the first sentence of each pair. The original encoder model encodes the
    second sentence. The embeddings are compared and scored using the generated labels (1 if positive, 0 if negative) using the binary cross
    entropy objective.

    Note that you must use the `ContrastiveTensionDataLoader` for this loss. The `pos_neg_ratio` of the ContrastiveTensionDataLoader can be
    used to determine the number of negative pairs per positive pair.

    Generally, :class:`ContrastiveTensionLossInBatchNegatives` is recommended over this loss, as it gives a stronger training signal.

    Args:
        model: SentenceTransformer model

    References:
        * Semantic Re-Tuning with Contrastive Tension: https://openreview.net/pdf?id=Ov_sMNau-PF
        * `Unsupervised Learning > CT <../../../examples/sentence_transformer/unsupervised_learning/CT/README.html>`_

    Inputs:
        +------------------+--------+
        | Texts            | Labels |
        +==================+========+
        | single sentences | none   |
        +------------------+--------+

    Relations:
        * :class:`ContrastiveTensionLossInBatchNegatives` uses in-batch negative sampling, which gives a stronger training signal than this loss.

    Example:
        ::

            from sentence_transformers import SentenceTransformer, losses
            from sentence_transformers.losses import ContrastiveTensionDataLoader

            model = SentenceTransformer('all-MiniLM-L6-v2')
            train_examples = [
                'This is the 1st sentence',
                'This is the 2nd sentence',
                'This is the 3rd sentence',
                'This is the 4th sentence',
                'This is the 5th sentence',
                'This is the 6th sentence',
                'This is the 7th sentence',
                'This is the 8th sentence',
                'This is the 9th sentence',
                'This is the final sentence',
            ]

            train_dataloader = ContrastiveTensionDataLoader(train_examples, batch_size=3, pos_neg_ratio=3)
            train_loss = losses.ContrastiveTensionLoss(model=model)

            model.fit(
                [(train_dataloader, train_loss)],
                epochs=10,
            )
    """

    def __init__(self, model: SentenceTransformer) -> None:
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)["sentence_embedding"]  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)["sentence_embedding"]

        sim_scores = (
            torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        )  # (bsz,) dot product, i.e. S1S2^T

        loss = self.criterion(sim_scores, labels.type_as(sim_scores))
        return loss

    @property
    def citation(self) -> str:
        return """
@inproceedings{carlsson2021semantic,
    title={Semantic Re-tuning with Contrastive Tension},
    author={Fredrik Carlsson and Amaru Cuba Gyllensten and Evangelia Gogoulou and Erik Ylip{\"a}{\"a} Hellqvist and Magnus Sahlgren},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Ov_sMNau-PF}
}
"""


class ContrastiveTensionLossInBatchNegatives(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct=cos_sim) -> None:
        """
        This loss expects only single sentences, without any labels. Positive and negative pairs are automatically created via random sampling,
        such that a positive pair consists of two identical sentences and a negative pair consists of two different sentences. An independent
        copy of the encoder model is created, which is used for encoding the first sentence of each pair. The original encoder model encodes the
        second sentence. Unlike :class:`ContrastiveTensionLoss`, this loss uses the batch negative sampling strategy, i.e. the negative pairs
        are sampled from the batch. Using in-batch negative sampling gives a stronger training signal than the original :class:`ContrastiveTensionLoss`.
        The performance usually increases with increasing batch sizes.

        The training :class:`datasets.Dataset` must contain one text column.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Semantic Re-Tuning with Contrastive Tension: https://openreview.net/pdf?id=Ov_sMNau-PF
            - `Unsupervised Learning > CT (In-Batch Negatives) <../../../examples/sentence_transformer/unsupervised_learning/CT_In-Batch_Negatives/README.html>`_

        Relations:
            * :class:`ContrastiveTensionLoss` does not select negative pairs in-batch, resulting in a weaker training signal than this loss.

        Inputs:
            +------------------+--------+
            | Texts            | Labels |
            +==================+========+
            | single sentences | none   |
            +------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, losses
                from torch.utils.data import DataLoader

                model = SentenceTransformer('all-MiniLM-L6-v2')
                train_examples = [
                    InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
                    InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0),
                ]
                train_examples = [
                    InputExample(texts=['This is the 1st sentence']),
                    InputExample(texts=['This is the 2nd sentence']),
                    InputExample(texts=['This is the 3rd sentence']),
                    InputExample(texts=['This is the 4th sentence']),
                    InputExample(texts=['This is the 5th sentence']),
                ]

                train_dataset = Dataset.from_list(train_sentences)
                train_loss = losses.ContrastiveTensionLossInBatchNegatives(model=model)

                args = SentenceTransformerTrainingArguments(
                    num_train_epochs=10,
                    per_device_train_batch_size=32,
                    warmup_steps=0.1,
                    eval_steps=0.1,
                    logging_steps=0.01,
                    learning_rate=5e-5,
                    save_strategy="no",
                    fp16=True,
                )

                trainer = SentenceTransformerTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    loss=train_loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = sentence_features[0]
        embeddings_a = self.model1(sentence_features)["sentence_embedding"]  # (bsz, hdim)
        embeddings_b = self.model2(sentence_features)["sentence_embedding"]

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.logit_scale.exp()  # self.scale
        labels = torch.arange(len(scores), dtype=torch.long, device=scores.device)
        return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels)) / 2

    @property
    def citation(self) -> str:
        return """
@inproceedings{carlsson2021semantic,
    title={Semantic Re-tuning with Contrastive Tension},
    author={Fredrik Carlsson and Amaru Cuba Gyllensten and Evangelia Gogoulou and Erik Ylip{\"a}{\"a} Hellqvist and Magnus Sahlgren},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=Ov_sMNau-PF}
}
"""


# CT Data Loader
# For CT, we need batches in a specific format
# In each batch, we have one positive pair (i.e. [sentA, sentA]) and 7 negative pairs (i.e. [sentA, sentB]).
# To achieve this, we create a custom DataLoader that produces batches with this property


class ContrastiveTensionDataLoader:
    def __init__(self, sentences, batch_size, pos_neg_ratio=8):
        self.sentences = sentences
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        self.collate_fn = None

        if self.batch_size % self.pos_neg_ratio != 0:
            raise ValueError(
                f"ContrastiveTensionDataLoader was loaded with a pos_neg_ratio of {pos_neg_ratio} and a batch size of {batch_size}. The batch size must be divisible by the pos_neg_ratio"
            )

    def __iter__(self):
        random.shuffle(self.sentences)
        sentence_idx = 0
        batch = []

        while sentence_idx + 1 < len(self.sentences):
            s1 = self.sentences[sentence_idx]
            if len(batch) % self.pos_neg_ratio > 0:  # Negative (different) pair
                sentence_idx += 1
                s2 = self.sentences[sentence_idx]
                label = 0
            else:  # Positive (identical pair)
                s2 = self.sentences[sentence_idx]
                label = 1

            sentence_idx += 1
            batch.append(InputExample(texts=[s1, s2], label=label))

            if len(batch) >= self.batch_size:
                yield self.collate_fn(batch) if self.collate_fn is not None else batch
                batch = []

    def __len__(self):
        return math.floor(len(self.sentences) / (2 * self.batch_size))
