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

    Generally, :class:`ContrastiveTensionLossInBatchNegatives` is recommended over this loss, as it gives a stronger training signal.

    Args:
        model: SentenceTransformer model

    References:
        * Semantic Re-Tuning with Contrastive Tension: https://openreview.net/pdf?id=Ov_sMNau-PF
        * `Unsupervised Learning > CT <../../../examples/sentence_transformer/unsupervised_learning/CT/README.html>`_

    Inputs:
        +================================+=========+
        | Texts                          | Labels  |
        +================================+=========+
        | (sentence_A, sentence_B) pairs | None    |
        +================================+=========+

    Relations:
        * :class:`ContrastiveTensionLossInBatchNegatives` uses in-batch negative sampling, which gives a stronger training signal than this loss.

    Example:

        Using a dataset with sentence pairs that sometimes are identical (positive pairs) and sometimes different (negative pairs):

        ::

            import random
            from datasets import Dataset
            from sentence_transformers import SentenceTransformer, losses
            from sentence_transformers.training_args import SentenceTransformerTrainingArguments
            from sentence_transformers.trainer import SentenceTransformerTrainer

            model = SentenceTransformer('all-MiniLM-L6-v2')
            # The dataset pairs must sometimes contain identical sentences (positive pairs) and sometimes different sentences (negative pairs).
            train_dataset = Dataset.from_dict({
                "sentence_A": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
                "sentence_B": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
            })
            train_loss = losses.ContrastiveTensionLoss(model=model)

            args = SentenceTransformerTrainingArguments(
                num_train_epochs=10,
                per_device_train_batch_size=32,
                eval_steps=0.1,
                logging_steps=0.01,
                learning_rate=5e-5,
                save_strategy="no",
                fp16=True,
            )
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss)
            trainer.train()

        With a dataset with single sentence pairs:

        ::

            import random
            from datasets import Dataset
            from sentence_transformers import SentenceTransformer, losses
            from sentence_transformers.training_args import SentenceTransformerTrainingArguments
            from sentence_transformers.trainer import SentenceTransformerTrainer

            model = SentenceTransformer('all-MiniLM-L6-v2')
            train_dataset = Dataset.from_dict({
                "text1": [
                    "It's nice weather outside today.",
                    "He drove to work.",
                    "It's so sunny.",
                ]
            })
            sentences = train_dataset['text1']

            def to_ct_pairs(sample, pos_neg_ratio=8):
                pos_neg_ratio = 1 / pos_neg_ratio
                sample["text2"] = sample["text1"] if random.random() < pos_neg_ratio else random.choice(sentences)
                return sample

            pos_neg_ratio = 8  # 1 positive pair for 7 negative pairs
            train_dataset = train_dataset.map(to_ct_pairs, fn_kwargs={"pos_neg_ratio": pos_neg_ratio})

            train_loss = losses.ContrastiveTensionLoss(model=model)

            args = SentenceTransformerTrainingArguments(
                num_train_epochs=10,
                per_device_train_batch_size=32,
                eval_steps=0.1,
                logging_steps=0.01,
                learning_rate=5e-5,
                save_strategy="no",
                fp16=True,
            )
            trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss)
            trainer.train()
    """

    def __init__(self, model: SentenceTransformer) -> None:
        super().__init__()
        self.model2 = model  # This will be the final model used during the inference time.
        self.model1 = copy.deepcopy(model)
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        sentence_features1, sentence_features2 = tuple(sentence_features)
        reps_1 = self.model1(sentence_features1)["sentence_embedding"]  # (bsz, hdim)
        reps_2 = self.model2(sentence_features2)["sentence_embedding"]

        sim_scores = (
            torch.matmul(reps_1[:, None], reps_2[:, :, None]).squeeze(-1).squeeze(-1)
        )  # (bsz,) dot product, i.e. S1S2^T

        if labels is None:
            # identical sentence -> 1, different -> 0
            input_ids1 = sentence_features1["input_ids"]
            input_ids2 = sentence_features2["input_ids"]
            attention_mask1 = sentence_features1.get("attention_mask")
            attention_mask2 = sentence_features2.get("attention_mask")

            if attention_mask1 is not None and attention_mask2 is not None:
                # Compare only non-padded tokens so sentences are considered identical even if they differ by left/right padding.
                labels = torch.as_tensor(
                    [
                        torch.equal(inputs1[mask1], inputs2[mask2])
                        for inputs1, mask1, inputs2, mask2 in zip(
                            input_ids1, attention_mask1.bool(), input_ids2, attention_mask2.bool()
                        )
                    ],
                    device=sim_scores.device,
                    dtype=sim_scores.dtype,
                )
            else:
                labels = (input_ids1 == input_ids2).all(dim=1).float()

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
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
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
                from datasets import Dataset

                model = SentenceTransformer('all-MiniLM-L6-v2')

                train_dataset = Dataset.from_dict({
                    "sentence": [
                       "It's nice weather outside today.",
                       "He drove to work.",
                       "It's so sunny.",
                    ]
                })

                train_loss = losses.ContrastiveTensionLossInBatchNegatives(model=model)

                args = SentenceTransformerTrainingArguments(
                    num_train_epochs=10,
                    per_device_train_batch_size=32,
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
