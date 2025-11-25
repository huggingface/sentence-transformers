from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch
import transformers
from packaging import version
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class SoftmaxLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        sentence_embedding_dimension: int,
        num_labels: int,
        concatenation_sent_rep: bool = True,
        concatenation_sent_difference: bool = True,
        concatenation_sent_multiplication: bool = False,
        loss_fct: Callable = nn.CrossEntropyLoss(),
    ) -> None:
        """
        This loss was used in our SBERT publication (https://huggingface.co/papers/1908.10084) to train the SentenceTransformer
        model on NLI data. It adds a softmax classifier on top of the output of two transformer networks.

        :class:`MultipleNegativesRankingLoss` is an alternative loss function that often yields better results,
        as per https://huggingface.co/papers/2004.09813.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            sentence_embedding_dimension (int): The dimension of the sentence embeddings.
            num_labels (int): The number of different labels.
            concatenation_sent_rep (bool): Whether to concatenate vectors u,v for the softmax classifier. Defaults to True.
            concatenation_sent_difference (bool): Whether to add abs(u-v) for the softmax classifier. Defaults to True.
            concatenation_sent_multiplication (bool): Whether to add u*v for the softmax classifier. Defaults to False.
            loss_fct (Callable): Custom pytorch loss function. If not set, uses nn.CrossEntropyLoss(). Defaults to nn.CrossEntropyLoss().

        References:
            - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks: https://huggingface.co/papers/1908.10084
            - `Training Examples > Natural Language Inference <../../../examples/sentence_transformer/training/nli/README.html>`_

        Requirements:
            1. sentence pairs with a class label

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (sentence_A, sentence_B) pairs        | class  |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": [
                        "A person on a horse jumps over a broken down airplane.",
                        "A person on a horse jumps over a broken down airplane.",
                        "A person on a horse jumps over a broken down airplane.",
                        "Children smiling and waving at camera",
                    ],
                    "sentence2": [
                        "A person is training his horse for a competition.",
                        "A person is at a diner, ordering an omelette.",
                        "A person is outdoors, on a horse.",
                        "There are children present.",
                    ],
                    "label": [1, 2, 0, 0],
                })
                loss = losses.SoftmaxLoss(model, model.get_sentence_embedding_dimension(), num_labels=3)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logger.info(f"Softmax loss: #Vectors concatenated: {num_vectors_concatenated}")
        self.classifier = nn.Linear(
            num_vectors_concatenated * sentence_embedding_dimension, num_labels, device=model.device
        )
        self.loss_fct = loss_fct

        if version.parse(transformers.__version__) < version.parse("4.43.0"):
            logger.warning(
                "SoftmaxLoss requires transformers >= 4.43.0 to work correctly. "
                "Otherwise, the classifier layer that maps embeddings to the labels cannot be updated. "
                "Consider updating transformers with `pip install transformers>=4.43.0`."
            )

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor | tuple[Tensor, Tensor]:
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        if labels is not None:
            loss = self.loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

    @property
    def citation(self) -> str:
        return """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
"""
