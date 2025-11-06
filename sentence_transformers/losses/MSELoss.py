from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer


class MSELoss(nn.Module):
    def __init__(self, model: SentenceTransformer) -> None:
        """
        Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
        is used when extending sentence embeddings to new languages as described in our publication
        Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.

        For an example, see `the distillation documentation <../../../examples/sentence_transformer/training/distillation/README.html>`_ on extending language models to new languages.

        Args:
            model: SentenceTransformerModel

        References:
            - Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://huggingface.co/papers/2004.09813
            - `Training > Model Distillation <../../../examples/sentence_transformer/training/distillation/README.html>`_
            - `Training > Multilingual Models <../../../examples/sentence_transformer/training/multilingual/README.html>`_

        Requirements:
            1. Usually uses a finetuned teacher M in a knowledge distillation setup

        Inputs:
            +-----------------------------------------+-----------------------------+
            | Texts                                   | Labels                      |
            +=========================================+=============================+
            | sentence                                | model sentence embeddings   |
            +-----------------------------------------+-----------------------------+
            | sentence_1, sentence_2, ..., sentence_N | model sentence embeddings   |
            +-----------------------------------------+-----------------------------+

        Relations:
            - :class:`MarginMSELoss` is equivalent to this loss, but with a margin through a negative pair.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "english": ["The first sentence",  "The second sentence", "The third sentence",  "The fourth sentence"],
                    "french": ["La première phrase",  "La deuxième phrase", "La troisième phrase",  "La quatrième phrase"],
                })

                def compute_labels(batch):
                    return {
                        "label": teacher_model.encode(batch["english"])
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.MSELoss(student_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Concatenate multiple inputs on the batch dimension
        if len(sentence_features) > 1:
            embeddings = torch.cat([self.model(inputs)["sentence_embedding"] for inputs in sentence_features], dim=0)
            # Repeat the labels for each input
            return self.loss_fct(embeddings, labels.repeat(len(sentence_features), 1))

        embeddings = self.model(sentence_features[0])["sentence_embedding"]
        return self.loss_fct(embeddings, labels)

    @property
    def citation(self) -> str:
        return """
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
"""
