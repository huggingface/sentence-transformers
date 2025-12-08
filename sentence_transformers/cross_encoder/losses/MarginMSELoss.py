from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.util import fullname


class MarginMSELoss(nn.Module):
    def __init__(self, model: CrossEncoder, activation_fn: nn.Module = nn.Identity(), **kwargs) -> None:
        """
        Computes the MSE loss between ``|sim(Query, Pos) - sim(Query, Neg)|`` and ``|gold_sim(Query, Pos) - gold_sim(Query, Neg)|``.
        This loss is often used to distill a cross-encoder model from a teacher cross-encoder model or gold labels.

        In contrast to :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss`, the two passages do not
        have to be strictly positive and negative, both can be relevant or not relevant for a given query. This can be
        an advantage of MarginMSELoss over MultipleNegativesRankingLoss.

        .. note::

            Be mindful of the magnitude of both the labels and what the model produces. If the teacher model produces
            logits with Sigmoid to bound them to [0, 1], then you may wish to use a Sigmoid activation function in the loss.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss.
            **kwargs: Additional keyword arguments passed to the underlying :class:`torch.nn.MSELoss`.

        References:
            - Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation: https://huggingface.co/papers/2010.02666
            - `Cross Encoder > Training Examples > Distillation <../../../examples/cross_encoder/training/distillation/README.html>`_

        Requirements:
            1. Your model must be initialized with `num_labels = 1` (a.k.a. the default) to predict one class.
            2. Usually uses a finetuned CrossEncoder teacher M in a knowledge distillation setup.

        Inputs:
            +------------------------------------------------+--------------------------------------------------------------------------------------------+-------------------------------+
            | Texts                                          | Labels                                                                                     | Number of Model Output Labels |
            +================================================+============================================================================================+===============================+
            | (query, passage_one, passage_two) triplets     | gold_sim(query, passage_one) - gold_sim(query, passage_two)                                | 1                             |
            +------------------------------------------------+--------------------------------------------------------------------------------------------+-------------------------------+
            | (query, passage_one, passage_two) triplets     | [gold_sim(query, passage_one), gold_sim(query, passage_two)]                               | 1                             |
            +------------------------------------------------+--------------------------------------------------------------------------------------------+-------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [gold_sim(query, positive) - gold_sim(query, negative_i) for i in 1..n]                    | 1                             |
            +------------------------------------------------+--------------------------------------------------------------------------------------------+-------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [gold_sim(query, positive), gold_sim(query, negative_1), ..., gold_sim(query, negative_n)] | 1                             |
            +------------------------------------------------+--------------------------------------------------------------------------------------------+-------------------------------+

        Relations:
            - :class:`MSELoss` is similar to this loss, but without a margin through the negative pair.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                student_model = CrossEncoder("microsoft/mpnet-base")
                teacher_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "positive": ["Pandas are a kind of bear.", "The capital of France is Paris."],
                    "negative": ["Pandas are a kind of fish.", "The capital of France is Berlin."],
                })

                def compute_labels(batch):
                    positive_scores = teacher_model.predict(list(zip(batch["query"], batch["positive"])))
                    negative_scores = teacher_model.predict(list(zip(batch["query"], batch["negative"])))
                    return {
                        "label": positive_scores - negative_scores
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.MarginMSELoss(student_model)

                trainer = CrossEncoderTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn
        self.loss_fct = nn.MSELoss(**kwargs)

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor | list[Tensor]) -> Tensor:
        anchors = inputs[0]
        positives = inputs[1]
        negatives = inputs[2:]
        batch_size = len(anchors)

        # If there's multiple scores, then `labels` is a list of tensors. We need to stack them into
        # a single tensor of shape (batch_size, num_columns - 1)
        if isinstance(labels, list):
            labels = torch.stack(labels)

        if labels.shape == (batch_size, len(negatives) + 1):
            # If labels are given as a single score for positive and multiple negatives,
            # we need to adjust the labels to be the difference between positive and negatives
            labels = labels[:, 0].unsqueeze(1) - labels[:, 1:]

        # Ensure the shape is (batch_size, num_negatives)
        if labels.shape == (batch_size,):
            labels = labels.unsqueeze(1)

        if labels.shape != (batch_size, len(negatives)):
            raise ValueError(
                f"Labels shape {labels.shape} does not match expected shape {(batch_size, len(negatives))}. "
                "Ensure that your dataset labels/scores are 1) lists of differences between positive scores and "
                "negatives scores (length `num_negatives`), or 2) lists of positive and negative scores "
                "(length `num_negatives + 1`)."
            )

        positive_pairs = list(zip(anchors, positives))
        positive_logits = self.logits_from_pairs(positive_pairs)
        negative_logits_list = []
        for negative in negatives:
            negative_pairs = list(zip(anchors, negative))
            negative_logits_list.append(self.logits_from_pairs(negative_pairs))

        margin_logits = positive_logits.unsqueeze(1) - torch.stack(negative_logits_list, dim=1)
        loss = self.loss_fct(margin_logits, labels.float())
        return loss

    def logits_from_pairs(self, pairs: list[tuple[str, str]]) -> Tensor:
        """
        Computes the logits for a list of pairs using the model.

        Args:
            pairs (list[tuple[str, str]]): A list of pairs of strings (query, passage).

        Returns:
            Tensor: The logits for the pairs.
        """
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0].view(-1)
        return self.activation_fn(logits)

    def get_config_dict(self):
        return {
            "activation_fn": fullname(self.activation_fn),
        }

    @property
    def citation(self) -> str:
        return """
@misc{hofstätter2021improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2021},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
