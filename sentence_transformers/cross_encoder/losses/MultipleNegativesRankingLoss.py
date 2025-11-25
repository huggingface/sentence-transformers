from __future__ import annotations

from collections.abc import Generator

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.util import fullname


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        num_negatives: int | None = 4,
        scale: int = 10.0,
        activation_fn: nn.Module | None = nn.Sigmoid(),
    ) -> None:
        """
        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

        * Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer)
          out of every single positive and negative (e.g. all answers) in the batch.

        If you provide the optional negatives, they will all be used as extra options from which the model must pick the
        correct positive. Within reason, the harder this "picking" is, the stronger the model will become. Because of
        this, a higher batch size results in more in-batch negatives, which then increases performance (to a point).

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, answer)) as it will sample in each batch ``n-1`` negative docs randomly.

        This loss is also known as InfoNCE loss, SimCSE loss, Cross-Entropy Loss with in-batch negatives, or simply
        in-batch negatives loss.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
            num_negatives (int, optional): Number of in-batch negatives to sample for each anchor. Defaults to 4.
            scale (int, optional): Output of similarity function is multiplied by scale value. Defaults to 10.0.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss. Defaults to :class:`~torch.nn.Sigmoid`.

        .. note::

            The current default values are subject to change in the future. Experimentation is encouraged.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652

        Requirements:
            1. Your model must be initialized with `num_labels = 1` (a.k.a. the default) to predict one class.

        Inputs:
            +-------------------------------------------------+--------+-------------------------------+
            | Texts                                           | Labels | Number of Model Output Labels |
            +=================================================+========+===============================+
            | (anchor, positive) pairs                        | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative) triplets           | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative_1, ..., negative_n) | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.
            - Use :class:`~sentence_transformers.util.mine_hard_negatives` with ``output_format="n-tuple"`` or
              ``output_format="triplet"`` to convert question-answer pairs to triplets with hard negatives.

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "answer": ["Pandas are a kind of bear.", "The capital of France is Paris."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.num_negatives = num_negatives
        self.scale = scale
        self.activation_fn = activation_fn

        self.cross_entropy_loss = nn.CrossEntropyLoss()

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

    def call_model_with_columns(self, anchors: list[str], candidates: list[str]) -> Tensor:
        pairs = list(zip(anchors, candidates))
        return self.call_model_with_pairs(pairs)

    def call_model_with_pairs(self, pairs: list[list[str]]) -> Tensor:
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]
        return logits.squeeze(1)

    def get_in_batch_negatives(
        self, anchors: list[str], candidates: list[list[str]]
    ) -> Generator[list[str], None, None]:
        batch_size = len(anchors)
        num_columns = len(candidates)

        # Given N anchors, we want to select num_negatives negatives for each anchor
        candidates_flattened = [candidate for sublist in candidates for candidate in sublist]

        # Create a mask for each anchor to each candidate index, where the matching positive
        # and hard negatives are masked out.
        mask = ~torch.eye(batch_size, dtype=torch.bool).repeat(1, num_columns)
        if self.num_negatives is not None and self.num_negatives < len(candidates_flattened):
            # From the remaining options, we randomly select num_negatives indices.
            negative_indices = torch.multinomial(mask.float(), self.num_negatives)
        else:
            # If num_negatives is None or larger than the number of candidates, we select all negatives
            # by using the mask as a slicer to get the indices of the negative candidates
            all_indices = torch.arange(batch_size).repeat(batch_size * num_columns, 1)
            negative_indices = all_indices[mask].reshape(batch_size, -1)

        for negative_indices_row in negative_indices.T:
            yield [candidates_flattened[negative_idx] for negative_idx in negative_indices_row]

    def calculate_loss(self, logits: Tensor, batch_size: int) -> Tensor:
        # (bsz, 1 + num_rand_negatives + num_hard_negatives)
        logits = torch.cat(logits, dim=0).reshape(-1, batch_size).T

        # Apply the post-processing on the logits
        if self.activation_fn:
            logits = self.activation_fn(logits)
        if self.scale:
            logits = logits * self.scale

        # For each sample in the batch, the first label is the positive, the rest are negatives
        labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)

        loss = self.cross_entropy_loss(logits, labels)
        return loss

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        anchors = inputs[0]
        positives = inputs[1]
        batch_size = len(anchors)

        scores = [self.call_model_with_columns(anchors, positives)]

        # In-batch negatives:
        for negatives in self.get_in_batch_negatives(anchors, inputs[1:]):
            scores.append(self.call_model_with_columns(anchors, negatives))

        # Hard negatives:
        for negatives in inputs[2:]:
            scores.append(self.call_model_with_columns(anchors, negatives))

        return self.calculate_loss(scores, batch_size)

    def get_config_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "num_negatives": self.num_negatives,
            "activation_fn": fullname(self.activation_fn),
        }
