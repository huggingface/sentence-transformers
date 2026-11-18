from __future__ import annotations

from collections.abc import Generator

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder.model import CrossEncoder
from sentence_transformers.util import batch_to_device, fullname


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
            model (:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`): A CrossEncoder model to be trained.
            num_negatives (int, optional): Maximum number of in-batch negatives to sample without replacement for each
                anchor, excluding its own positive and hard negatives. Use None for all available candidates or 0
                for no in-batch negatives. All explicit hard negatives are always included. Defaults to 4.
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
            | Inputs                                          | Labels | Number of Model Output Labels |
            +=================================================+========+===============================+
            | (anchor, positive) pairs                        | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative) triplets           | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative_1, ..., negative_n) | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+

        Variable-length hard negatives:
            Use exactly three input columns: ``(anchor, positive, [negative_1, ..., negative_n])``.
            Each negative list may have a different length, including zero. The loss averages over anchors,
            not candidate pairs, so rows with more negatives do not receive extra weight. This format is
            supported by the CrossEncoder variants of these losses, not the SentenceTransformer variants.
            See `Variable-length hard negatives <../../../docs/cross_encoder/loss_overview.html#variable-length-hard-negatives>`_
            for a training example and the list-aware no-duplicates sampler configuration.

        Recommendations:
            - For scalar text columns, use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.base.sampler.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.
            - Use :class:`~sentence_transformers.util.hard_negatives.mine_hard_negatives` with ``output_format="n-tuple"`` or
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

    def call_model_with_columns(
        self, anchors: list[str], candidates: list[str], prompt: str | None = None, task: str | None = None
    ) -> Tensor:
        pairs = list(zip(anchors, candidates))
        return self.call_model_with_pairs(pairs, prompt=prompt, task=task)

    def call_model_with_pairs(
        self, pairs: list[list[str]], prompt: str | None = None, task: str | None = None
    ) -> Tensor:
        inputs = self.model.preprocess(pairs, prompt=prompt, task=task)
        inputs = batch_to_device(inputs, self.model.device)
        logits = self.model(inputs)["scores"]
        return logits.squeeze(1)

    def get_in_batch_negatives(
        self, anchors: list[str], candidates: list[list[str]]
    ) -> Generator[list[str], None, None]:
        if self.num_negatives == 0:
            return
        batch_size = len(anchors)
        num_columns = len(candidates)

        # Given N anchors, we want to select num_negatives negatives for each anchor
        candidates_flattened = [candidate for sublist in candidates for candidate in sublist]

        # Create a mask for each anchor to each candidate index, where the matching positive
        # and hard negatives are masked out.
        mask = ~torch.eye(batch_size, dtype=torch.bool).repeat(1, num_columns)

        # Valid negatives for each anchor are all candidates except the ones that are the positive or hard negatives
        # And num_columns = 1(positive) + num_hard_negatives
        valid_negative_num = len(candidates_flattened) - num_columns

        if self.num_negatives is not None and self.num_negatives < valid_negative_num:
            # From the remaining options, we randomly select num_negatives indices.
            negative_indices = torch.multinomial(mask.float(), self.num_negatives)
        else:
            # If num_negatives is None or larger than the number of valid negatives, we select all negatives
            # by using the mask as a slicer to get the indices of the negative candidates
            all_indices = torch.arange(batch_size * num_columns).repeat(batch_size, 1)
            negative_indices = all_indices[mask].reshape(batch_size, -1)

        for negative_indices_row in negative_indices.T:
            yield [candidates_flattened[negative_idx] for negative_idx in negative_indices_row]

    def _get_ragged_pairs(
        self, inputs: list[list[str] | list[list[str]]]
    ) -> tuple[list[tuple[str, str]], list[int]] | None:
        """Build anchor-major pairs for a list-valued hard-negative column.

        Keep the fixed-column path unchanged, including its sampling and model-call order.
        Candidate ownership, rather than text equality, excludes a row's own candidates
        from its in-batch pool, just as in the fixed-column format.
        """
        if not any(isinstance(value, list) for column in inputs[2:] for value in column):
            return None
        if len(inputs) != 3:
            raise ValueError("List-valued negatives require exactly three columns: anchor, positive, negatives.")
        anchors, positives, negatives = inputs
        batch_size = len(anchors)
        if not batch_size or len(positives) != batch_size or len(negatives) != batch_size:
            raise ValueError("Anchor, positive, and negatives columns must have the same non-zero length.")
        if not all(isinstance(text, str) for text in anchors + positives):
            raise ValueError("Each anchor and positive must be a string.")
        if not all(isinstance(row, list) and all(isinstance(text, str) for text in row) for row in negatives):
            raise ValueError("Each entry in the negatives column must be a list of strings, which may be empty.")
        if self.num_negatives is not None and self.num_negatives < 0:
            raise ValueError("num_negatives must be non-negative or None.")

        candidates = [text for positive, row in zip(positives, negatives) for text in [positive, *row]]
        pairs = []
        candidate_counts = []
        offset = 0
        for anchor, row in zip(anchors, negatives):
            owned_count = 1 + len(row)
            available = len(candidates) - owned_count
            if self.num_negatives == 0:
                indices = range(0)
            elif self.num_negatives is not None and self.num_negatives < available:
                # Sample positions without replacement from the pool with this row removed.
                indices = torch.randperm(available)[: self.num_negatives].tolist()
            else:
                indices = range(available)
            selected = candidates[offset : offset + owned_count]
            selected.extend(candidates[index if index < offset else index + owned_count] for index in indices)
            pairs.extend((anchor, candidate) for candidate in selected)
            candidate_counts.append(len(selected))
            offset += owned_count
        return pairs, candidate_counts

    def calculate_loss(
        self, logits: list[Tensor], batch_size: int, candidate_counts: list[int] | None = None
    ) -> Tensor:
        if candidate_counts is None:
            # Fixed columns are scored column-major, as before.
            logits = torch.cat(logits, dim=0).reshape(-1, batch_size).T
            if self.activation_fn:
                logits = self.activation_fn(logits)
            if self.scale:
                logits = logits * self.scale
        else:
            # Transform each real candidate row before padding. Besides changing
            # the softmax denominator, padded infinities can produce NaN gradients
            # in parameterized activations such as PReLU.
            rows = []
            for row in torch.cat(logits).split(candidate_counts):
                row = row.unsqueeze(0)
                if self.activation_fn:
                    row = self.activation_fn(row)
                if self.scale:
                    row = row * self.scale
                rows.append(row.squeeze(0))
            logits = nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=-torch.inf)

        # For each sample in the batch, the first label is the positive, the rest are negatives
        labels = torch.zeros(batch_size, device=logits.device, dtype=torch.long)

        loss = self.cross_entropy_loss(logits, labels)
        return loss

    def forward(
        self,
        inputs: list[list[str] | list[list[str]]],
        labels: Tensor,
        prompt: str | None = None,
        task: str | None = None,
    ) -> Tensor:
        if ragged := self._get_ragged_pairs(inputs):
            pairs, candidate_counts = ragged
            batch_size = len(inputs[0])
            scores = [
                self.call_model_with_pairs(pairs[start : start + batch_size], prompt=prompt, task=task)
                for start in range(0, len(pairs), batch_size)
            ]
            return self.calculate_loss(scores, batch_size, candidate_counts)

        anchors = inputs[0]
        positives = inputs[1]
        batch_size = len(anchors)

        scores = [self.call_model_with_columns(anchors, positives, prompt=prompt, task=task)]

        # In-batch negatives:
        for negatives in self.get_in_batch_negatives(anchors, inputs[1:]):
            scores.append(self.call_model_with_columns(anchors, negatives, prompt=prompt, task=task))

        # Hard negatives:
        for negatives in inputs[2:]:
            scores.append(self.call_model_with_columns(anchors, negatives, prompt=prompt, task=task))

        return self.calculate_loss(scores, batch_size)

    def get_config_dict(self) -> dict[str, float]:
        return {
            "scale": self.scale,
            "num_negatives": self.num_negatives,
            "activation_fn": fullname(self.activation_fn),
        }
