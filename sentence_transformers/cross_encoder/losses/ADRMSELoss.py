from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class ADRMSELoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        """
        ADR-MSE (Approx Discounted Rank Mean Squared Error) listwise ranking loss for cross-encoders.
        This loss directly minimizes the error between true rank positions and differentiable
        approximations of predicted ranks, with log-discount weighting inspired by nDCG.

        The predicted ranks are approximated in a differentiable manner using the ApproxRank
        formulation: for each document, the approximate rank is the sum of sigmoids over score
        differences with all other documents.

        .. note::

            The number of documents per query can vary between samples with the ``ADRMSELoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:

                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.

                Defaults to None.

        References:
            - Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking: https://huggingface.co/papers/2405.07920
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Texts                                  | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        Recommendations:
            - Use :class:`~sentence_transformers.util.mine_hard_negatives` with ``output_format="labeled-list"``
              to convert question-answer pairs to the required input format with hard negatives.

        Relations:
            - :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` takes the same inputs, and generally
              outperforms other listwise losses.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "docs": [
                        ["Pandas are a kind of bear.", "Pandas are kind of like fish."],
                        ["The capital of France is Paris.", "Paris is the capital of France.", "Paris is quite large."],
                    ],
                    "labels": [[1, 0], [1, 1, 0]],
                })
                loss = losses.ADRMSELoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn or nn.Identity()
        self.mini_batch_size = mini_batch_size

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def approximate_ranks(self, scores: Tensor, mask: Tensor) -> Tensor:
        """Compute differentiable approximate ranks using the ApproxRank formulation.

        For each document i: approx_rank(i) = 1 + sum_{j != i} sigmoid(s_j - s_i).
        Higher scores get lower (better) ranks. Padded positions are excluded via the mask.
        """
        score_diffs = scores.unsqueeze(2) - scores.unsqueeze(1)
        pairwise = torch.sigmoid(score_diffs)
        pairwise = pairwise * mask.unsqueeze(1).float()
        pairwise = pairwise * (1 - torch.eye(scores.size(1), device=scores.device)).unsqueeze(0)
        approx_ranks = 1.0 + pairwise.sum(dim=2)
        return approx_ranks

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute ADR-MSE loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Mean ADR-MSE loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "ADRMSELoss expects a list of labels for each sample, but got a single value for each sample."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"ADRMSELoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
            )

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        # Create input pairs for the model
        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        mini_batch_size = self.mini_batch_size or batch_size
        if mini_batch_size <= 0:
            mini_batch_size = len(pairs)

        logits_list = []
        for i in range(0, len(pairs), mini_batch_size):
            mini_batch_pairs = pairs[i : i + mini_batch_size]

            tokens = self.model.tokenizer(
                mini_batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = tokens.to(self.model.device)

            logits = self.model(**tokens)[0].view(-1)
            logits_list.append(logits)

        logits = torch.cat(logits_list, dim=0)
        logits = self.activation_fn(logits)

        # Place logits into a padded matrix
        logits_matrix = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        mask = torch.zeros((batch_size, max_docs), dtype=torch.bool, device=self.model.device)
        mask[batch_indices, doc_indices] = True

        # Build padded labels matrix
        labels_matrix = torch.full((batch_size, max_docs), 0.0, device=self.model.device)
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()

        # Derive true ranks from labels (padded positions get worst ranks)
        labels_for_ranking = labels_matrix.clone()
        labels_for_ranking[~mask] = float("-inf")
        true_ranks = labels_for_ranking.argsort(dim=1, descending=True).argsort(dim=1).float() + 1.0

        approx_ranks = self.approximate_ranks(logits_matrix, mask)

        # Calculate discounted squared rank error
        discount = 1.0 / torch.log2(true_ranks + 1.0)
        squared_error = (true_ranks - approx_ranks) ** 2
        loss = discount * squared_error

        # Apply mask and reduction
        loss = loss * mask.float()
        num_valid = mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        return loss.sum() / num_valid

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        return {
            "activation_fn": fullname(self.activation_fn),
            "mini_batch_size": self.mini_batch_size,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{reddy2024rankdistillm,
    title={Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking},
    author={Reddy, Revanth Gangi and Doo, JaeHyeok and Xu, Yifei and Sultan, Arafat and Bhat, Ganesh and Zhai, ChengXiang and Ji, Heng},
    year={2024},
    eprint={2405.07920},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
