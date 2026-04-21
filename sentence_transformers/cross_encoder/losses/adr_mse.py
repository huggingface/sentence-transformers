from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder.model import CrossEncoder
from sentence_transformers.util import batch_to_device, fullname


class ADRMSELoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        alpha: float = 1.0,
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        r"""
        ADR-MSE (Approx Discounted Rank Mean Squared Error) listwise ranking loss for cross-encoders.
        This loss directly minimizes the error between true rank positions and differentiable
        approximations of predicted ranks, with log-discount weighting inspired by nDCG.

        The predicted rank of each document is approximated in a differentiable manner using the
        ApproxRank formulation :math:`\hat{\pi}(d_i) = 1 + \sum_{j \neq i} \sigma(\alpha \cdot (s_j - s_i))`,
        where :math:`\sigma` is the logistic sigmoid. Tied labels are handled with average ranks,
        so documents sharing a relevance label share a fractional target rank that the ApproxRank
        can attain when the model assigns them equal scores.

        .. note::

            The number of documents per query can vary between samples with the ``ADRMSELoss``.

        .. note::

            Ranks for tied labels are averaged so that the ApproxRank target is attainable
            when the model assigns tied documents equal scores, avoiding a non-zero minimum
            loss from arbitrary tie-breaking.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            alpha (float): Temperature inside the ApproxRank sigmoid. Higher values give a tighter
                (less smooth) approximation of the hard rank, at the cost of weaker gradients for
                well-separated pairs. Defaults to 1.0, matching the Rank-DistiLLM paper.
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
        self.alpha = alpha
        self.activation_fn = activation_fn or nn.Identity()
        self.mini_batch_size = mini_batch_size

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def approximate_ranks(self, scores: Tensor, mask: Tensor) -> Tensor:
        """Compute differentiable approximate ranks using the ApproxRank formulation.

        For each document i: ``approx_rank(i) = 1 + sum_{j != i} sigmoid(alpha * (s_j - s_i))``.
        Higher scores get lower (better) ranks. Padded positions are excluded via the mask.
        """
        score_diffs = scores.unsqueeze(1) - scores.unsqueeze(2)
        pairwise = torch.sigmoid(self.alpha * score_diffs)
        pairwise = pairwise * mask.unsqueeze(1).float()
        pairwise = pairwise * (1 - torch.eye(scores.size(1), device=scores.device)).unsqueeze(0)
        approx_ranks = 1.0 + pairwise.sum(dim=2)
        return approx_ranks

    def forward(
        self,
        inputs: list[list[str] | list[list[str]]],
        labels: list[Tensor],
        prompt: str | None = None,
        task: str | None = None,
    ) -> Tensor:
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
            raise ValueError(f"ADRMSELoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs.")

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

            tokens = self.model.preprocess(mini_batch_pairs, prompt=prompt, task=task)
            tokens = batch_to_device(tokens, self.model.device)

            logits = self.model(tokens)["scores"].view(-1)
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

        # Derive true ranks from labels with average-rank handling of ties, so that
        # documents sharing a label share a fractional target rank that the ApproxRank
        # can attain when the model assigns tied docs equal scores. Padded positions are
        # excluded from the counts via the mask and are masked out of the loss later.
        labels_j = labels_matrix.unsqueeze(1)
        labels_i = labels_matrix.unsqueeze(2)
        valid_j = mask.unsqueeze(1)
        greater = ((labels_j > labels_i) & valid_j).sum(dim=-1).float()
        equal = ((labels_j == labels_i) & valid_j).sum(dim=-1).float()
        true_ranks = greater + (equal + 1.0) / 2.0

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
            "alpha": self.alpha,
            "activation_fn": fullname(self.activation_fn),
            "mini_batch_size": self.mini_batch_size,
        }

    @property
    def citation(self) -> str:
        return """
@inbook{schlatt2025rankdistillm,
   title={Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-ranking},
   author={Schlatt, Ferdinand and Fröbe, Maik and Scells, Harrisen and Zhuang, Shengyao and Koopman, Bevan and Zuccon, Guido and Stein, Benno and Potthast, Martin and Hagen, Matthias},
   booktitle={Advances in Information Retrieval (ECIR 2025)},
   year={2025},
   doi={10.1007/978-3-031-88714-7_31},
}
"""
