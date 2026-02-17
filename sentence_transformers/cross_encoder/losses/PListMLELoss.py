from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class PListMLELambdaWeight(nn.Module):
    """Base class for implementing weighting schemes in Position-Aware ListMLE Loss."""

    def __init__(self, rank_discount_fn=None) -> None:
        """
        Initialize a lambda weight for PListMLE loss.

        Args:
            rank_discount_fn: Function that computes a discount for each rank position.
                              If None, uses default discount of 2^(num_docs - rank) - 1.
        """
        super().__init__()
        self.rank_discount_fn = rank_discount_fn

    def forward(self, mask: Tensor) -> Tensor:
        """
        Calculate position-aware weights for the PListMLE loss.

        Args:
            mask: A boolean mask indicating valid positions [batch_size, num_docs]

        Returns:
            Tensor: Weights for each position [batch_size, num_docs]
        """
        if self.rank_discount_fn is not None:
            return self.rank_discount_fn(mask)

        # Apply default rank discount: 2^(num_docs - rank) - 1
        num_docs_per_query = mask.sum(dim=1, keepdim=True)
        ranks = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
        weights = torch.pow(2.0, num_docs_per_query - ranks) - 1.0
        weights = weights * mask
        return weights


class PListMLELoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        lambda_weight: PListMLELambdaWeight | None = PListMLELambdaWeight(),
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
        respect_input_order: bool = True,
    ) -> None:
        """
        PListMLE loss for learning to rank with position-aware weighting. This loss function implements
        the ListMLE ranking algorithm which uses a list-wise approach based on maximum likelihood
        estimation of permutations. It maximizes the likelihood of the permutation induced by the
        ground truth labels with position-aware weighting.

        This loss is also known as Position-Aware ListMLE or p-ListMLE.

        .. note::

            The number of documents per query can vary between samples with the ``PListMLELoss``.

        Args:
            model (CrossEncoder): CrossEncoder model to be trained
            lambda_weight (PListMLELambdaWeight, optional): Weighting scheme to use. When specified,
                implements Position-Aware ListMLE which applies different weights to different rank
                positions. Default is None (standard PListMLE).
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the
                loss. Defaults to :class:`~torch.nn.Identity`.
            mini_batch_size (int, optional): Number of samples to process in each forward pass. This has a significant
                impact on the memory consumption and speed of the training process. Three cases are possible:

                - If ``mini_batch_size`` is None, the ``mini_batch_size`` is set to the batch size.
                - If ``mini_batch_size`` is greater than 0, the batch is split into mini-batches of size ``mini_batch_size``.
                - If ``mini_batch_size`` is <= 0, the entire batch is processed at once.

                Defaults to None.
            respect_input_order (bool): Whether to respect the original input order of documents.
                If True, assumes the input documents are already ordered by relevance (most relevant first).
                If False, sorts documents by label values. Defaults to True.

        References:
            - Position-Aware ListMLE: A Sequential Learning Process for Ranking: https://auai.org/uai2014/proceedings/individuals/164.pdf
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_

        Requirements:
            1. Query with multiple documents (listwise approach)
            2. Documents must have relevance scores/labels. Both binary and continuous labels are supported.
            3. Documents must be sorted in a defined rank order.

        Inputs:
            +----------------------------------------+--------------------------------+-------------------------------+
            | Texts                                  | Labels                         | Number of Model Output Labels |
            +========================================+================================+===============================+
            | (query, [doc1, doc2, ..., docN])       | [score1, score2, ..., scoreN]  | 1                             |
            +----------------------------------------+--------------------------------+-------------------------------+

        Recommendations:
            - Use :class:`~sentence_transformers.util.hard_negatives.mine_hard_negatives` with ``output_format="labeled-list"``
              to convert question-answer pairs to the required input format with hard negatives.

        Relations:
            - The :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` is an extension of the
              :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss` and allows for positional weighting
              of the loss. :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` generally outperforms
              :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss` and is recommended over it.
            - :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` takes the same inputs, and generally
              outperforms this loss.

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

                # Either: Position-Aware ListMLE with default weighting
                lambda_weight = losses.PListMLELambdaWeight()
                loss = losses.PListMLELoss(model, lambda_weight=lambda_weight)

                # or: Position-Aware ListMLE with custom weighting function
                def custom_discount(ranks): # e.g. ranks: [1, 2, 3, 4, 5]
                    return 1.0 / torch.log1p(ranks)
                lambda_weight = losses.PListMLELambdaWeight(rank_discount_fn=custom_discount)
                loss = losses.PListMLELoss(model, lambda_weight=lambda_weight)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.lambda_weight = lambda_weight
        self.activation_fn = activation_fn or nn.Identity()
        self.mini_batch_size = mini_batch_size
        self.respect_input_order = respect_input_order
        self.eps = 1e-10

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute PListMLE loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Mean PListMLE loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "PListMLELoss expects a list of labels for each sample, but got a single value for each sample."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"PListMLELoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
            )

        queries, docs_list = inputs
        docs_per_query = [len(docs) for docs in docs_list]
        max_docs = max(docs_per_query)
        batch_size = len(queries)

        if docs_per_query != [len(labels) for labels in labels]:
            raise ValueError(
                f"Number of documents per query in inputs ({docs_per_query}) does not match number of labels per query ({[len(labels) for labels in labels]})."
            )

        pairs = [(query, document) for query, docs in zip(queries, docs_list) for document in docs]

        if not pairs:
            # Handle edge case where there are no documents
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

        # Create output tensor filled with a very small value for padded logits
        logits_matrix = torch.full((batch_size, max_docs), 1e-16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        # Create a mask for valid entries
        mask = torch.zeros_like(logits_matrix, dtype=torch.bool)
        mask[batch_indices, doc_indices] = True

        # Convert labels to tensor matrix
        labels_matrix = torch.full_like(logits_matrix, -float("inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()

        if not torch.any(mask):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        if not self.respect_input_order:
            # Sort by labels in descending order if not respecting input order.
            sorted_labels, indices = labels_matrix.sort(descending=True, dim=1)
            sorted_logits = torch.gather(logits_matrix, 1, indices)
        else:
            # Use the original input order, assuming it's already ordered by relevance
            sorted_logits = logits_matrix

        # Compute log-likelihood using Plackett-Luce model
        scores = sorted_logits.exp()
        cumsum_scores = torch.flip(torch.cumsum(torch.flip(scores, [1]), 1), [1])
        log_probs = sorted_logits - torch.log(cumsum_scores + self.eps)

        # Apply position-aware lambda weights if specified. If None, then this loss
        # is just ListMLE.
        if self.lambda_weight is not None:
            lambda_weight = self.lambda_weight(mask)
            # Normalize weights to sum to 1
            lambda_weight = lambda_weight / (lambda_weight.sum(dim=1, keepdim=True) + self.eps)
            log_probs = log_probs * lambda_weight

        # Sum the log probabilities for each list and mask padded entries
        log_probs[~mask] = 0.0
        per_query_losses = -torch.sum(log_probs, dim=1)

        if not torch.any(per_query_losses):
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        # Average loss over all lists
        return torch.mean(per_query_losses)

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {
            "lambda_weight": None if self.lambda_weight is None else fullname(self.lambda_weight),
            "activation_fn": fullname(self.activation_fn),
            "mini_batch_size": self.mini_batch_size,
            "respect_input_order": self.respect_input_order,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{lan2014position,
  title={Position-Aware ListMLE: A Sequential Learning Process for Ranking},
  author={Lan, Yanyan and Zhu, Yadong and Guo, Jiafeng and Niu, Shuzi and Cheng, Xueqi},
  booktitle={UAI},
  volume={14},
  pages={449--458},
  year={2014}
}
"""
