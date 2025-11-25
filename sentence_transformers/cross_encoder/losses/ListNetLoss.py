from __future__ import annotations

import torch
from torch import Tensor, nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import fullname


class ListNetLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
    ) -> None:
        """
        ListNet loss for learning to rank. This loss function implements the ListNet ranking algorithm
        which uses a list-wise approach to learn ranking models. It minimizes the cross entropy
        between the predicted ranking distribution and the ground truth ranking distribution.
        The implementation is optimized to handle padded documents efficiently by only processing
        valid documents during model inference.

        .. note::

            The number of documents per query can vary between samples with the ``ListNetLoss``.

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
            - Learning to Rank: From Pairwise Approach to Listwise Approach: https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
            - Context-Aware Learning to Rank with Self-Attention: https://huggingface.co/papers/2005.10084
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
                loss = losses.ListNetLoss(model)

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
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} supports a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str], list[list[str]]], labels: list[Tensor]) -> Tensor:
        """
        Compute ListNet loss for a batch of queries and their documents.

        Args:
            inputs: List of (queries, documents_list)
            labels: Ground truth relevance scores, shape (batch_size, num_documents)

        Returns:
            Tensor: Mean ListNet loss over the batch
        """
        if isinstance(labels, Tensor):
            raise ValueError(
                "ListNetLoss expects a list of labels for each sample, but got a single value for each sample."
            )

        if len(inputs) != 2:
            raise ValueError(
                f"ListNetLoss expects two inputs (queries, documents_list), but got {len(inputs)} inputs."
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

        # Create output tensor filled with 0 (padded logits will be ignored via labels)
        logits_matrix = torch.full((batch_size, max_docs), -1e16, device=self.model.device)

        # Place logits in the desired positions in the logit matrix
        doc_indices = torch.cat([torch.arange(len(docs)) for docs in docs_list], dim=0)
        batch_indices = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(docs_per_query))
        logits_matrix[batch_indices, doc_indices] = logits

        # Idem for labels, but fill with -inf to 0 out padded logits in the loss
        labels_matrix = torch.full_like(logits_matrix, float("-inf"))
        labels_matrix[batch_indices, doc_indices] = torch.cat(labels, dim=0).float()
        labels_matrix = labels_matrix.to(self.model.device)

        # Compute cross entropy loss between distributions
        loss = self.cross_entropy_loss(logits_matrix, labels_matrix.softmax(dim=1))

        return loss

    def get_config_dict(self) -> dict[str, float]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        return {"activation_fn": fullname(self.activation_fn), "mini_batch_size": self.mini_batch_size}

    @property
    def citation(self) -> str:
        return """
@inproceedings{cao2007learning,
    title={Learning to Rank: From Pairwise Approach to Listwise Approach},
    author={Cao, Zhe and Qin, Tao and Liu, Tie-Yan and Tsai, Ming-Feng and Li, Hang},
    booktitle={Proceedings of the 24th international conference on Machine learning},
    pages={129--136},
    year={2007}
}
"""
