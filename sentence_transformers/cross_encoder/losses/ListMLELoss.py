from __future__ import annotations

from torch import nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.PListMLELoss import PListMLELoss


class ListMLELoss(PListMLELoss):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module | None = nn.Identity(),
        mini_batch_size: int | None = None,
        respect_input_order: bool = True,
    ) -> None:
        """
        This loss function implements the ListMLE learning to rank algorithm, which uses a list-wise
        approach based on maximum likelihood estimation of permutations. It maximizes the likelihood
        of the permutation induced by the ground truth labels.

        .. note::

            The number of documents per query can vary between samples with the ``ListMLELoss``.

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
            respect_input_order (bool): Whether to respect the original input order of documents.
                If True, assumes the input documents are already ordered by relevance (most relevant first).
                If False, sorts documents by label values. Defaults to True.

        References:
            - Listwise approach to learning to rank: theory and algorithm: https://dl.acm.org/doi/abs/10.1145/1390156.1390306
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

                # Standard ListMLE loss respecting input order
                loss = losses.ListMLELoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(
            model=model,
            lambda_weight=None,
            activation_fn=activation_fn,
            mini_batch_size=mini_batch_size,
            respect_input_order=respect_input_order,
        )

    def get_config_dict(self) -> dict[str, float | int | str | None]:
        """
        Get configuration parameters for this loss function.

        Returns:
            Dictionary containing the configuration parameters
        """
        config = super().get_config_dict()
        del config["lambda_weight"]
        return config

    @property
    def citation(self) -> str:
        return """
@inproceedings{10.1145/1390156.1390306,
    title = {Listwise Approach to Learning to Rank - Theory and Algorithm},
    author = {Xia, Fen and Liu, Tie-Yan and Wang, Jue and Zhang, Wensheng and Li, Hang},
    booktitle = {Proceedings of the 25th International Conference on Machine Learning},
    pages = {1192-1199},
    year = {2008},
    url = {https://doi.org/10.1145/1390156.1390306},
}
"""
