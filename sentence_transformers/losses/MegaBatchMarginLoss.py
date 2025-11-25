from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class MegaBatchMarginLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        positive_margin: float = 0.8,
        negative_margin: float = 0.3,
        use_mini_batched_version: bool = True,
        mini_batch_size: int = 50,
    ) -> None:
        """
        Given a large batch (like 500 or more examples) of (anchor_i, positive_i) pairs, find for each pair in the batch
        the hardest negative, i.e. find j != i such that cos_sim(anchor_i, positive_j) is maximal. Then create from this a
        triplet (anchor_i, positive_i, positive_j) where positive_j serves as the negative for this triplet.

        Then train as with the triplet loss.

        Args:
            model: SentenceTransformerModel
            positive_margin: Positive margin, cos(anchor, positive)
                should be > positive_margin
            negative_margin: Negative margin, cos(anchor, negative)
                should be < negative_margin
            use_mini_batched_version: As large batch sizes require a lot
                of memory, we can use a mini-batched version. We break
                down the large batch into smaller batches with fewer
                examples.
            mini_batch_size: Size for the mini-batches. Should be a
                divisor for the batch size in your data loader.

        References:
            - This loss function was inspired by the ParaNMT paper: https://www.aclweb.org/anthology/P18-1042/

        Requirements:
            1. (anchor, positive) pairs
            2. Large batches (500 or more examples)

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, losses
                from datasets import Dataset

                train_batch_size = 250
                train_mini_batch_size = 32

                model = SentenceTransformer('all-MiniLM-L6-v2')
                train_dataset = Dataset.from_dict({
                    "anchor": [f"This is sentence number {i}" for i in range(500)],
                    "positive": [f"This is sentence number {i}" for i in range(1, 501)],
                })
                loss = losses.MegaBatchMarginLoss(model=model, mini_batch_size=train_mini_batch_size)

                args = SentenceTransformerTrainingArguments(
                    output_dir="output",
                    per_device_train_batch_size=train_batch_size,
                )
                trainer = SentenceTransformerTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.mini_batch_size = mini_batch_size
        self.forward = self.forward_mini_batched if use_mini_batched_version else self.forward_non_mini_batched

    def forward_mini_batched(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        anchor, positive = sentence_features
        feature_names = list(anchor.keys())
        batch_size = len(positive[next(iter(positive))])

        all_positive_emb = []
        with torch.no_grad():
            self.model.eval()
            for start_idx in range(0, batch_size, self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                input_mini_batch = {k: v[start_idx:end_idx] for k, v in positive.items()}
                all_positive_emb.append(self.model(input_mini_batch)["sentence_embedding"].detach())
            self.model.train()
        all_positive_emb = torch.cat(all_positive_emb, dim=0)

        diagonal_matrix = torch.eye(len(all_positive_emb), len(all_positive_emb), device=all_positive_emb.device)

        # Iterate over the triplets (anchor, positive, hardest_negative) in smaller mini_batch sizes
        for start_idx in range(0, len(all_positive_emb), self.mini_batch_size):
            end_idx = start_idx + self.mini_batch_size
            anchor_emb = self.model({key: anchor[key][start_idx:end_idx] for key in feature_names})[
                "sentence_embedding"
            ]

            # Find hard negatives. For each anchor, find the hardest negative
            # Store them in the triplets (anchor, positive, hardest_negative)
            hard_negative_features = {key: [] for key in feature_names}
            with torch.no_grad():
                cos_scores = util.pytorch_cos_sim(anchor_emb, all_positive_emb)
                negative_scores = (
                    cos_scores - 2 * diagonal_matrix[start_idx:end_idx]
                )  # Remove positive scores along the diagonal, set them to -1 so that they are not selected by the max() operation
                negatives_max, negatives_ids = torch.max(negative_scores, dim=1)

            for hard_negative_id in negatives_ids:
                for key in feature_names:
                    hard_negative_features[key].append(positive[key][hard_negative_id])

            for key in feature_names:
                hard_negative_features[key] = torch.stack(hard_negative_features[key])

            # Compute differentiable negative and positive embeddings
            positive_emb = self.model({key: positive[key][start_idx:end_idx] for key in feature_names})[
                "sentence_embedding"
            ]
            negative_emb = self.model(hard_negative_features)["sentence_embedding"]

            assert anchor_emb.shape == positive_emb.shape
            assert anchor_emb.shape == negative_emb.shape

            # Compute loss
            pos_cosine = F.cosine_similarity(anchor_emb, positive_emb)
            neg_cosine = F.cosine_similarity(anchor_emb, negative_emb)
            losses = F.relu(self.positive_margin - pos_cosine) + F.relu(neg_cosine - self.negative_margin)
            losses = losses.mean()

            # Backpropagate unless it is the last mini batch. The last mini-batch will be back propagated by the outside train loop
            if end_idx < len(cos_scores):
                losses.backward()

        return losses

    ##### Non mini-batched version ###
    def forward_non_mini_batched(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        embeddings_a, embeddings_b = reps

        cos_scores = util.pytorch_cos_sim(embeddings_a, embeddings_b)
        positive_scores = torch.diagonal(cos_scores)
        negative_scores = cos_scores - (
            2 * torch.eye(*cos_scores.shape, device=cos_scores.device)
        )  # Remove positive scores along the diagonal
        negatives_max, _ = torch.max(negative_scores, dim=1)
        losses = F.relu(self.positive_margin - positive_scores) + F.relu(negatives_max - self.negative_margin)
        return losses.mean()

    @property
    def citation(self) -> str:
        return """
@inproceedings{wieting-gimpel-2018-paranmt,
    title = "{P}ara{NMT}-50{M}: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations",
    author = "Wieting, John and Gimpel, Kevin",
    editor = "Gurevych, Iryna and Miyao, Yusuke",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1042",
    doi = "10.18653/v1/P18-1042",
    pages = "451--462",
}
"""
