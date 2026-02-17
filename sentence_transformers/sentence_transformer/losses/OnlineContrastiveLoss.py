from __future__ import annotations

from collections.abc import Iterable

import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.model import SentenceTransformer

from .ContrastiveLoss import SiameseDistanceMetric


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self, model: SentenceTransformer, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ) -> None:
        """
        This Online Contrastive loss is similar to :class:`ConstrativeLoss`, but it selects hard positive (positives that
        are far apart) and hard negative pairs (negatives that are close) and computes the loss only for these pairs.
        This loss often yields better performances than ContrastiveLoss.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used
            margin: Negative samples (label == 0) should have a distance
                of at least the margin value.

        References:
            - `Training Examples > Quora Duplicate Questions <../../../examples/sentence_transformer/training/quora_duplicate_questions/README.html>`_

        Requirements:
            1. (anchor, positive/negative) pairs
            2. Data should include hard positives and hard negatives

        Inputs:
            +-----------------------------------------------+------------------------------+
            | Texts                                         | Labels                       |
            +===============================================+==============================+
            | (anchor, positive/negative) pairs             | 1 if positive, 0 if negative |
            +-----------------------------------------------+------------------------------+

        Relations:
            - :class:`ContrastiveLoss` is similar, but does not use hard positive and hard negative pairs.
              :class:`OnlineContrastiveLoss` often yields better results.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "label": [1, 0],
                })
                loss = losses.OnlineContrastiveLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor, size_average=False) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        distance_matrix = self.distance_metric(embeddings[0], embeddings[1])
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
