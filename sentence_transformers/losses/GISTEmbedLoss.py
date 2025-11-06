from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class GISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
        contrast_anchors: bool = True,
        contrast_positives: bool = True,
        gather_across_devices: bool = False,
    ) -> None:
        """
        This loss is used to train a SentenceTransformer model using the GISTEmbed algorithm.
        It takes a model and a guide model as input, and uses the guide model to guide the
        in-batch negative sample selection. The cosine similarity is used to compute the loss
        and the temperature parameter is used to scale the cosine similarities.

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        Args:
            model: SentenceTransformer model based on a `transformers` model.
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities. Inverse of the ``scale`` parameter
                in :class:`MultipleNegativesRankingLoss`.
            margin_strategy: Strategy used for false negative filtering. One of {"absolute", "relative"}.
            margin: The margin value for filtering negatives. Defaults to 0.0, together with the "absolute" strategy,
                this only removes negatives that are more similar to the query than the positive is to the query.
            contrast_anchors: If True, include anchor-anchor pairs in the loss computation, resulting in the embeddings
                of the anchors being pushed further apart. Defaults to True, following the original GISTEmbed paper.
            contrast_positives: If True, include positive-positive pairs in the loss computation, resulting in the embeddings
                of the positives being pushed further apart. Defaults to True, following the original GISTEmbed paper,
                but setting to False may yield better results in some retrieval tasks.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.

        References:
            - For further details, see: https://huggingface.co/papers/2402.16829

        Requirements:
            1. (anchor, positive, negative) triplets
            2. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`MultipleNegativesRankingLoss` is similar to this loss, but it does not use
              a guide model to guide the in-batch negative sample selection. `GISTEmbedLoss` yields
              a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                guide = SentenceTransformer("all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.GISTEmbedLoss(model, guide)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.guide = guide
        self.temperature = temperature
        self.similarity_fct = nn.CosineSimilarity(dim=-1)
        if not hasattr(model, "tokenizer") or not hasattr(guide, "tokenizer"):
            raise ValueError("Both the training model and the guiding model must have a tokenizer attribute.")
        if not isinstance(model.tokenizer, PreTrainedTokenizerBase) or not isinstance(
            guide.tokenizer, PreTrainedTokenizerBase
        ):
            raise ValueError(
                "Both the training model and the guiding model must use a PreTrainedTokenizer from transformers."
            )
        self.must_retokenize = (
            model.tokenizer.get_vocab() != guide.tokenizer.get_vocab() or guide.max_seq_length < model.max_seq_length
        )
        if self.must_retokenize:
            self.tokenizer = self.model.tokenizer

            if isinstance(self.model[0], StaticEmbedding):
                raise ValueError(
                    "If we must retokenize because the guide model has a different tokenizer, "
                    "then the Sentence Transformer model must not be based on a StaticEmbedding."
                )

        if margin_strategy not in ("absolute", "relative"):
            raise ValueError("margin_strategy must be 'absolute' or 'relative'.")
        self.margin_strategy = margin_strategy
        self.margin = margin
        self.contrast_anchors = contrast_anchors
        self.contrast_positives = contrast_positives
        self.gather_across_devices = gather_across_devices
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def sim_matrix(self, embed1: Tensor, embed2: Tensor) -> Tensor:
        return self.similarity_fct(embed1.unsqueeze(1), embed2.unsqueeze(0))

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        with torch.no_grad():
            if self.must_retokenize:
                decoded = [
                    self.tokenizer.batch_decode(sentence_feature["input_ids"], skip_special_tokens=True)
                    for sentence_feature in sentence_features
                ]
                sentence_features = [self.guide.tokenize(sentences) for sentences in decoded]
                sentence_features = [
                    {key: value.to(self.guide.device) for key, value in sentence_feature.items()}
                    for sentence_feature in sentence_features
                ]

            guide_embeddings = [
                self.guide(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features
            ]

        negative = None
        negative_guide = None

        if len(embeddings) == 2:
            anchor, positive = embeddings
            anchor_guide, positive_guide = guide_embeddings
        elif len(embeddings) == 3:
            anchor, positive, negative = embeddings
            anchor_guide, positive_guide, negative_guide = guide_embeddings
        else:
            raise ValueError(f"Expected 2 or 3 embeddings, got {len(embeddings)}")
        batch_size = anchor.size(0)
        offset = 0

        if self.gather_across_devices:
            # Gather the candidates across all devices, with gradients, but not the anchors. We compute only this
            # device's anchors with all candidates from all devices, such that the backward pass on the document
            # embeddings can flow back to the original devices.
            positive = all_gather_with_grad(positive)
            positive_guide = all_gather_with_grad(positive_guide)
            if negative is not None:
                negative = all_gather_with_grad(negative)
                negative_guide = all_gather_with_grad(negative_guide)
            # All have this shape: (batch_size * world_size * (1 + num_negatives), embedding_dim)

            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        # Compute the similarities given the training and guide embeddings.
        ap_sim = self.sim_matrix(anchor, positive)
        guided_ap_sim = self.sim_matrix(anchor_guide, positive_guide)

        # Define the anchor threshold
        guided_sim = guided_ap_sim.diagonal(offset=offset).view(-1, 1)

        # This uses guided (teacher) similarity as a dynamic threshold to identify and suppress false negatives
        def mask_false_negatives(guided_sim_mat, sim_mat, positive_mask: Tensor | None = None):
            if self.margin_strategy == "absolute":
                # Remove samples whose guided similarity is higher than (positive_sim - margin)
                mask = guided_sim_mat > (guided_sim - self.margin)
            elif self.margin_strategy == "relative":
                # Remove samples whose guided similarity is higher than (positive_sim * margin)
                mask = guided_sim_mat > (guided_sim * (1 - self.margin))

            if positive_mask is not None:
                # Ensure true positive pairs are not masked out
                mask = mask & ~positive_mask
            sim_mat[mask] = -torch.inf
            return sim_mat

        # Create a mask to protect true positive pairs in the anchor-positive matrix (i.e., diagonal elements)
        positive_mask = torch.eye(*guided_ap_sim.shape, dtype=torch.bool, device=guided_ap_sim.device)

        # Apply false negative suppression to each similarity matrix using guided similarity as anchor
        ap_sim = mask_false_negatives(guided_ap_sim, ap_sim, positive_mask=positive_mask)  # anchor-positive
        scores = [ap_sim]

        if self.contrast_anchors:
            aa_sim = self.sim_matrix(anchor, anchor)
            guided_aa_sim = self.sim_matrix(anchor_guide, anchor_guide)
            aa_sim = mask_false_negatives(guided_aa_sim, aa_sim)  # anchor-anchor
            scores.append(aa_sim)

        if self.contrast_positives:
            pp_sim = self.sim_matrix(positive[offset : offset + batch_size], positive)
            guided_pp_sim = self.sim_matrix(positive_guide[offset : offset + batch_size], positive_guide)
            pp_sim = mask_false_negatives(guided_pp_sim, pp_sim)  # positive-positive
            scores.append(pp_sim)

        # Handle the case where we have a negative sample
        if negative is not None:
            an_sim = self.sim_matrix(anchor, negative)
            guided_an_sim = self.sim_matrix(anchor_guide, negative_guide)
            an_sim = mask_false_negatives(guided_an_sim, an_sim)  # anchor-negative
            scores.append(an_sim)

        scores = torch.cat(scores, dim=1) / self.temperature

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i. This means that we can just use arange
        range_labels = torch.arange(offset, offset + batch_size, device=anchor.device)

        return self.cross_entropy_loss(scores, range_labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "contrast_anchors": self.contrast_anchors,
            "contrast_positives": self.contrast_positives,
            "gather_across_devices": self.gather_across_devices,
        }

    @property
    def citation(self) -> str:
        return """
@misc{solatorio2024gistembed,
    title={GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning},
    author={Aivin V. Solatorio},
    year={2024},
    eprint={2402.16829},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
