from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.base.losses.gradcache import (
    CachedLossMixin,
    _validate_mini_batch_num_tokens,
    has_static_embedding_input,
)
from sentence_transformers.sentence_transformer.model import SentenceTransformer


class MegaBatchMarginLoss(CachedLossMixin, nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        positive_margin: float = 0.8,
        negative_margin: float = 0.3,
        use_mini_batched_version: bool = True,
        mini_batch_size: int = 50,
        mini_batch_num_tokens: int | None = None,
        show_progress_bar: bool = False,
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
                of memory, we can use a mini-batched version that only
                keeps one mini-batch of activations alive at a time
                (GradCache, https://huggingface.co/papers/2101.06983).
                This does not change the loss or the gradient, but it
                embeds every input twice, so it trades roughly a 2x
                slowdown for the memory saving. It is not compatible
                with a model based on a :class:`~sentence_transformers.sentence_transformer.modules.StaticEmbedding`,
                whose inputs have no batch dimension to split along.
            mini_batch_size: Size for the mini-batches. It does not have
                to divide the batch size in your data loader. It bounds
                both the activation memory of the embedding steps and the
                size of the similarity matrix used to mine the hardest
                negatives, so lower it if you run out of memory.
            mini_batch_num_tokens: If set, the embedding mini-batches are packed by total (non-padding)
                token count instead of by ``mini_batch_size`` sequences, which speeds up training on
                variable-length data. Most effective for models that avoid padded compute, e.g. flash
                attention with input flattening. See the Speeding up Inference documentation for details.
            show_progress_bar: If True, a progress bar for the
                mini-batches is shown during training. The default is
                False.

        References:
            - This loss function was inspired by the ParaNMT paper: https://www.aclweb.org/anthology/P18-1042/

        Requirements:
            1. (anchor, positive) pairs
            2. Large batches (500 or more examples)

        Inputs:
            +---------------------------------------+--------+
            | Inputs                                | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.sentence_transformer.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - The hardest negative is mined from the other positives in the batch, so the batch size directly
              determines the quality of the negatives. ``use_mini_batched_version=True`` is what lets you raise it.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, losses
                from datasets import Dataset

                train_batch_size = 250
                train_mini_batch_size = 32

                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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
        if use_mini_batched_version and has_static_embedding_input(model):
            raise ValueError(
                "The mini-batched MegaBatchMarginLoss is not compatible with a SentenceTransformer model based on a "
                "StaticEmbedding, whose inputs cannot be split into mini-batches along a batch dimension. "
                "Consider passing use_mini_batched_version=False instead."
            )

        self.model = model
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        _validate_mini_batch_num_tokens(mini_batch_num_tokens)
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens
        self.show_progress_bar = show_progress_bar
        # Both only apply to the mini-batched path: without it there is nothing to slice, and the
        # backward pass runs straight from the returned loss rather than from a hook.
        self.requires_media_counts = use_mini_batched_version
        self.uses_gradient_cache = use_mini_batched_version
        self.forward = self.forward_mini_batched if use_mini_batched_version else self.forward_non_mini_batched

    @staticmethod
    def _validate_input(sentence_features: list[dict[str, Tensor]]) -> None:
        if len(sentence_features) != 2:
            raise ValueError(
                f"MegaBatchMarginLoss expects exactly 2 input columns, (anchor, positive), but got "
                f"{len(sentence_features)}. It mines the hardest negative for each anchor from the other "
                "positives in the batch, so it has no use for an explicit negative column. Consider "
                "CachedMultipleNegativesRankingLoss if you want to train with explicit negatives."
            )

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the margin loss over the whole batch, from the per-mini-batch embeddings.

        The labels are unused: the negatives are mined from the other positives in the batch. For each
        anchor, the hardest negative is the most similar positive belonging to a *different* anchor, so
        this needs every (anchor, positive) similarity. Only ``mini_batch_size`` rows of that matrix are
        materialized at a time.
        """
        anchors = torch.cat(reps[0])
        positives = torch.cat(reps[1])
        batch_size = len(anchors)

        # Chunking only saves memory when each chunk's graph is freed as we go: back-propagated
        # here, or never built. Otherwise every chunk keeps its own normalized copy of `positives`.
        chunk_size = self.mini_batch_size if with_backward or not torch.is_grad_enabled() else batch_size

        losses = []
        for begin in range(0, batch_size, chunk_size):
            end = min(begin + chunk_size, batch_size)
            rows = torch.arange(end - begin, device=anchors.device)
            diagonal = rows + begin

            # (chunk_size, batch_size): this anchor chunk against every positive in the batch
            cos_scores = util.pytorch_cos_sim(anchors[begin:end], positives)
            positive_scores = cos_scores[rows, diagonal]

            # Drop the positive scores off the diagonal, so that max() picks the hardest negative
            # rather than the anchor's own positive.
            negative_scores = cos_scores.clone()
            negative_scores[rows, diagonal] = -torch.inf
            negatives_max, _ = torch.max(negative_scores, dim=1)

            chunk_loss = (
                F.relu(self.positive_margin - positive_scores) + F.relu(negatives_max - self.negative_margin)
            ).sum() / batch_size
            if with_backward:
                chunk_loss.backward()
                chunk_loss = chunk_loss.detach()
            losses.append(chunk_loss)

        return sum(losses)

    def forward_mini_batched(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        self._validate_input(sentence_features)
        return self.forward_cached(sentence_features, labels)

    ##### Non mini-batched version ###
    def forward_non_mini_batched(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        self._validate_input(sentence_features)
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.calculate_loss([[rep] for rep in reps])

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "positive_margin": self.positive_margin,
            "negative_margin": self.negative_margin,
            "mini_batch_size": self.mini_batch_size,
            "mini_batch_num_tokens": self.mini_batch_num_tokens,
        }

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
