from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.base.losses.gradcache import (
    RandContext,
    _backward_hook,
    _create_minibatch,
    _minibatch_ranges,
    _validate_mini_batch_num_tokens,
    has_static_embedding_input,
)
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad, is_dist_initialized


class CachedGISTEmbedLoss(nn.Module):
    # Enables per-sample media counting in Transformer.preprocess for VLM minibatching
    requires_media_counts = True

    # Back-propagates from a hook on the returned loss (see `gradcache.uses_gradient_cache`).
    uses_gradient_cache = True

    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
        mini_batch_num_tokens: int | None = None,
        show_progress_bar: bool = False,
        margin_strategy: Literal["absolute", "relative"] = "absolute",
        margin: float = 0.0,
        contrast_anchors: bool = True,
        contrast_positives: bool = True,
        gather_across_devices: bool = False,
    ) -> None:
        """
        This loss is a combination of :class:`GISTEmbedLoss` and :class:`CachedMultipleNegativesRankingLoss`.
        Typically, :class:`MultipleNegativesRankingLoss` requires a larger batch size for better performance.
        :class:`GISTEmbedLoss` yields stronger training signals than :class:`MultipleNegativesRankingLoss` due to the
        use of a guide model for in-batch negative sample selection. Meanwhile, :class:`CachedMultipleNegativesRankingLoss`
        allows for scaling of the batch size by dividing the computation into two stages of embedding and loss
        calculation, which both can be scaled by mini-batches (https://huggingface.co/papers/2101.06983).

        By combining the guided selection from :class:`GISTEmbedLoss` and Gradient Cache from
        :class:`CachedMultipleNegativesRankingLoss`, it is possible to reduce memory usage while maintaining performance
        levels comparable to those of :class:`GISTEmbedLoss`.

        You can apply different false-negative filtering strategies to discard hard negatives that are too similar to
        the positive. Two strategies are supported:

            - "absolute": Discards negatives whose similarity score is greater than ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than ``positive_score - abs(positive_score) * margin``.

        Args:
            model: SentenceTransformer model
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the faster the training is, but the more memory is used. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            mini_batch_num_tokens: If set, the embedding mini-batches are packed by total (non-padding)
                token count instead of by ``mini_batch_size`` sequences, which speeds up training on
                variable-length data. Most effective for models that avoid padded compute, e.g. flash
                attention with input flattening. See the Speeding up Inference documentation for details.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.
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
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983
            - GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning https://huggingface.co/papers/2402.16829

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large batch sizes for superior performance, but has slower training time than :class:`MultipleNegativesRankingLoss`

        Inputs:
            +-------------------------------------------------+--------+
            | Inputs                                          | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.sentence_transformer.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Equivalent to :class:`GISTEmbedLoss`, but with caching that allows for much higher batch sizes

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                guide = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedGISTEmbedLoss(
                    model,
                    guide,
                    mini_batch_size=64,
                    margin_strategy="absolute",   # or "relative" (e.g., margin=0.05 for max. 95% of positive similarity)
                    margin=0.1
                )

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if has_static_embedding_input(model):
            raise ValueError(
                "CachedGISTEmbedLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using GISTEmbedLoss instead."
            )
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
        _validate_mini_batch_num_tokens(mini_batch_num_tokens)
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens
        self.show_progress_bar = show_progress_bar
        self.must_retokenize = (
            model.tokenizer.vocab != guide.tokenizer.vocab or guide.max_seq_length < model.max_seq_length
        )
        if self.must_retokenize:
            self.tokenizer = model.tokenizer
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

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, Tensor | None, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
            if with_grad:
                # The with-gradient pass is the backward hook's re-embedding, which only needs the
                # model's embeddings: the guide's were already used to compute the cached gradients.
                guide_reps = None
            else:
                with torch.no_grad():
                    if self.must_retokenize:
                        decoded = self.tokenizer.batch_decode(
                            sentence_feature_minibatch["input_ids"], skip_special_tokens=True
                        )
                        sentence_feature_minibatch = self.guide.preprocess(decoded)
                        sentence_feature_minibatch = {
                            key: value.to(self.guide.device) for key, value in sentence_feature_minibatch.items()
                        }
                    guide_reps = self.guide(sentence_feature_minibatch)["sentence_embedding"]

        return reps, guide_reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
        ranges: list[tuple[int, int]] | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        if ranges is None:
            ranges = _minibatch_ranges(sentence_feature, self.mini_batch_size, self.mini_batch_num_tokens)
        for i, (begin, end) in enumerate(
            tqdm.tqdm(
                ranges,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            reps, guide_reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, guide_reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(
        self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]]
    ) -> tuple[Tensor, list[list[Tensor]]]:
        """Calculate the cross-entropy loss and return it alongside the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, reps_guided, with_backward=True)
        loss = loss.detach().requires_grad_()
        cache = [[r.grad for r in rs] for rs in reps]
        return loss, cache

    def calculate_loss(
        self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]], *, with_backward: bool = False
    ) -> Tensor:
        """Generalized function to calculate the cross-entropy loss without caching gradients."""
        if len(reps) != len(reps_guided):
            raise ValueError("reps and reps_guided must have the same length")

        # Concatenate embeddings along the batch dimension
        anchors = torch.cat(reps[0])  # (batch_size, embedding_dim)
        anchors_guide = torch.cat(reps_guided[0])  # (batch_size, embedding_dim)
        candidates = [torch.cat(r) for r in reps[1:]]  # 1 + nneg items of (bsz, hdim)
        candidates_guide = [torch.cat(r) for r in reps_guided[1:]]  # 1 + nneg items of (bsz, hdim)

        batch_size = anchors.size(0)
        offset = 0

        if self.gather_across_devices:
            # Gather the candidates across all devices, with gradients, but not the anchors. We compute only this
            # device's anchors with all candidates from all devices, such that the backward pass on the document
            # embeddings can flow back to the original devices.
            candidates = [all_gather_with_grad(candidate) for candidate in candidates]
            candidates_guide = [all_gather_with_grad(candidate) for candidate in candidates_guide]
            # All have this shape: 1 + nneg items of (batch_size * world_size, embedding_dim)

            if is_dist_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i. This means that we can just use arange
        range_labels = torch.arange(offset, offset + batch_size, device=anchors.device)

        losses: list[torch.Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = begin + self.mini_batch_size

            # Compute the similarities given the training and guide embeddings.
            ap_sim = self.sim_matrix(anchors[begin:end], candidates[0])  # anchor-positive similarity
            guided_ap_sim = self.sim_matrix(anchors_guide[begin:end], candidates_guide[0])

            # Define the anchor threshold
            guided_sim = guided_ap_sim.diagonal(offset=offset + begin).view(-1, 1)

            # This uses guided (teacher) similarity as a dynamic threshold to identify and suppress false negatives
            def mask_false_negatives(guided_sim_mat, sim_mat, positive_mask: Tensor | None = None):
                if self.margin_strategy == "absolute":
                    # Remove samples whose guided similarity is higher than (positive_sim - margin)
                    mask = guided_sim_mat > (guided_sim - self.margin)
                elif self.margin_strategy == "relative":
                    # Remove samples whose guided similarity is higher than (positive_sim - |positive_sim| * margin)
                    mask = guided_sim_mat > (guided_sim - guided_sim.abs() * self.margin)

                if positive_mask is not None:
                    # Ensure true positive pairs are not masked out
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            # Protect each anchor's true positive from false-negative suppression using the same
            # gathered column index as the CE target.
            positive_mask = torch.zeros_like(guided_ap_sim, dtype=torch.bool)
            rows = torch.arange(guided_ap_sim.size(0), device=guided_ap_sim.device)
            positive_mask[rows, offset + begin + rows] = True

            # Apply false negative suppression to each similarity matrix using guided similarity as anchor
            ap_sim = mask_false_negatives(guided_ap_sim, ap_sim, positive_mask=positive_mask)  # anchor-positive
            scores = [ap_sim]

            if self.contrast_anchors:
                aa_sim = self.sim_matrix(anchors[begin:end], anchors)
                guided_aa_sim = self.sim_matrix(anchors_guide[begin:end], anchors_guide)
                aa_sim = mask_false_negatives(guided_aa_sim, aa_sim)  # anchor-anchor
                scores.append(aa_sim)

            if self.contrast_positives:
                pp_sim = self.sim_matrix(
                    candidates[0][offset + begin : min(offset + end, offset + batch_size)], candidates[0]
                )
                guided_pp_sim = self.sim_matrix(
                    candidates_guide[0][offset + begin : min(offset + end, offset + batch_size)], candidates_guide[0]
                )
                pp_sim = mask_false_negatives(guided_pp_sim, pp_sim)  # positive-positive
                scores.append(pp_sim)

            # If there are negatives (len(candidates) > 1), process them
            if len(candidates) > 1:
                for i in range(1, len(candidates)):  # Start from 1 since the first is the positive
                    neg_sim = self.sim_matrix(anchors[begin:end], candidates[i])
                    guided_neg_sim = self.sim_matrix(anchors_guide[begin:end], candidates_guide[i])
                    neg_sim = mask_false_negatives(guided_neg_sim, neg_sim)
                    scores.append(neg_sim)  # anchor-negative

            # Concatenate all scores into a single tensor
            scores = torch.cat(scores, dim=1)  # (mbsz, num_scores)

            # Normalize the scores and calculate the cross-entropy loss
            scores = scores / self.temperature
            loss_mbatch: torch.Tensor = (
                self.cross_entropy_loss(scores, range_labels[begin:end]) * len(scores) / batch_size
            )
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        sentence_features = list(sentence_features)
        grad_enabled = torch.is_grad_enabled()

        # Compute the mini-batch boundaries before any forward pass: modules may modify the features
        # in place while embedding, and step (3) must replay exactly the boundaries step (1) used.
        ranges = [
            _minibatch_ranges(sentence_feature, self.mini_batch_size, self.mini_batch_num_tokens)
            for sentence_feature in sentence_features
        ]

        reps = []
        reps_guided = []
        # Copy random states to guarantee exact reproduction of the embeddings during the second forward
        # pass, i.e. step (3). Only the backward hook replays them, so skip them during evaluation.
        random_states = []
        for sentence_feature, column_ranges in zip(sentence_features, ranges):
            reps_mbs = []
            reps_guided_mbs = []
            random_state_mbs = []
            for reps_mb, reps_guided_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=grad_enabled,
                ranges=column_ranges,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                reps_guided_mbs.append(reps_guided_mb.detach())  # does not requires gradient
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            reps_guided.append(reps_guided_mbs)
            random_states.append(random_state_mbs)

        if not grad_enabled:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            return self.calculate_loss(reps, reps_guided)

        # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
        loss, cache = self.calculate_loss_and_cache_gradients(reps, reps_guided)

        # Step (3): a 2nd embedding step with gradients, connecting the cached gradients into
        # the backward chain. The hook gets this pass's cache, so another forward cannot clobber it.
        loss.register_hook(
            partial(
                _backward_hook,
                sentence_features=sentence_features,
                loss_obj=self,
                cache=cache,
                random_states=random_states,
                ranges=ranges,
            )
        )
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
            "mini_batch_size": self.mini_batch_size,
            "mini_batch_num_tokens": self.mini_batch_num_tokens,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "contrast_anchors": self.contrast_anchors,
            "contrast_positives": self.contrast_positives,
            "gather_across_devices": self.gather_across_devices,
        }
