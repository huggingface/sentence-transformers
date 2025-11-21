from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states
from transformers import PreTrainedTokenizerBase

from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedGISTEmbedLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # TODO: This if-statement is for if the model does not require gradients, which may happen if the model
                # contains a Router where one of the routes is frozen. It should be possible to not have to call
                # embed_minibatch_iter in that case, as it's unnecessarily expensive.
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedGISTEmbedLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        guide: SentenceTransformer,
        temperature: float = 0.01,
        mini_batch_size: int = 32,
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

            - "absolute": Discards negatives whose similarity score is greater than or equal to ``positive_score - margin``.
            - "relative": Discards negatives whose similarity score is greater than or equal to ``positive_score * (1 - margin)``.

        Args:
            model: SentenceTransformer model
            guide: SentenceTransformer model to guide the in-batch negative sample selection.
            temperature: Temperature parameter to scale the cosine similarities.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
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
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Equivalent to :class:`GISTEmbedLoss`, but with caching that allows for much higher batch sizes

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
        if isinstance(model[0], StaticEmbedding):
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
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
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
    ) -> tuple[Tensor, Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
            with torch.no_grad():
                if self.must_retokenize:
                    decoded = self.tokenizer.batch_decode(
                        sentence_feature_minibatch["input_ids"], skip_special_tokens=True
                    )
                    sentence_feature_minibatch = self.guide.tokenize(decoded)
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
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, begin in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, guide_reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, guide_reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]]) -> Tensor:
        """Generalized function to calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, reps_guided, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(
        self, reps: list[list[Tensor]], reps_guided: list[list[Tensor]], with_backward: bool = False
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

            if torch.distributed.is_initialized():
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
                    # Remove samples whose guided similarity is higher than (positive_sim * margin)
                    mask = guided_sim_mat > (guided_sim * (1 - self.margin))

                if positive_mask is not None:
                    # Ensure true positive pairs are not masked out
                    mask = mask & ~positive_mask
                sim_mat[mask] = -torch.inf
                return sim_mat

            # Create a mask to protect true positive pairs in the anchor-positive matrix (i.e., diagonal elements)
            positive_mask = torch.eye(*guided_ap_sim.shape, dtype=torch.bool, device=guided_ap_sim.device)
            positive_mask = positive_mask.roll(begin)

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
        reps = []
        reps_guided = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
        for sentence_feature in sentence_features:
            reps_mbs = []
            reps_guided_mbs = []
            random_state_mbs = []
            for reps_mb, reps_guided_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                reps_guided_mbs.append(reps_guided_mb.detach())  # does not requires gradient
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            reps_guided.append(reps_guided_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps, reps_guided)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps, reps_guided)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "guide": self.guide,
            "temperature": self.temperature,
            "mini_batch_size": self.mini_batch_size,
            "margin_strategy": self.margin_strategy,
            "margin": self.margin,
            "contrast_anchors": self.contrast_anchors,
            "contrast_positives": self.contrast_positives,
            "gather_across_devices": self.gather_across_devices,
        }
