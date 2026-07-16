from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import RandContext, _backward_hook
from sentence_transformers.cross_encoder.losses.multiple_negatives_ranking import MultipleNegativesRankingLoss
from sentence_transformers.cross_encoder.model import CrossEncoder
from sentence_transformers.util import batch_to_device


class CachedMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    # Back-propagates from a hook on the returned loss (see `gradcache.uses_gradient_cache`).
    uses_gradient_cache = True

    def __init__(
        self,
        model: CrossEncoder,
        num_negatives: int | None = 4,
        scale: float = 10.0,
        activation_fn: nn.Module | None = nn.Sigmoid(),
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss` that
        caches the gradients of the logits wrt. the loss. This allows for much higher batch sizes without extra
        memory usage. However, it is slightly slower.

        In detail:

            (1) It first does a quick prediction step without gradients/computation graphs to get all the logits;
            (2) Calculate the loss, backward up to the logits and cache the gradients wrt. to the logits;
            (3) A 2nd prediction step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in
        mini-batches and requires a lot memory when the batch size is large. The gradient caching will sacrifice around
        20% computation time according to the paper.

        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

        * Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer)
          out of every single positive and negative (e.g. all answers) in the batch.

        If you provide the optional negatives, they will all be used as extra options from which the model must pick the
        correct positive. Within reason, the harder this "picking" is, the stronger the model will become. Because of
        this, a higher batch size results in more in-batch negatives, which then increases performance (to a point).

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, answer)) as it will sample in each batch ``n-1`` negative docs randomly.

        This loss is also known as InfoNCE loss with GradCache.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`): A CrossEncoder model to be trained.
            num_negatives (int, optional): Number of in-batch negatives to sample for each anchor. Defaults to 4.
            scale (int, optional): Output of similarity function is multiplied by scale value. Defaults to 10.0.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss. Defaults to :class:`~torch.nn.Sigmoid`.
            mini_batch_size (int, optional): Mini-batch size for the forward pass. This informs the memory usage. Defaults to 32.
            show_progress_bar (bool, optional): Whether to show a progress bar during the forward pass. Defaults to False.

        .. note::

            The current default values are subject to change in the future. Experimentation is encouraged.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_
            - `Cross Encoder > Training Examples > Rerankers <../../../examples/cross_encoder/training/rerankers/README.html>`_

        Requirements:
            1. Your model must be initialized with `num_labels = 1` (a.k.a. the default) to predict one class.
            2. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance,
               but slower training time than :class:`MultipleNegativesRankingLoss`.

        Inputs:
            +-------------------------------------------------+--------+-------------------------------+
            | Inputs                                          | Labels | Number of Model Output Labels |
            +=================================================+========+===============================+
            | (anchor, positive) pairs                        | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative) triplets           | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative_1, ..., negative_n) | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.sentence_transformer.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.
            - Use :class:`~sentence_transformers.util.hard_negatives.mine_hard_negatives` with ``output_format="n-tuple"`` or
              ``output_format="triplet"`` to convert question-answer pairs to triplets with hard negatives.

        Relations:
            - Equivalent to :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss`, but with
              caching that allows for much higher batch sizes (and thus better performance) without extra memory usage.
              This loss also trains slower than :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss`.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "answer": ["Pandas are a kind of bear.", "The capital of France is Paris."],
                })
                loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(model, num_negatives=num_negatives, scale=scale, activation_fn=activation_fn)
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def predict_minibatch(
        self,
        pairs: list[list[str]],
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
        prompt: str | None = None,
        task: str | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding logits.

        Pairs are tokenized and moved to the device before the RNG snapshot: RandContext only captures
        state for devices it sees tensors for. Snapshotting the raw strings caught no device state, so on
        CUDA the backward re-prediction drew different dropout masks than the cached gradients belonged to.
        """
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        features = self.model.preprocess(pairs, prompt=prompt, task=task)
        features = batch_to_device(features, self.model.device)
        with random_state_context:
            with grad_context():
                random_state = RandContext(*features.values()) if copy_random_state else None
                logits = self.model(features)["scores"].squeeze(1)
        return logits, random_state

    def predict_minibatch_iter(
        self,
        pairs: list[list[str]],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
        prompt: str | None = None,
        task: str | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        for i, b in enumerate(
            tqdm.trange(
                0,
                len(pairs),
                self.mini_batch_size,
                desc="Predict mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            mini_batch_pairs = pairs[b:e]

            logits, random_state = self.predict_minibatch(
                pairs=mini_batch_pairs,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
                prompt=prompt,
                task=task,
            )
            yield logits, random_state  # logits: (mbsz,)

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Any],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
        ranges: list[tuple[int, int]] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Adapter for the shared backward hook, which iterates per-column features. This loss has one
        "column": the flat pair list, bundled with the prompt/task it must be tokenized with. The
        ``ranges`` are unused: this loss slices its pairs by count (token budgets would require
        tokenizing the whole pair list up front, a follow-up)."""
        yield from self.predict_minibatch_iter(
            pairs=sentence_feature["pairs"],
            with_grad=with_grad,
            copy_random_state=copy_random_state,
            random_states=random_states,
            prompt=sentence_feature["prompt"],
            task=sentence_feature["task"],
        )

    def calculate_loss_and_cache_gradients(self, logits: list[Tensor], batch_size: int) -> tuple[Tensor, list[Tensor]]:
        """Calculate the cross-entropy loss and return it alongside the gradients wrt. the logits."""
        loss = self.calculate_loss(logits, batch_size)
        loss.backward()
        loss = loss.detach().requires_grad_()
        cache = [logit.grad for logit in logits]
        return loss, cache

    def forward(
        self, inputs: list[list[str]], labels: Tensor | None = None, prompt: str | None = None, task: str | None = None
    ) -> Tensor:
        # Step (1): A quick prediction step without gradients/computation graphs to get all the logits
        anchors = inputs[0][::]
        candidates = inputs[1][::]
        batch_size = len(anchors)

        # In-batch negatives:
        for negatives in self.get_in_batch_negatives(inputs[0], inputs[1:]):
            anchors.extend(inputs[0])
            candidates.extend(negatives)

        # Hard negatives:
        for negatives in inputs[2:]:
            anchors.extend(inputs[0])
            candidates.extend(negatives)

        pairs = list(zip(anchors, candidates))
        grad_enabled = torch.is_grad_enabled()

        logits = []
        random_states = []
        for minibatch_logits, random_state in self.predict_minibatch_iter(
            pairs=pairs,
            with_grad=False,
            # Only the backward hook replays them, so don't pay for RNG snapshots during evaluation.
            copy_random_state=grad_enabled,
            prompt=prompt,
            task=task,
        ):
            logits.append(minibatch_logits.detach().requires_grad_())
            random_states.append(random_state)

        if not grad_enabled:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            return self.calculate_loss(logits, batch_size)

        # Step (2): Calculate the loss, backward up to the logits and cache the gradients wrt. to the logits
        loss, cache = self.calculate_loss_and_cache_gradients(logits, batch_size)

        # Step (3): a 2nd prediction step with gradients, connecting the cached gradients into
        # the backward chain. The hook gets this pass's cache, so another forward cannot clobber it.
        loss.register_hook(
            partial(
                _backward_hook,
                sentence_features=[{"pairs": pairs, "prompt": prompt, "task": task}],
                loss_obj=self,
                cache=[cache],
                random_states=[random_states],
                ranges=[None],
            )
        )
        return loss

    def get_config_dict(self):
        return {**super().get_config_dict(), "mini_batch_size": self.mini_batch_size}
