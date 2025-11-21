from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss


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
    pairs: list[list[str]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        # for sentence_feature, grad, random_states in zip(pairs, loss_obj.cache, loss_obj.random_states):
        for (minibatch_logits, _), minibatch_grad in zip(
            loss_obj.predict_minibatch_iter(
                pairs=pairs,
                with_grad=True,
                copy_random_state=False,
                random_states=loss_obj.random_states,
            ),
            loss_obj.cache,
        ):
            surrogate = torch.dot(minibatch_logits.flatten(), minibatch_grad.flatten()) * grad_output
            surrogate.backward()


class CachedMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
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
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
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
            | Texts                                           | Labels | Number of Model Output Labels |
            +=================================================+========+===============================+
            | (anchor, positive) pairs                        | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative) triplets           | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+
            | (anchor, positive, negative_1, ..., negative_n) | none   | 1                             |
            +-------------------------------------------------+--------+-------------------------------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.
            - Use :class:`~sentence_transformers.util.mine_hard_negatives` with ``output_format="n-tuple"`` or
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
        super().__init__(model, num_negatives, scale, activation_fn)
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

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
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        with random_state_context:
            with grad_context():
                random_state = RandContext(pairs) if copy_random_state else None
                logits = self.call_model_with_pairs(pairs)
        return logits, random_state

    def predict_minibatch_iter(
        self,
        pairs: list[list[str]],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
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
            )
            yield logits, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, logits: list[Tensor], batch_size: int) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(logits, batch_size)
        loss.backward()
        loss = loss.detach().requires_grad_()

        self.cache = [logit.grad for logit in logits]

        return loss

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
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

        logits = []
        self.random_states = []
        for minibatch_logits, random_state in self.predict_minibatch_iter(
            pairs=pairs,
            with_grad=False,
            copy_random_state=True,
        ):
            logits.append(minibatch_logits.detach().requires_grad_())
            self.random_states.append(random_state)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(logits, batch_size)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, pairs=pairs, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(logits, batch_size)

        return loss

    def get_config_dict(self):
        return {**super().get_config_dict(), "mini_batch_size": self.mini_batch_size}
