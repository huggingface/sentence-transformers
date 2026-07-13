"""Shared GradCache machinery (https://huggingface.co/papers/2101.06983).

A "cached" loss trades compute for memory in three steps:

    (1) a quick embedding step without gradients/computation graphs, to get all the embeddings;
    (2) calculate the loss, backward up to the embeddings, and cache the gradients wrt. the embeddings;
    (3) a 2nd embedding step with gradients/computation graphs, connecting the cached gradients into
        the backward chain.

Only one mini-batch of activations is alive at a time in (1) and (3), which is what bounds the memory.
The second forward pass reproduces the first exactly by replaying its RNG state (:class:`RandContext`),
so dropout draws the same masks and the cached gradients belong to the embeddings they are applied to.

:class:`CachedLossMixin` implements all of this; a loss only has to provide ``calculate_loss``.
:class:`GradCacheLoss` is the generic consumer: it wraps any loss that can compute itself from
precomputed embeddings (``compute_loss_from_embeddings``) and adds gradient caching to it.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Protocol

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        # torch.utils.checkpoint.get_device_states() fails when it sees MPS tensors (it
        # calls the non-existent torch.mps.device()), so capture the MPS RNG state for
        # top-level MPS tensor arguments and filter them out before calling it. The MPS
        # state is restored in __enter__ so the cached second forward replays the same
        # randomness (e.g. dropout).
        self.fwd_mps_state = (
            torch.mps.get_rng_state()
            if any(isinstance(t, torch.Tensor) and t.device.type == "mps" for t in tensors)
            else None
        )
        non_mps_tensors = tuple(t for t in tensors if not (isinstance(t, torch.Tensor) and t.device.type == "mps"))
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*non_mps_tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        if self.fwd_mps_state is not None:
            # This fork_rng call uses the default device_type="cuda", so save the outer
            # MPS state here and restore it in __exit__ (mirroring fork_rng for CPU/CUDA).
            self._mps_state_outside = torch.mps.get_rng_state()
            torch.mps.set_rng_state(self.fwd_mps_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.fwd_mps_state is not None:
            torch.mps.set_rng_state(self._mps_state_outside)
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _get_batch_size(sentence_feature: dict[str, Any]) -> int:
    """Get the number of samples in sentence features, handling both padded and flattened inputs.

    With padded inputs, the batch size is the first dimension of any tensor.
    With flattened inputs (from ``DataCollatorWithFlattening``), the batch size is derived
    from ``cu_seq_lens_q`` which has shape ``(num_seqs + 1,)``.
    """
    if "cu_seq_lens_q" in sentence_feature:
        return len(sentence_feature["cu_seq_lens_q"]) - 1
    # Prefer known batch-indexed keys to avoid accidentally using flattened tensors
    # like pixel_values whose first dimension may differ from the batch size in
    # vision-language models (e.g. Qwen2-VL).
    for key in ("input_ids", "attention_mask"):
        if key in sentence_feature and isinstance(sentence_feature[key], torch.Tensor):
            return sentence_feature[key].shape[0]
    return next(
        value.shape[0] for value in sentence_feature.values() if isinstance(value, torch.Tensor) and value.ndim > 0
    )


def _create_minibatch(sentence_feature: dict[str, Any], begin: int, end: int) -> dict[str, Any]:
    """Create a mini-batch from sentence features, handling padded, flattened, and VLM inputs.

    With padded inputs, this simply slices tensors along the batch dimension.
    With flattened inputs (from ``DataCollatorWithFlattening``), this extracts the token ranges
    for sequences ``begin:end`` and rebuilds the metadata (``cu_seq_lens_q``, ``seq_idx``, etc.).

    VLMs like Qwen2-VL flatten per-sample visual tokens into a single tensor
    (e.g. ``pixel_values`` shape ``(total_visual_tokens, hidden_dim)``) with a grid tensor
    (e.g. ``image_grid_thw`` shape ``(num_items, 3)``) whose per-row product gives the token
    count per item.  ``num_images_per_sample`` / ``num_videos_per_sample`` (precomputed by
    ``Transformer.preprocess``) map grid rows to samples; when unavailable we fall back to
    assuming one grid row per sample when ``grid.shape[0] == batch_size``.
    """
    if "cu_seq_lens_q" not in sentence_feature:
        batch_size = _get_batch_size(sentence_feature)
        end = min(end, batch_size)

        custom_ranges: dict[str, tuple[int, int]] = {}
        for grid_key, pixel_key, count_key in (
            ("image_grid_thw", "pixel_values", "num_images_per_sample"),
            ("video_grid_thw", "pixel_values_videos", "num_videos_per_sample"),
        ):
            grid = sentence_feature.get(grid_key)
            pixel_values = sentence_feature.get(pixel_key)
            if grid is None or pixel_values is None:
                continue

            num_per_sample = sentence_feature.get(count_key)
            if num_per_sample is not None:
                cumsum_items = num_per_sample.cumsum(dim=0)
                grid_begin = 0 if begin == 0 else int(cumsum_items[begin - 1].item())
                grid_end = int(cumsum_items[end - 1].item())
                custom_ranges[grid_key] = (grid_begin, grid_end)
            elif grid.shape[0] == batch_size:
                grid_begin, grid_end = begin, end
            else:
                continue

            if grid_begin < grid_end:
                tokens_per_item = grid.prod(dim=1)
                token_cumsum = tokens_per_item.cumsum(dim=0)
                token_begin = 0 if grid_begin == 0 else int(token_cumsum[grid_begin - 1].item())
                token_end = int(token_cumsum[grid_end - 1].item())
            else:
                token_begin, token_end = 0, 0
            custom_ranges[pixel_key] = (token_begin, token_end)

        result: dict[str, Any] = {}
        for key, value in sentence_feature.items():
            if not isinstance(value, torch.Tensor):
                result[key] = value
            elif key in custom_ranges:
                r_begin, r_end = custom_ranges[key]
                result[key] = value[r_begin:r_end]
            else:
                result[key] = value[begin:end]
        return result

    cu_seq_lens_q = sentence_feature["cu_seq_lens_q"]
    num_seqs = len(cu_seq_lens_q) - 1
    end = min(end, num_seqs)

    token_begin = int(cu_seq_lens_q[begin].item())
    token_end = int(cu_seq_lens_q[end].item())
    total_tokens = int(cu_seq_lens_q[-1].item())

    new_cu_seq_lens = cu_seq_lens_q[begin : end + 1] - cu_seq_lens_q[begin]

    result: dict[str, Any] = {}
    for key, value in sentence_feature.items():
        if key in ("cu_seq_lens_q", "cu_seq_lens_k"):
            result[key] = new_cu_seq_lens
        elif key in ("max_length_q", "max_length_k"):
            mb_seq_lens = new_cu_seq_lens[1:] - new_cu_seq_lens[:-1]
            result[key] = int(mb_seq_lens.max().item())
        elif key == "seq_idx":
            result[key] = value[..., token_begin:token_end] - begin
        elif isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[-1] == total_tokens:
            # Heuristic: tensors whose last dimension matches the total token count are assumed
            # to be token-level (e.g. input_ids, position_ids). This covers all known keys from
            # DataCollatorWithFlattening without hard-coding them.
            result[key] = value[..., token_begin:token_end]
        else:
            result[key] = value
    return result


def uses_gradient_cache(loss: Any) -> bool:
    """Whether ``loss`` defers its backward pass to a hook on the loss tensor it returns.

    Such a loss re-embeds each mini-batch during the *backward* pass, by which time a decorator that
    patched ``SentenceTransformer.forward`` for the duration of the forward pass has been removed
    again. ``MatryoshkaLoss`` and ``AdaptiveLayerLoss`` both work by patching that forward, so they
    have to treat these losses specially -- MatryoshkaLoss by decorating ``calculate_loss`` instead,
    AdaptiveLayerLoss by warning that the combination is unsupported.

    Losses report this by setting ``uses_gradient_cache``; :class:`CachedLossMixin` sets it to True,
    and a loss that can turn the caching off at construction time (``MegaBatchMarginLoss``) overrides
    it per instance.
    """
    return getattr(loss, "uses_gradient_cache", False)


def has_static_embedding_input(model: Any) -> bool:
    """Whether the model embeds its inputs with a StaticEmbedding, directly or behind a Router.

    StaticEmbedding features are an EmbeddingBag (``input_ids``, ``offsets``) with no batch dimension,
    so they cannot be sliced into mini-batches; losses that mini-batch must reject such models.
    """
    from sentence_transformers.sentence_transformer.modules import Router, StaticEmbedding

    # A Router keeps its input modules one level down, which is where a StaticEmbedding would sit.
    input_modules = (
        [route[0] for route in model[0].sub_modules.values()] if isinstance(model[0], Router) else [model[0]]
    )
    return any(isinstance(module, StaticEmbedding) for module in input_modules)


class CachedLoss(Protocol):
    """The structural contract :func:`_backward_hook` relies on: a loss that can re-embed a
    mini-batch with gradients, replaying the RNG state of the first forward pass.

    :class:`CachedLossMixin` is the standard implementation, but ``CachedSpladeLoss`` also
    satisfies this protocol with its own. Implementations may yield extra elements after the
    embeddings (e.g. ``CachedGISTEmbedLoss`` yields the guide model's embeddings too); the
    backward hook only reads the first element.
    """

    mini_batch_size: int

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]: ...


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedLoss,
    cache: list[list[Tensor]],
    random_states: list[list[RandContext]],
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch.

    ``cache`` and ``random_states`` belong to one specific forward pass and are passed in rather than
    read off ``loss_obj``, so that a second forward pass before the first backward pass cannot make
    this hook back-propagate the wrong batch's gradients.

    Every mini-batch is scaled by ``grad_output``, which is whatever the outer backward pass hands us
    -- so the fp16 gradient scaler and the gradient accumulation division reach all of them.
    """
    with torch.enable_grad():
        for sentence_feature, grad, random_state in zip(sentence_features, cache, random_states):
            for (reps_mb, *_), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_state,
                ),
                grad,
            ):
                if not reps_mb.requires_grad:
                    # e.g. a frozen Router route: skip remaining minibatches as none need backprop
                    break
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()


class CachedLossMixin:
    """The GradCache forward pass, shared by the losses that cache the gradients wrt. their embeddings.

    Subclasses must be an ``nn.Module`` holding a ``model``, must set ``mini_batch_size``, and must
    implement :meth:`calculate_loss`. They then call :meth:`forward_cached` from their ``forward``.
    """

    model: Any
    mini_batch_size: int
    show_progress_bar: bool = False

    # Enables per-sample media counting in Transformer.preprocess, so that _create_minibatch can
    # slice VLM inputs (e.g. Qwen2-VL's flattened pixel_values) along the batch dimension.
    requires_media_counts = True

    # See `uses_gradient_cache`. A subclass that can turn the caching off overrides this per instance.
    uses_gradient_cache: bool = True

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the loss over the whole batch, from the per-mini-batch embeddings.

        When ``with_backward`` is set, back-propagate the loss (chunk by chunk, if the implementation
        chunks it) and return the detached total, so that no part of the loss graph outlives its own
        backward pass. Losses that don't need the labels simply ignore them.
        """
        raise NotImplementedError

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of inputs."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mini_batch_size, dim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do a forward pass on every mini-batch of the input features and yield the embeddings."""
        batch_size = _get_batch_size(sentence_feature)
        for i, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            yield self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=begin + self.mini_batch_size,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )

    def calculate_loss_and_cache_gradients(
        self, reps: list[list[Tensor]], labels: Tensor | None = None
    ) -> tuple[Tensor, list[list[Tensor]]]:
        """Compute the loss and return it alongside the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, labels, with_backward=True)
        loss = loss.detach().requires_grad_()
        cache = [[rep.grad for rep in rep_mbs] for rep_mbs in reps]
        return loss, cache

    def forward_cached(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        """Run the three-step GradCache forward pass. See the module docstring."""
        sentence_features = list(sentence_features)
        grad_enabled = torch.is_grad_enabled()

        # Step (1): embed every mini-batch without gradients, keeping the RNG state of each forward
        # pass so that step (3) can reproduce it exactly.
        reps = []
        random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                # Only the backward hook replays them, and it is only registered when gradients are
                # enabled, so don't pay for the RNG snapshots during evaluation.
                copy_random_state=grad_enabled,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            random_states.append(random_state_mbs)

        if not grad_enabled:
            # In evaluation there are no gradients to cache and no backward pass to hook into.
            return self.calculate_loss(reps, labels)

        # Step (2): compute the loss over the whole batch and cache the gradients wrt. the embeddings.
        loss, cache = self.calculate_loss_and_cache_gradients(reps, labels)

        # Step (3): re-embed each mini-batch with gradients and connect the cached gradients into the
        # backward chain. The cache is handed to the hook rather than stored on `self`, so that it
        # cannot be clobbered by another forward pass, and so that it is not held across the
        # optimizer step.
        loss.register_hook(
            partial(
                _backward_hook,
                sentence_features=sentence_features,
                loss_obj=self,
                cache=cache,
                random_states=random_states,
            )
        )
        return loss


def reconstruct_loss_components(total: Tensor, components: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rebuild a per-component loss dict around a single gradient-carrying total.

    The trainer sums a dict-valued loss for its backward pass, but after gradient caching only
    ``total`` carries the gradient. Exactly one entry must therefore hold it, so the first component
    is adjusted such that the dict still sums exactly to ``total``; the rest are detached values that
    only serve the per-component logging.
    """
    components = {key: value.detach() for key, value in components.items()}
    first = next(iter(components))
    others = sum((value for key, value in components.items() if key != first), start=torch.zeros_like(total))
    components[first] = total - others
    return components


class GradCacheLoss(CachedLossMixin, nn.Module):
    def __init__(
        self,
        model: Any,
        loss: nn.Module,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Adds GradCache (https://huggingface.co/papers/2101.06983) to another loss, so that it can be
        trained with much larger batch sizes without additional memory usage.

        Losses with in-batch interactions (e.g. in-batch negatives) improve with the batch size, but the
        batch cannot simply be split into independently back-propagated mini-batches, as the samples
        interact inside the loss. Gradient caching solves this in three steps:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Only one mini-batch of model activations is alive at a time, so ``mini_batch_size`` bounds the
        memory usage while ``per_device_train_batch_size`` sets the effective batch size the loss sees.
        The result is the exact same loss and gradient as training the wrapped loss with the full batch,
        at the cost of one extra (no-gradient) forward pass per batch.

        The wrapped loss must be computable from precomputed embeddings, i.e. expose
        ``compute_loss_from_embeddings(embeddings, labels)``, which most losses in this library do. Some
        losses cannot be wrapped, and each has a better alternative:

        - :class:`GISTEmbedLoss` embeds every batch with a second (guide) model: use :class:`CachedGISTEmbedLoss`.
        - :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` returns per-component losses
          and is scheduled by the sparse trainer: use
          :class:`~sentence_transformers.sparse_encoder.losses.CachedSpladeLoss`.
        - Losses with their own trainable parameters (e.g. :class:`SoftmaxLoss`) would have those
          parameters miss the loss scaling that the backward hook applies to the model's gradient.
        - Per-sample losses whose ``compute_loss_from_embeddings`` does not take the training labels
          (e.g. :class:`MSELoss`): their samples don't interact inside the loss, so plain gradient
          accumulation already trains them with an exactly equivalent gradient.

        Args:
            model: SentenceTransformer model
            loss: The loss to add gradient caching to, initialized with the same model.
            mini_batch_size: Mini-batch size for the forward pass. This denotes how much memory is actually
                used during training and evaluation; the larger the mini-batch size, the faster the
                training is, but the more memory is used. It does not affect the loss or the gradient.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The
                default is False.

        Requirements:
            1. The wrapped loss must expose ``compute_loss_from_embeddings(embeddings, labels)``.
            2. Should be used with large ``per_device_train_batch_size`` and a ``mini_batch_size`` that fits
               in memory, for superior performance at the training speed of two forward passes per batch.

        Inputs:
            +-------------------------------------------------+--------------------------------+
            | Inputs                                          | Labels                         |
            +=================================================+================================+
            | whatever the wrapped loss accepts               | whatever the wrapped loss uses |
            +-------------------------------------------------+--------------------------------+

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is this wrapper around
              :class:`MultipleNegativesRankingLoss`, plus a loss computation that is itself chunked over
              mini-batches, which additionally bounds the memory of the batch-by-batch similarity matrix.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.GradCacheLoss(model, losses.MultipleNegativesRankingLoss(model), mini_batch_size=32)

                args = SentenceTransformerTrainingArguments(
                    output_dir="output",
                    per_device_train_batch_size=1024,  # the loss sees the full batch, mini_batch_size bounds the memory
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
        self._validate_wrappable(model, loss)
        self.model = model
        self.loss = loss
        self.mini_batch_size = mini_batch_size
        self.show_progress_bar = show_progress_bar
        # Per-component values of the last dict-valued loss computation; logging-only (see forward).
        self._loss_components: dict[str, Tensor] | None = None

    @staticmethod
    def _validate_wrappable(model: Any, loss: nn.Module) -> None:
        from sentence_transformers.sentence_transformer.losses.gist_embed import GISTEmbedLoss
        from sentence_transformers.sparse_encoder.losses.splade import SpladeLoss

        if uses_gradient_cache(loss):
            raise ValueError(
                f"{loss.__class__.__name__} already uses gradient caching, so wrapping it in GradCacheLoss "
                "would embed every input four times per training step for no benefit. Use it directly."
            )
        if isinstance(loss, GISTEmbedLoss):
            raise ValueError(
                "GISTEmbedLoss embeds every batch with a second (guide) model, which GradCacheLoss cannot "
                "mini-batch for it. Use CachedGISTEmbedLoss instead."
            )
        if isinstance(loss, SpladeLoss):
            raise ValueError(
                "SpladeLoss is scheduled by the SparseEncoderTrainer, which would not recognize it inside "
                "a GradCacheLoss. Use CachedSpladeLoss instead."
            )

        model_parameters = {id(parameter) for parameter in model.parameters()}
        own_parameters = [
            name
            for name, parameter in loss.named_parameters()
            if id(parameter) not in model_parameters and parameter.requires_grad
        ]
        if own_parameters:
            raise ValueError(
                f"{loss.__class__.__name__} has trainable parameters of its own ({own_parameters[0]}, ...), "
                "which GradCacheLoss cannot support: the backward hook scales the model's gradient by "
                "whatever the outer backward pass provides (e.g. the gradient accumulation division), but "
                "the loss's own parameters receive their gradient before that scaling is known."
            )

        if not hasattr(loss, "compute_loss_from_embeddings"):
            raise ValueError(
                f"{loss.__class__.__name__} does not expose compute_loss_from_embeddings(embeddings, labels), "
                "which GradCacheLoss needs to compute the loss from the mini-batched embeddings."
            )
        parameters = list(inspect.signature(loss.compute_loss_from_embeddings).parameters.values())
        if len(parameters) < 2 or parameters[1].name != "labels":
            second = parameters[1].name if len(parameters) >= 2 else "nothing"
            raise ValueError(
                f"The second parameter of {loss.__class__.__name__}.compute_loss_from_embeddings is "
                f"{second!r}, not 'labels', so GradCacheLoss cannot pass the training labels through to it."
            )

        if has_static_embedding_input(model):
            raise ValueError(
                "GradCacheLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding, "
                "whose inputs cannot be split into mini-batches along a batch dimension. StaticEmbedding "
                "models are cheap to compute; use the wrapped loss directly instead."
            )

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the wrapped loss from the concatenated per-mini-batch embeddings.

        Unlike the dedicated Cached* losses, the loss computation itself is not chunked -- chunking
        requires knowledge of the loss's structure. ``mini_batch_size`` bounds the model activations,
        which dominate the memory use.
        """
        embeddings = [torch.cat(rep_mbs) if len(rep_mbs) > 1 else rep_mbs[0] for rep_mbs in reps]
        loss_output = self.loss.compute_loss_from_embeddings(embeddings, labels)
        if isinstance(loss_output, dict):
            self._loss_components = {key: value.detach() for key, value in loss_output.items()}
            loss = sum(loss_output.values())
        else:
            self._loss_components = None
            loss = loss_output
        if with_backward:
            loss.backward()
            loss = loss.detach()
        return loss

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> Tensor | dict[str, Tensor]:
        self._loss_components = None
        loss = self.forward_cached(sentence_features, labels)
        if self._loss_components is None:
            return loss
        # The wrapped loss reports per-component values; keep them visible to the trainer's logging.
        return reconstruct_loss_components(loss, self._loss_components)

    def get_config_dict(self) -> dict[str, Any]:
        return {"loss": self.loss, "mini_batch_size": self.mini_batch_size}

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
