from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_scores
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import _create_minibatch
from sentence_transformers.util import (
    all_gather_padded,
    cat_padded_token_embeddings,
    get_rank,
    get_world_size,
    stack_padded_token_embeddings,
)


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


class RandContext:
    """A random-state snapshot used to reproduce a forward pass during the GradCache 2nd-phase backward.

    Reference: https://github.com/luyug/GradCache.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        if torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is not supported for this operation. Please use CPU or CUDA.")
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
    loss_obj: CachedMultiVectorMultipleNegativesRankingLoss,
) -> None:
    """Re-run the embedding forward (with gradients enabled) for each mini-batch and plug the cached
    per-embedding gradients back into the autograd graph."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for idx, (sentence_feature, grad, random_states) in enumerate(
            zip(sentence_features, loss_obj.cache, loss_obj.random_states)
        ):
            task = sentence_feature.get("task", "query" if idx == 0 else "document")
            for (reps_mb, _, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    task=task,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()


class CachedMultiVectorMultipleNegativesRankingLoss(nn.Module):
    """A GradCache version of :class:`MultiVectorMultipleNegativesRankingLoss`.

    Enables much larger effective batch sizes than the non-cached loss at the cost of being slightly slower:
    embeddings are computed in chunks of ``mini_batch_size`` (under ``torch.no_grad``), the cross-entropy
    loss is computed once over the full batch, the per-embedding gradients are cached, and a second pass
    re-runs the embedding forward in chunks with gradients enabled, feeding the cached gradients into the
    final ``loss.backward()``.

    Reference: https://github.com/luyug/GradCache (Gao et al., 2021).

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder`.
        score_metric: Scoring callable. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores`. Pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` for XTR-style scoring.
        mini_batch_size: Chunk size for the **embedding** forward / backward pass. Keep small enough that a
            single chunk fits in GPU memory.
        score_mini_batch_size: Chunk size for the **scoring** phase (independent of ``mini_batch_size``).
            Smaller values trim transient scoring intermediates ``(Q, Q*N, q_tokens, d_tokens)`` which are
            usually the bottleneck at large effective batch sizes. Defaults to ``mini_batch_size``.
        scale: ``1 / temperature``. Scores are multiplied by ``scale`` before cross-entropy. Defaults to
            ``1.0`` (``temperature=1.0``), matching PyLate. MaxSim is an unbounded sum over query-token
            similarities, so (unlike bounded cosine, where the dense loss uses ``scale=20.0``) it needs no
            amplification. ``scale=20`` would saturate the softmax. See
            :class:`MultiVectorMultipleNegativesRankingLoss` for the full rationale.
        temperature: Optional alias for ``1 / scale``.
        size_average: Whether to average (``True``, default) or sum the cross-entropy loss across the batch.
        gather_across_devices: If True, AllGather document embeddings across DDP ranks.
        show_progress_bar: If True, show a TQDM progress bar for the embedding / scoring steps.
    """

    # Enables per-sample media counting in Transformer.preprocess for VLM minibatching
    requires_media_counts = True

    def __init__(
        self,
        model: MultiVectorEncoder,
        score_metric: Callable | None = None,
        mini_batch_size: int = 32,
        score_mini_batch_size: int | None = None,
        scale: float = 1.0,
        temperature: float | None = None,
        size_average: bool = True,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_metric = score_metric if score_metric is not None else colbert_scores
        self.mini_batch_size = mini_batch_size
        self.score_mini_batch_size = score_mini_batch_size if score_mini_batch_size is not None else mini_batch_size
        if temperature is not None:
            scale = 1.0 / temperature
        self.scale = scale
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.show_progress_bar = show_progress_bar

        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

    def get_config_dict(self) -> dict[str, Any]:
        score_metric = getattr(self.score_metric, "__name__", type(self.score_metric).__name__)
        # Configured metric objects (e.g. XTRScores) expose their own config, include it.
        metric_config = getattr(self.score_metric, "get_config_dict", None)
        if metric_config is not None:
            args = ", ".join(f"{key}={value!r}" for key, value in metric_config().items())
            score_metric = f"{score_metric}({args})"
        return {
            "score_metric": score_metric,
            "mini_batch_size": self.mini_batch_size,
            "score_mini_batch_size": self.score_mini_batch_size,
            "scale": self.scale,
            "size_average": self.size_average,
            "gather_across_devices": self.gather_across_devices,
        }

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        task: str,
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, Tensor, RandContext | None]:
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        # Grid-aware slicing: flattened VLM tensors (pixel_values) and FA2 metadata can't be sliced per sample.
        mb = _create_minibatch(sentence_feature, begin, end)
        with random_state_context:
            with grad_context():
                random_state = (
                    RandContext(*[v for v in mb.values() if isinstance(v, Tensor)]) if copy_random_state else None
                )
                outputs = self.model(mb, task=task)
                # If a Normalize module is in the pipeline, token_embeddings is already L2-normalized.
                embeddings = outputs["token_embeddings"]
                # After MultiVectorMask, attention_mask is the per-row scoring mask.
                mask = outputs["attention_mask"].bool()
        return embeddings, mask, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        task: str,
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
        bsz = _get_batch_size(sentence_feature)
        for i, b in enumerate(
            tqdm.trange(0, bsz, self.mini_batch_size, desc="Embed mini-batches", disable=not self.show_progress_bar)
        ):
            e = b + self.mini_batch_size
            reps, mask, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                task=task,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, mask, random_state

    def _calculate_loss(
        self,
        reps: list[list[Tensor]],
        masks_chunks: list[list[Tensor]],
        with_backward: bool,
    ) -> Tensor:
        # Each per-column ``reps[i]`` is a list of mini-batch ``(B_mini, T_mini, D)`` chunks. For
        # native-resolution VLMs (Qwen2-VL family) each mini-batch can emit a different ``T``, so
        # pad each chunk to the column's max ``T`` before concatenating along the batch axis.
        catted = [cat_padded_token_embeddings(r, m) for r, m in zip(reps, masks_chunks)]
        embeddings_anchor = catted[0][0]
        embeddings_other = [emb for emb, _ in catted[1:]]
        masks = [mask for _, mask in catted]
        batch_size = len(embeddings_anchor)

        if self.gather_across_devices:
            # Pad the token axis to the cross-rank max before gathering: each rank pads its columns to
            # its own batch-longest, so T differs per rank and all_gather needs a uniform shape per rank.
            gathered = [all_gather_padded(e, m, with_grad=True) for e, m in zip(embeddings_other, masks[1:])]
            embeddings_other = [e for e, _ in gathered]
            masks = [masks[0], *[m for _, m in gathered]]

        N = len(embeddings_other)
        docs_stacked, docs_mask_stacked = stack_padded_token_embeddings(embeddings_other, masks[1:])
        q_mask = masks[0]

        labels = torch.arange(batch_size, device=reps[0][0].device) * N
        if self.gather_across_devices:
            labels = labels + get_rank() * batch_size * N

        losses: list[Tensor] = []
        for begin in tqdm.trange(
            0, batch_size, self.score_mini_batch_size, desc="Score mini-batches", disable=not self.show_progress_bar
        ):
            end = begin + self.score_mini_batch_size
            scores = self.score_metric(
                embeddings_anchor[begin:end],
                docs_stacked,
                queries_mask=q_mask[begin:end],
                documents_mask=docs_mask_stacked,
            )
            loss_mb = F.cross_entropy(scores * self.scale, labels[begin:end], reduction="sum")
            # Average inside the graph: dividing the detached sum after backward would leave gradients scaled by batch_size.
            if self.size_average:
                loss_mb = loss_mb / batch_size
            if self.gather_across_devices:
                loss_mb = loss_mb * get_world_size()
            if with_backward:
                loss_mb.backward()
                loss_mb = loss_mb.detach()
            losses.append(loss_mb)

        return torch.stack(losses).sum()

    def _calculate_loss_and_cache_gradients(
        self,
        reps: list[list[Tensor]],
        masks_chunks: list[list[Tensor]],
    ) -> Tensor:
        loss = self._calculate_loss(reps, masks_chunks, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        reps: list[list[Tensor]] = []
        masks_chunks: list[list[Tensor]] = []
        self.random_states = []

        for idx, sentence_feature in enumerate(sentence_features):
            # Collator-stamped task (column 0 is the query unless router_mapping overrides it).
            task = sentence_feature.get("task", "query" if idx == 0 else "document")
            reps_mbs: list[Tensor] = []
            mask_mbs: list[Tensor] = []
            random_state_mbs: list[RandContext] = []
            for reps_mb, mask_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                task=task,
                with_grad=False,
                copy_random_state=True,
            ):
                # Only token_embeddings are gradient-cached: extra per-token channels must ride
                # INSIDE this tensor (trailing columns), a separate tensor would get no gradient
                # in the second pass.
                reps_mbs.append(reps_mb.detach().requires_grad_())
                mask_mbs.append(mask_mb)
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            masks_chunks.append(mask_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            loss = self._calculate_loss_and_cache_gradients(reps, masks_chunks)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self._calculate_loss(reps, masks_chunks, with_backward=False)
        return loss

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
