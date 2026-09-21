from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from packaging.version import Version
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from sentence_transformers.base.losses.gradcache import CachedLossMixin


class TinyModel(nn.Module):
    def __init__(self, routed=False, buffers=False):
        super().__init__()
        self.linear = nn.Linear(5, 4)
        self.forward_calls = []
        self.document = nn.Linear(5, 4) if routed else None
        if buffers:
            self.register_buffer("offset", torch.ones(4))

    def forward(self, features):
        layer = self.document if self.document is not None and features["column"] else self.linear
        reps = layer(features["values"])
        if hasattr(self, "offset"):
            reps = reps + self.offset
        if features["frozen"].all():
            reps = reps.detach()
        else:
            reps = torch.where(features["frozen"].unsqueeze(1), reps.detach(), reps)
        self.forward_calls.append((torch.is_grad_enabled(), reps.requires_grad))
        return {"sentence_embedding": reps}


class TinyCachedLoss(nn.Module, CachedLossMixin):
    def __init__(self, model, mini_batch_size=4, mini_batch_num_tokens=None):
        super().__init__()
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens

    def calculate_loss(self, reps, labels=None, *, with_backward=False):
        anchors, positives = (torch.cat(column) for column in reps)
        loss = F.cross_entropy(anchors @ positives.T, torch.arange(len(anchors)))
        if with_backward:
            loss.backward()
        return loss.detach()

    def forward(self, features):
        return self.forward_cached(features)


def _features(rank, case):
    generator = torch.Generator().manual_seed(100 + rank)
    features = []
    for column in range(2):
        lengths = torch.randint(1, 5, (9,), generator=generator)
        frozen = torch.zeros(9, dtype=torch.bool)
        if (
            case == "all_frozen"
            or (case.startswith("frozen_last") and column == 1)
            or (case == "frozen_first" and column == 0)
            or (case == "frozen_one_rank" and column == 1 and rank == 1)
        ):
            frozen[:] = True
        elif case == "mixed_frozen":
            frozen[4:] = True
        features.append(
            {
                "input_ids": torch.zeros(9, 4, dtype=torch.long),
                "attention_mask": torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1),
                "values": torch.randn(9, 5, generator=generator),
                "frozen": frozen,
                "column": column,
            }
        )
    return features


def _run_gradcache_ddp(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method, timeout=timedelta(seconds=30)
    )
    try:
        for case in (
            "dense",
            "single_minibatch",
            "token_budget",
            "buffers",
            "frozen_last",
            "frozen_last_unused",
            "frozen_one_rank",
            "frozen_first",
            "mixed_frozen",
            "all_frozen",
            "routed",
        ):
            torch.manual_seed(42)
            model = TinyModel(routed=case == "routed", buffers=case in ("buffers", "frozen_one_rank"))
            reference = deepcopy(model)
            ddp = DistributedDataParallel(model, find_unused_parameters=case in ("routed", "frozen_last_unused"))
            sync_count = [0]

            def count_sync(state, bucket):
                sync_count[0] += 1
                tensor = bucket.buffer()
                dist.all_reduce(tensor)
                tensor.div_(world_size)
                future = torch.futures.Future()
                future.set_result(tensor)
                return future

            ddp.register_comm_hook(None, count_sync)
            loss_fn = TinyCachedLoss(
                ddp,
                mini_batch_size=32 if case == "single_minibatch" else 4,
                mini_batch_num_tokens=6 if case in ("token_budget", "buffers") else None,
            )
            features = _features(rank, case)
            if ddp.find_unused_parameters and Version(torch.__version__) < Version("2.4"):
                with pytest.raises(ValueError, match="find_unused_parameters=True require PyTorch >=2.4"):
                    loss_fn(features).backward()
                continue
            optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            for _ in range(2):
                optimizer.zero_grad()
                reference_optimizer.zero_grad()
                sync_count[0] = 0
                for accumulating in (True, False):
                    model.forward_calls.clear()
                    with ddp.no_sync() if accumulating else nullcontext():
                        loss = loss_fn(features) / 2
                        assert not any(with_grad for with_grad, _ in model.forward_calls), case
                        num_minibatches = len(model.forward_calls)
                        model.forward_calls.clear()
                        loss.backward()
                    replay = model.forward_calls[:num_minibatches]
                    needs_flush = not accumulating and any(trainable for _, trainable in replay) and not replay[-1][1]
                    assert len(model.forward_calls) == num_minibatches + int(needs_flush), case
                    expected_syncs = 0 if accumulating or case == "all_frozen" else 1
                    assert sync_count[0] == expected_syncs, (case, rank, accumulating, sync_count[0])
                    assert ddp.require_backward_grad_sync

                    anchors, positives = [reference(feature)["sentence_embedding"] for feature in features]
                    reference_loss = F.cross_entropy(anchors @ positives.T, torch.arange(len(anchors))) / 2
                    torch.testing.assert_close(loss, reference_loss)
                    if reference_loss.requires_grad:
                        reference_loss.backward()

                for actual, expected in zip(ddp.parameters(), reference.parameters()):
                    if expected.grad is None:
                        assert actual.grad is None
                        continue
                    dist.all_reduce(expected.grad)
                    expected.grad.div_(world_size)
                    torch.testing.assert_close(
                        actual.grad, expected.grad, atol=1e-6, rtol=1e-5, msg=lambda msg: f"{case}: {msg}"
                    )
                optimizer.step()
                reference_optimizer.step()
                for actual, expected in zip(ddp.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, msg=case)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("world_size", [1, 2])
def test_gradcache_ddp(tmp_path, world_size):
    init_method = (tmp_path / "store").as_uri()
    if world_size == 1:
        num_threads = torch.get_num_threads()
        try:
            _run_gradcache_ddp(0, world_size, init_method)
        finally:
            torch.set_num_threads(num_threads)
    else:
        mp.spawn(_run_gradcache_ddp, args=(world_size, init_method), nprocs=world_size, join=True)
