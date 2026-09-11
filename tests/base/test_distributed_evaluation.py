from __future__ import annotations

import socket
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Dense
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer
from sentence_transformers.util.distributed import distributed_evaluation


class RecordingWordEmbeddings(WordEmbeddings):
    def __init__(self):
        vocab = ["[PAD]", "cat", "dog", "bird", "fish", "hello", "world"]
        super().__init__(
            WhitespaceTokenizer(vocab=vocab),
            torch.rand(len(vocab), 16, generator=torch.Generator().manual_seed(12)),
        )
        self.seen = []

    def preprocess(self, inputs, **kwargs):
        self.seen.extend(inputs)
        if "fail" in inputs:
            raise ValueError("worker inference failed")
        texts = [" ".join(value) if isinstance(value, (tuple, list)) else value for value in inputs]
        return super().preprocess(texts, **kwargs)


def _make_model(model_type):
    words = RecordingWordEmbeddings()
    if model_type == "dense":
        return SentenceTransformer(modules=[words, Pooling(16, "mean")], device="cpu")
    if model_type == "sparse":
        return SparseEncoder(modules=[words, Pooling(16, "mean")], device="cpu")
    if model_type == "multi_vector":
        return MultiVectorEncoder(modules=[words], device="cpu")
    torch.manual_seed(12)
    head = Dense(16, 2, activation_function=None, module_output_name="scores")
    return CrossEncoder(modules=[words, Pooling(16, "mean"), head], device="cpu")


def _assert_output_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.layout == expected.layout
        assert actual.device == expected.device
        torch.testing.assert_close(actual, expected)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_output_equal(actual[key], expected[key])
    else:
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_output_equal(value, reference)


def _run_distributed_inference(rank, world_size, port, model_type, use_cuda):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        model = _make_model(model_type)
        inputs = ["cat", "dog bird fish", "fish", "hello world", "bird dog"]
        method = "predict" if model_type == "cross_encoder" else "encode"
        if model_type == "cross_encoder":
            inputs = [("hello", text) for text in inputs]
        cases = [
            (method, inputs, {}),
            (method, inputs[:1], {}),
            (method, inputs[0], {}),
            (method, [], {}),
            (method, inputs, {"device": "cpu"}),
        ]
        if model_type == "dense":
            cases.extend(
                [
                    (method, inputs, {"convert_to_tensor": True}),
                    (method, inputs, {"convert_to_numpy": False}),
                    (method, inputs, {"output_value": "token_embeddings"}),
                    (method, inputs, {"precision": "int8"}),
                    (method, inputs, {"precision": "binary"}),
                    ("encode_query", inputs, {"truncate_dim": 8, "normalize_embeddings": True}),
                    ("encode_document", inputs, {"prompt": "hello "}),
                ]
            )
        elif model_type == "sparse":
            cases.extend(
                [
                    (method, inputs, {"convert_to_tensor": False}),
                    (method, inputs, {"convert_to_sparse_tensor": False, "max_active_dims": 3}),
                    ("encode_query", inputs, {"save_to_cpu": True}),
                ]
            )
        elif model_type == "multi_vector":
            cases.extend(
                [
                    (method, inputs, {"convert_to_numpy": True}),
                    ("encode_query", inputs, {"normalize_embeddings": True}),
                    ("encode_document", inputs, {"prompt": "hello "}),
                ]
            )
        else:
            cases.extend(
                [
                    (method, inputs, {"convert_to_tensor": True, "apply_softmax": True}),
                    (method, inputs, {"convert_to_numpy": False}),
                    (method, inputs, {"activation_fn": lambda scores: scores + 1}),
                ]
            )

        for name, values, kwargs in cases:
            model.to(device)
            inference = getattr(model, name)
            expected = inference(values, show_progress_bar=False, **kwargs)
            model[0].seen.clear()
            with distributed_evaluation(model):
                if rank == 0:
                    actual = inference(values, show_progress_bar=False, **kwargs)
                    _assert_output_equal(actual, expected)
            assert model._distributed_inference is None
            shards = [None] * world_size
            dist.all_gather_object(shards, model[0].seen)
            normalized = [values] if model.is_singular_input(values) else values
            processed = [value for shard in shards for value in shard]
            assert sorted(processed) == sorted(normalized)
            if "device" in kwargs or "activation_fn" in kwargs:
                assert not shards[1]
            elif len(normalized) >= world_size:
                assert all(shards)

        if model_type == "dense":
            for failing_inputs in (["cat", "fail"], ["fail"] * world_size):
                with pytest.raises(RuntimeError, match="worker inference failed") as exc_info:
                    with distributed_evaluation(model):
                        if rank == 0:
                            model.encode(failing_inputs)
                for failed_rank, text in enumerate(failing_inputs):
                    if text == "fail":
                        assert f"Rank {failed_rank}:\n" in str(exc_info.value)
                assert "in preprocess" in str(exc_info.value)
                assert "raise ValueError" in str(exc_info.value)
                assert model._distributed_inference is None
            with pytest.raises((ValueError, RuntimeError), match="metric failed"):
                with distributed_evaluation(model):
                    if rank == 0:
                        raise ValueError("metric failed")
            with distributed_evaluation(model):
                if rank == 0:
                    assert model.encode(["cat", "dog"]).shape == (2, 16)

        model.to(device)
        for clear_defaults in (False, True):
            if rank == 0 or clear_defaults:
                model.prompts["evaluation"] = "hello "
                model.default_prompt_name = "evaluation"
                if model_type == "dense":
                    model.truncate_dim = 8
                elif model_type == "sparse":
                    model.max_active_dims = 3
                elif model_type == "cross_encoder":
                    model.activation_fn = torch.nn.Softmax(dim=-1)

            if rank == 0 and clear_defaults:
                model.default_prompt_name = None
                if model_type == "dense":
                    model.truncate_dim = None
                elif model_type == "sparse":
                    model.max_active_dims = None
                elif model_type == "cross_encoder":
                    model.activation_fn = None

            for kwargs in ({}, {"prompt_name": "evaluation"}, {"prompt": "dog "}):
                if rank == 0:
                    expected = getattr(model, method)(inputs, show_progress_bar=False, **kwargs)
                model[0].seen.clear()
                with distributed_evaluation(model):
                    if rank == 0:
                        actual = getattr(model, method)(inputs, show_progress_bar=False, **kwargs)
                        _assert_output_equal(actual, expected)
                shards = [None] * world_size
                dist.all_gather_object(shards, model[0].seen)
                assert all(shards)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_type", ["dense", "sparse", "multi_vector", "cross_encoder"])
@pytest.mark.parametrize(
    "use_cuda",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2 or not dist.is_nccl_available(), reason="Requires two GPUs and NCCL"
            ),
        ),
    ],
)
def test_distributed_evaluator_inference(model_type, use_cuda):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_run_distributed_inference, args=(2, port, model_type, use_cuda), nprocs=2, join=True)
