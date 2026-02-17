from __future__ import annotations

from collections import UserDict
from typing import Any

import pytest
import torch
from torch import Tensor

from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder


class BaseModelPreprocessTest:
    def create_dense_text_model(
        self, request: pytest.FixtureRequest
    ) -> Any:  # pragma: no cover - to be implemented by subclasses
        """Return a dense text model instance for the concrete test class.

        Subclasses should use ``request.getfixturevalue(...)`` to obtain the
        concrete model fixture, e.g. ``stsb_bert_tiny_model`` or
        ``splade_bert_tiny_model``.
        """
        raise NotImplementedError

    def create_text_inputs(self) -> list:  # pragma: no cover - to be implemented by subclasses
        raise NotImplementedError

    @pytest.fixture(autouse=True)
    def _setup_model(self, request: pytest.FixtureRequest) -> None:
        # Each test gets a fresh model instance from the subclass implementation.
        self.model = self.create_dense_text_model(request)

    @pytest.fixture(autouse=True)
    def _setup_inputs(self) -> None:
        # Each test gets fresh inputs from the subclass implementation.
        self.inputs = self.create_text_inputs()

    def test_preprocess(self) -> None:
        model = self.model
        inputs = self.inputs
        input_length = len(inputs)
        features = model.preprocess(inputs)

        assert isinstance(features, (dict, UserDict))

        for key, value in features.items():
            if isinstance(value, Tensor):
                assert value.size(0) == input_length

    def test_preprocess_accepts_prompt(self) -> None:
        model = self.model
        inputs = self.inputs

        features_without_prompt = model.preprocess(inputs)
        features_with_prompt = model.preprocess(inputs, prompt="Instruction: ")

        assert isinstance(features_without_prompt, (dict, UserDict))
        assert isinstance(features_with_prompt, (dict, UserDict))
        # At minimum, all keys present without a prompt should also
        # be present when a prompt is used, even if additional keys
        # (e.g. prompt length metadata) are added.
        assert set(features_without_prompt.keys()).issubset(features_with_prompt.keys())
        assert any(
            not torch.equal(features_without_prompt[key], features_with_prompt[key])
            for key in features_without_prompt.keys()
            if isinstance(features_without_prompt[key], Tensor)
        )


class TestSentenceTransformerPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> SentenceTransformer:
        return request.getfixturevalue("stsb_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return ["This is a test.", "Another test sentence."]


class TestCrossEncoderPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> CrossEncoder:
        return request.getfixturevalue("reranker_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return [("This is a test.", "This is a test."), ("Another test sentence.", "Yet another sentence.")]


class TestSparseEncoderPreprocess(BaseModelPreprocessTest):
    def create_dense_text_model(self, request: pytest.FixtureRequest) -> SparseEncoder:
        return request.getfixturevalue("splade_bert_tiny_model")

    def create_text_inputs(self) -> list:
        return ["This is a test.", "Another test sentence."]
