from __future__ import annotations

import logging

import pytest

from sentence_transformers.base.modules.input_module import InputModule


class _TokenizeOnlyModule(InputModule):
    """Subclass that overrides tokenize() but not preprocess() (legacy pattern)."""

    def forward(self, features):
        return features

    def tokenize(self, texts, **kwargs):
        return {"tokens": texts}

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        pass


class _PreprocessModule(InputModule):
    """Subclass that properly overrides preprocess()."""

    def forward(self, features):
        return features

    def preprocess(self, inputs, prompt=None, **kwargs):
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)
        return {"tokens": inputs}

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        pass


class _BareModule(InputModule):
    """Subclass that overrides neither preprocess() nor tokenize()."""

    def forward(self, features):
        return features

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        pass


class TestPreprocessBackwardCompat:
    def test_tokenize_only_subclass_delegates_and_warns(self, caplog):
        module = _TokenizeOnlyModule()
        with caplog.at_level(logging.WARNING):
            result = module.preprocess(["hello", "world"])
        assert result == {"tokens": ["hello", "world"]}
        assert any("overrides `tokenize` instead of `preprocess`" in r.message for r in caplog.records)

    def test_tokenize_only_subclass_prepends_prompt(self):
        module = _TokenizeOnlyModule()
        result = module.preprocess(["hello"], prompt="search: ")
        assert result == {"tokens": ["search: hello"]}

    def test_preprocess_subclass_works_without_warning(self, caplog):
        module = _PreprocessModule()
        with caplog.at_level(logging.WARNING):
            result = module.preprocess(["hello"])
        assert result == {"tokens": ["hello"]}
        assert not any("tokenize" in r.message for r in caplog.records)

    def test_preprocess_subclass_with_prompt(self):
        module = _PreprocessModule()
        result = module.preprocess(["hello"], prompt="query: ")
        assert result == {"tokens": ["query: hello"]}

    def test_bare_subclass_raises_not_implemented(self):
        module = _BareModule()
        with pytest.raises(NotImplementedError, match="must implement the `preprocess` method"):
            module.preprocess(["hello"])


class TestTokenizeDeprecation:
    def test_tokenize_on_preprocess_subclass_warns_and_delegates(self, caplog):
        module = _PreprocessModule()
        with caplog.at_level(logging.WARNING):
            result = module.tokenize(["hello"])
        assert result == {"tokens": ["hello"]}
        assert any("tokenize" in r.message and "deprecated" in r.message for r in caplog.records)

    def test_tokenize_on_bare_subclass_raises(self):
        module = _BareModule()
        with pytest.raises(NotImplementedError):
            module.tokenize(["hello"])
