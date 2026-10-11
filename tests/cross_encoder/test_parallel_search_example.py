from __future__ import annotations

import argparse
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    from mcp.types import CallToolResult, TextContent
except ModuleNotFoundError as error:
    if error.name == "mcp":
        raise unittest.SkipTest(
            "Install parallel_search_requirements.txt to test the optional MCP example."
        ) from error
    raise

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cross_encoder/applications/parallel_search_reranking.py"
spec = importlib.util.spec_from_file_location("parallel_search_reranking", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ParallelSearchExampleTests(unittest.TestCase):
    def test_structured_and_text_results_preserve_sources(self):
        data = {
            "results": [
                {"title": "First", "url": "https://example.org/first", "excerpts": ["one", " ", "two"]},
                {"title": None, "url": "https://example.org/second", "excerpts": ["three"]},
                {"title": "Empty", "url": "https://example.org/empty", "excerpts": []},
            ]
        }
        structured = CallToolResult(content=[], structuredContent=data)
        text = CallToolResult(content=[TextContent(type="text", text=json.dumps(data))])
        expected = [
            {"title": "First", "url": "https://example.org/first", "text": "one\n\ntwo"},
            {"title": "https://example.org/second", "url": "https://example.org/second", "text": "three"},
        ]
        self.assertEqual(example.search_documents(structured), expected)
        self.assertEqual(example.search_documents(text), expected)

    def test_tool_error_is_not_treated_as_empty_search(self):
        result = CallToolResult(isError=True, content=[TextContent(type="text", text="rate limit")])
        with self.assertRaisesRegex(RuntimeError, "returned an error"):
            example.search_documents(result)

    def test_reranking_retains_correct_url_after_reordering(self):
        documents = [
            {"text": "unrelated", "url": "https://example.org/a", "title": "A"},
            {"text": "relevant", "url": "https://example.org/b", "title": "B"},
        ]
        model = Mock()
        model.rank.return_value = [{"corpus_id": 1, "score": 4.5}, {"corpus_id": 0, "score": -1.0}]
        ranked = example.rerank("query", documents, model, top_k=2)
        model.rank.assert_called_once_with("query", ["unrelated", "relevant"], top_k=2)
        self.assertEqual([item["url"] for item in ranked], [documents[1]["url"], documents[0]["url"]])
        self.assertEqual([item["score"] for item in ranked], [4.5, -1.0])

    def test_empty_results_do_not_invoke_model(self):
        model = Mock()
        self.assertEqual(example.rerank("query", [], model, top_k=3), [])
        model.rank.assert_not_called()

    def test_top_k_must_be_positive(self):
        for value in ["0", "-1"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                example.positive_int(value)


if __name__ == "__main__":
    unittest.main()
