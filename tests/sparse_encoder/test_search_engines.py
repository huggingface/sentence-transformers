from __future__ import annotations

import sys
import types

import pytest

from sentence_transformers.sparse_encoder.search_engines import semantic_search_seismic


class StubSeismicIndex:
    """Returns what Seismic returns: one row per query, in arbitrary order, empty rows for misses."""

    def __init__(self, rows):
        self.rows = rows

    def batch_search(self, **kwargs):
        return self.rows


@pytest.fixture
def stub_seismic_module(monkeypatch):
    """Seismic is not a test dependency, so stand in for the import that the function performs."""
    module = types.ModuleType("seismic")
    module.SeismicDataset = object
    module.SeismicIndex = StubSeismicIndex
    module.get_seismic_string = lambda: "U30"
    monkeypatch.setitem(sys.modules, "seismic", module)


def test_seismic_query_without_matches(stub_seismic_module) -> None:
    """A query that matches no document comes back as an empty row that carries no query id."""
    # Seismic hands back the rows for queries 2, <none> and 0, in that order.
    rows = [
        [("2", 0.9, "2"), ("2", 0.6, "3")],
        [],
        [("0", 0.8, "0"), ("0", 0.7, "1")],
    ]
    queries = [[("fruit", 1.0)], [("nomatch", 1.0)], [("veg", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=2)

    assert results == [
        [{"corpus_id": 0, "score": 0.8}, {"corpus_id": 1, "score": 0.7}],
        [],
        [{"corpus_id": 2, "score": 0.9}, {"corpus_id": 3, "score": 0.6}],
    ]


def test_seismic_orders_results_by_query(stub_seismic_module) -> None:
    """Every row carries its query id, so out-of-order rows still land on the right query."""
    rows = [
        [("1", 0.5, "7")],
        [("0", 0.4, "9")],
    ]
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=1)

    assert results == [[{"corpus_id": 9, "score": 0.4}], [{"corpus_id": 7, "score": 0.5}]]


def test_seismic_all_queries_without_matches(stub_seismic_module) -> None:
    """With no query matching anything there is no id to order by at all."""
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex([[], []]), top_k=3)

    assert results == [[], []]
