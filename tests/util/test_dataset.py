from __future__ import annotations

import logging

import pytest
from datasets import Dataset, DatasetDict

from sentence_transformers.util import resolve_ids


@pytest.fixture
def queries() -> Dataset:
    return Dataset.from_dict({"query_id": ["q1", "q2"], "text": ["Q one", "Q two"]})


@pytest.fixture
def documents() -> Dataset:
    return Dataset.from_dict({"document_id": ["d1", "d2", "d3"], "text": ["A", "B", "C"]})


class TestShapes:
    def test_kd_shape_explodes_lists_into_numbered_columns(self, queries: Dataset, documents: Dataset) -> None:
        train = Dataset.from_dict(
            {"query_id": ["q1"], "document_ids": [["d1", "d2", "d3"]], "scores": [[1.0, 0.5, 0.2]]}
        )
        train.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}))
        assert train[0] == {
            "query": "Q one",
            "document_1": "A",
            "document_2": "B",
            "document_3": "C",
            "scores": [1.0, 0.5, 0.2],
        }

    def test_explode_positions_align_across_rows(self, documents: Dataset) -> None:
        # Column j holds the j-th candidate of every row.
        transform = resolve_ids({"document_ids": documents})
        out = transform({"document_ids": [["d1", "d2"], ["d3", "d1"]]})
        assert out == {"document_1": ["A", "C"], "document_2": ["B", "A"]}

    def test_ragged_lists_raise(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents})
        with pytest.raises(ValueError, match="differing lengths"):
            transform({"document_ids": [["d1", "d2"], ["d3"]]})

    def test_ragged_lists_unified_by_max_list_length(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents}, max_list_length=1)
        out = transform({"document_ids": [["d1", "d2"], ["d3"]]})
        assert out == {"document_1": ["A", "C"]}

    def test_triplet_shape_shares_one_lookup(self, queries: Dataset, documents: Dataset) -> None:
        train = Dataset.from_dict({"query_id": ["q1"], "positive_id": ["d1"], "negative_id": ["d3"]})
        train.set_transform(resolve_ids({"query_id": queries, "positive_id": documents, "negative_id": documents}))
        assert train[0] == {"query": "Q one", "positive": "A", "negative": "C"}

    def test_output_order_follows_lookups_insertion(self, queries: Dataset, documents: Dataset) -> None:
        # The losses read columns positionally, so the first lookups key must come out first.
        train = Dataset.from_dict({"document_ids": [["d1"]], "query_id": ["q1"], "scores": [[1.0]]})
        transform = resolve_ids({"query_id": queries, "document_ids": documents})
        out = transform(train[:])
        assert list(out.keys()) == ["query", "document_1", "scores"]

    def test_empty_batch(self, queries: Dataset) -> None:
        transform = resolve_ids({"query_id": queries})
        assert transform({"query_id": []}) == {"query": []}

    def test_integer_ids(self) -> None:
        # MS MARCO-style integer IDs are the most common variant after strings.
        queries = Dataset.from_dict({"query_id": [7, 8], "text": ["Q7", "Q8"]})
        documents = Dataset.from_dict({"document_id": [100, 200], "text": ["A", "B"]})
        transform = resolve_ids({"query_id": queries, "document_ids": documents})
        out = transform({"query_id": [8], "document_ids": [[200, 100]]})
        assert out == {"query": ["Q8"], "document_1": ["B"], "document_2": ["A"]}

    def test_set_transform_and_batched_map_agree(self, queries: Dataset, documents: Dataset) -> None:
        rows = {"query_id": ["q1", "q2"], "document_ids": [["d1", "d2"], ["d2", "d3"]]}
        transform = resolve_ids({"query_id": queries, "document_ids": documents})

        lazy = Dataset.from_dict(rows)
        lazy.set_transform(transform)
        eager = Dataset.from_dict(rows).map(transform, batched=True, remove_columns=list(rows))
        assert [lazy[i] for i in range(2)] == [eager[i] for i in range(2)]


class TestListsOutputFormat:
    def test_nested_kd_shape(self, queries: Dataset, documents: Dataset) -> None:
        # The CrossEncoder listwise shape: one query with a nested candidate list + score list.
        train = Dataset.from_dict(
            {"query_id": ["q1"], "document_ids": [["d1", "d2", "d3"]], "scores": [[1.0, 0.5, 0.2]]}
        )
        train.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}, output_format="lists"))
        assert train[0] == {"query": "Q one", "documents": ["A", "B", "C"], "scores": [1.0, 0.5, 0.2]}

    def test_ragged_rows_allowed(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents}, output_format="lists")
        out = transform({"document_ids": [["d1", "d2"], ["d3"]]})
        assert out == {"documents": [["A", "B"], ["C"]]}

    def test_truncation_applies(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents}, output_format="lists", max_list_length=1)
        out = transform({"document_ids": [["d1", "d2"], ["d3"]], "scores": [[0.9, 0.5], [0.7]]})
        assert out["documents"] == [["A"], ["C"]]
        assert out["scores"] == [[0.9], [0.7]]

    @pytest.mark.parametrize(
        ("input_col", "output_col"),
        [
            ("document_ids", "documents"),
            ("query_ids", "queries"),
            ("candidates", "candidates"),
        ],
    )
    def test_plural_naming(self, documents: Dataset, input_col: str, output_col: str) -> None:
        transform = resolve_ids({input_col: documents}, output_format="lists")
        out = transform({input_col: [["d1"]]})
        assert list(out.keys()) == [output_col]

    def test_single_columns_unaffected(self, queries: Dataset, documents: Dataset) -> None:
        transform = resolve_ids({"query_id": queries, "positive_id": documents}, output_format="lists")
        out = transform({"query_id": ["q1"], "positive_id": ["d1"]})
        assert out == {"query": ["Q one"], "positive": ["A"]}


class TestMaxListLength:
    def test_truncates_ids_and_list_labels_in_parallel(self, queries: Dataset, documents: Dataset) -> None:
        train = Dataset.from_dict(
            {"query_id": ["q1"], "document_ids": [["d1", "d2", "d3"]], "scores": [[1.0, 0.5, 0.2]]}
        )
        train.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=2))
        row = train[0]
        assert row["document_1"] == "A"
        assert row["document_2"] == "B"
        assert "document_3" not in row
        assert row["scores"] == [1.0, 0.5]

    def test_tuple_rows_count_as_lists(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents}, max_list_length=2)
        out = transform({"document_ids": [("d1", "d2", "d3")], "scores": [(0.9, 0.5, 0.1)]})
        assert out["document_1"] == ["A"]
        assert out["document_2"] == ["B"]
        assert out["scores"] == [(0.9, 0.5)]

    def test_scalar_label_not_truncated(self, queries: Dataset, documents: Dataset) -> None:
        transform = resolve_ids({"query_id": queries, "positive_id": documents}, max_list_length=1)
        out = transform({"query_id": ["q1"], "positive_id": ["d1"], "label": [1.0]})
        assert out["label"] == [1.0]


class TestColumnSelection:
    def test_unmapped_columns_dropped_and_default_labels_kept(self, queries: Dataset, documents: Dataset) -> None:
        transform = resolve_ids({"query_id": queries, "document_ids": documents})
        out = transform(
            {
                "query_id": ["q1"],
                "document_ids": [["d1"]],
                "scores": [[1.0]],
                "source": ["web"],
            }
        )
        assert "source" not in out
        assert out["scores"] == [[1.0]]

    def test_custom_keep_columns(self, queries: Dataset, documents: Dataset) -> None:
        transform = resolve_ids(
            {"query_id": queries, "document_ids": documents}, keep_columns=["my_score"], max_list_length=1
        )
        out = transform(
            {"query_id": ["q1"], "document_ids": [["d1", "d2"]], "my_score": [[9.0, 3.0]], "scores": [[1.0, 0.5]]}
        )
        assert out["document_1"] == ["A"]
        assert out["my_score"] == [[9.0]]
        assert "scores" not in out

    def test_absent_keep_columns_ignored(self, queries: Dataset) -> None:
        transform = resolve_ids({"query_id": queries}, keep_columns=["label", "scores"])
        assert transform({"query_id": ["q1"]}) == {"query": ["Q one"]}


class TestOutputNaming:
    @pytest.mark.parametrize(
        ("input_col", "output_col"),
        [
            ("query_id", "query"),
            ("positive_id", "positive"),
            ("doc_idx", "doc"),
            ("candidates", "candidates"),
        ],
    )
    def test_suffix_stripping_single_columns(self, documents: Dataset, input_col: str, output_col: str) -> None:
        transform = resolve_ids({input_col: documents})
        out = transform({input_col: ["d1"]})
        assert list(out.keys()) == [output_col]

    @pytest.mark.parametrize(
        ("input_col", "output_base"),
        [
            ("document_ids", "document"),
            ("query_ids", "query"),
            ("candidates", "candidates"),
        ],
    )
    def test_numbering_from_singular_base(self, documents: Dataset, input_col: str, output_base: str) -> None:
        transform = resolve_ids({input_col: documents})
        out = transform({input_col: [["d1", "d2"]]})
        assert list(out.keys()) == [f"{output_base}_1", f"{output_base}_2"]


class TestLookupSpecs:
    def test_two_tuple_explicit_id_col(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": (documents, "document_id")})
        assert transform({"document_ids": [["d2"]]}) == {"document_1": ["B"]}

    def test_three_tuple_for_wide_lookup(self) -> None:
        wide = Dataset.from_dict({"document_id": ["d1"], "title": ["T"], "text": ["A"]})
        transform = resolve_ids({"document_ids": (wide, "document_id", "text")})
        assert transform({"document_ids": [["d1"]]}) == {"document_1": ["A"]}

    def test_generic_id_column_in_both_lookups(self) -> None:
        qs = Dataset.from_dict({"id": ["q1"], "text": ["Q one"]})
        docs = Dataset.from_dict({"id": ["d1"], "text": ["A"]})
        transform = resolve_ids({"query_id": qs, "positive_id": docs})
        assert transform({"query_id": ["q1"], "positive_id": ["d1"]}) == {"query": ["Q one"], "positive": ["A"]}


class TestBuildTimeErrors:
    def test_dataset_dict_rejected(self, queries: Dataset) -> None:
        with pytest.raises(TypeError, match="expected a `Dataset`"):
            resolve_ids({"query_id": DatasetDict({"train": queries})})

    def test_non_dataset_rejected(self) -> None:
        with pytest.raises(TypeError, match="must hold a `Dataset`"):
            resolve_ids({"query_id": {"query_id": ["q1"]}})

    def test_wrong_tuple_length(self, queries: Dataset) -> None:
        with pytest.raises(ValueError, match="tuple of length 1"):
            resolve_ids({"query_id": (queries,)})

    def test_unknown_explicit_id_col(self, queries: Dataset) -> None:
        with pytest.raises(ValueError, match="not a column of the lookup dataset"):
            resolve_ids({"query_id": (queries, "nope")})

    def test_unknown_explicit_value_col(self, queries: Dataset) -> None:
        with pytest.raises(ValueError, match="value column 'nope'"):
            resolve_ids({"query_id": (queries, "query_id", "nope")})

    def test_ambiguous_id_inference(self) -> None:
        two_ids = Dataset.from_dict({"document_id": ["d1"], "source_id": ["s1"], "text": ["A"]})
        with pytest.raises(ValueError, match="Cannot infer the ID column"):
            resolve_ids({"document_ids": two_ids})

    def test_no_id_column(self) -> None:
        no_id = Dataset.from_dict({"name": ["d1"], "text": ["A"]})
        with pytest.raises(ValueError, match="Cannot infer the ID column"):
            resolve_ids({"document_ids": no_id})

    def test_ambiguous_value_inference(self) -> None:
        wide = Dataset.from_dict({"document_id": ["d1"], "title": ["T"], "text": ["A"]})
        with pytest.raises(ValueError, match="Cannot infer the value column"):
            resolve_ids({"document_ids": wide})

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_max_list_length(self, documents: Dataset, bad: int) -> None:
        with pytest.raises(ValueError, match="max_list_length must be a positive integer"):
            resolve_ids({"document_ids": documents}, max_list_length=bad)

    def test_string_keep_columns_rejected(self, documents: Dataset) -> None:
        with pytest.raises(TypeError, match="keep_columns must be a list or tuple"):
            resolve_ids({"document_ids": documents}, keep_columns="scores")

    def test_invalid_output_format_rejected(self, documents: Dataset) -> None:
        with pytest.raises(ValueError, match='output_format must be "columns" or "lists"'):
            resolve_ids({"document_ids": documents}, output_format="rows")


class TestTransformTimeBehaviour:
    def test_missing_id_raises_keyerror(self, queries: Dataset, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents})
        with pytest.raises(KeyError, match="'d999' from input column 'document_ids'"):
            transform({"document_ids": [["d1", "d999"]]})

    def test_string_encoded_list_gets_a_hint(self, documents: Dataset) -> None:
        # A string-encoded list column is looked up as one giant ID. The KeyError should point at
        # the real cause instead of leaving the user with a baffling missing-ID message.
        transform = resolve_ids({"document_ids": documents})
        with pytest.raises(KeyError, match="string-encoded list"):
            transform({"document_ids": ["['d1', 'd2']"]})

    def test_duplicate_lookup_ids_warn_and_resolve_to_last(self, caplog) -> None:
        duplicated = Dataset.from_dict({"document_id": ["d1", "d1"], "text": ["first", "SECOND"]})
        with caplog.at_level(logging.WARNING):
            transform = resolve_ids({"document_id": duplicated})
        assert any("duplicate" in record.message for record in caplog.records)
        assert transform({"document_id": ["d1"]}) == {"document": ["SECOND"]}

    def test_same_output_name_collision_raises(self, documents: Dataset) -> None:
        transform = resolve_ids({"doc_id": documents, "doc_idx": documents})
        with pytest.raises(ValueError, match="produced more than once"):
            transform({"doc_id": ["d1"], "doc_idx": ["d2"]})

    def test_single_and_list_with_same_base_coexist(self, documents: Dataset) -> None:
        # doc_id -> "doc" and doc_ids -> "doc_1", ...: distinct output names, no collision.
        transform = resolve_ids({"doc_id": documents, "doc_ids": documents})
        assert transform({"doc_id": ["d1"], "doc_ids": [["d2", "d3"]]}) == {
            "doc": ["A"],
            "doc_1": ["B"],
            "doc_2": ["C"],
        }

    def test_expanded_position_collision_raises_at_transform(self, documents: Dataset) -> None:
        # "document_1" resolves to "document_1", colliding with the first expanded position of
        # "document_ids".
        transform = resolve_ids({"document_ids": documents, "document_1": documents})
        with pytest.raises(ValueError, match="produced more than once"):
            transform({"document_ids": [["d1", "d2"]], "document_1": ["d3"]})

    def test_nested_plural_collision_raises_at_transform(self, documents: Dataset) -> None:
        # In nested mode, "documents" (no ID suffix) and "document_ids" both emit the plural "documents".
        transform = resolve_ids({"document_ids": documents, "documents": documents}, output_format="lists")
        with pytest.raises(ValueError, match="produced more than once"):
            transform({"document_ids": [["d1"]], "documents": ["d2"]})

    def test_kept_column_collision_raises(self, documents: Dataset) -> None:
        # A kept column landing on a resolved output name must raise instead of silently replacing
        # the resolved values with the raw ones.
        transform = resolve_ids({"scores_id": documents}, keep_columns=["scores"])
        with pytest.raises(ValueError, match="produced more than once"):
            transform({"scores_id": ["d1"], "scores": [[1.0]]})

    def test_empty_id_lists_raise_in_columns_mode(self, documents: Dataset) -> None:
        transform = resolve_ids({"document_ids": documents})
        with pytest.raises(ValueError, match="empty ID lists"):
            transform({"document_ids": [[], []]})


class TestRobustness:
    def test_transform_is_picklable(self, queries: Dataset, documents: Dataset) -> None:
        # DataLoader workers pickle the dataset (including its transform) on spawn-start platforms,
        # so dataloader_num_workers > 0 requires this.
        import pickle

        train = Dataset.from_dict({"query_id": ["q1"], "document_ids": [["d1", "d2"]]})
        train.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}))
        restored = pickle.loads(pickle.dumps(train))
        assert restored[0] == train[0]

    def test_torch_formatted_lookup_resolves(self, documents: Dataset) -> None:
        # A torch-formatted lookup would otherwise produce unhashable 0-d tensor index keys.
        formatted = Dataset.from_dict({"document_id": [1, 2], "text": ["A", "B"]}).with_format("torch")
        transform = resolve_ids({"document_ids": formatted})
        assert transform({"document_ids": [[2, 1]]}) == {"document_1": ["B"], "document_2": ["A"]}

    def test_shared_index_survives_format_normalization(self, documents: Dataset) -> None:
        # Two entries sharing one lookup dataset still share one ID index (keyed on the original
        # object, not the format-normalized copy).
        transform = resolve_ids({"positive_id": documents, "negative_id": documents})
        indices = [resolution[1] for resolution in transform.keywords["resolutions"].values()]
        assert indices[0] is indices[1]

    @pytest.mark.parametrize("dataset_format", ["pandas", "numpy"])
    def test_formatted_train_dataset_raises_clear_error(
        self, queries: Dataset, documents: Dataset, dataset_format: str
    ) -> None:
        # map hands the transform batches in the dataset's current format, which resolve_ids does
        # not support: fail loud with the with_format(None) remedy
        train = Dataset.from_dict({"query_id": ["q1"], "document_ids": [["d1", "d2"]], "scores": [[0.9, 0.1]]})
        transform = resolve_ids({"query_id": queries, "document_ids": documents})
        with pytest.raises(TypeError, match="with_format"):
            train.with_format(dataset_format).map(transform, batched=True)

    def test_set_transform_ignores_prior_format(self, queries: Dataset, documents: Dataset) -> None:
        # set_transform replaces any prior format, so this path never sees formatted batches.
        train = Dataset.from_dict({"query_id": ["q1"], "document_ids": [["d1", "d2"]], "scores": [[0.9, 0.1]]})
        train = train.with_format("numpy")
        train.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}))
        assert train[0] == {"query": "Q one", "document_1": "A", "document_2": "B", "scores": [0.9, 0.1]}

    def test_formatted_train_dataset_raises_without_list_ids(self, queries: Dataset, documents: Dataset) -> None:
        # Single-ID columns used to slip through: numpy strings hash like str, so the lookups
        # succeeded and the format was only rejected when an ID list turned up.
        train = Dataset.from_dict({"query_id": ["q1"], "positive_id": ["d1"]})
        transform = resolve_ids({"query_id": queries, "positive_id": documents})
        with pytest.raises(TypeError, match="with_format"):
            train.with_format("numpy").map(transform, batched=True)

    def test_max_list_length_truncates_tensor_labels_in_step(self, queries: Dataset, documents: Dataset) -> None:
        # A torch-formatted label column arrives as a tensor, which resolves fine but must still be
        # capped, else the scores outnumber the expanded document columns.
        train = Dataset.from_dict(
            {"query_id": ["q1"], "document_ids": [["d1", "d2", "d3"]], "scores": [[0.9, 0.5, 0.1]]}
        )
        transform = resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=2)
        mapped = train.with_format("torch").map(transform, batched=True, remove_columns=train.column_names)
        assert sorted(mapped.column_names) == ["document_1", "document_2", "query", "scores"]
        assert len(mapped[0]["scores"]) == 2
