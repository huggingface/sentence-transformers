from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sentence_transformers.util import is_datasets_available


class _GenericNanoDatasetMixin:
    def _initialize_generic_nano_state(
        self,
        *,
        dataset_id: str,
        dataset_name_to_human_readable: Mapping[str, str] | None,
        split_prefix: str,
        strict_dataset_name_validation: bool,
        auto_expand_splits_when_dataset_names_none: bool,
        name: str | None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_name_to_human_readable = (
            dict(dataset_name_to_human_readable) if dataset_name_to_human_readable else None
        )
        self.split_prefix = split_prefix
        self.strict_dataset_name_validation = strict_dataset_name_validation
        self.auto_expand_splits_when_dataset_names_none = auto_expand_splits_when_dataset_names_none
        self._configured_name = name
        self.evaluator_name = name or dataset_id.split("/")[-1]
        self._subset_to_split_names_cache: dict[str, list[str]] = {}

    def _resolve_dataset_names(self, dataset_names: list[str] | None) -> list[str]:
        if dataset_names is not None:
            return dataset_names
        if not self.auto_expand_splits_when_dataset_names_none:
            raise ValueError("dataset_names cannot be None when auto split expansion is disabled.")
        return self._get_available_splits("queries")

    def _normalize_prompt_mapping(
        self,
        prompt_mapping: str | dict[str, str] | None,
        dataset_names: list[str],
    ) -> str | dict[str, str] | None:
        if prompt_mapping is None or isinstance(prompt_mapping, str):
            return prompt_mapping

        lower_to_prompt = {key.lower(): value for key, value in prompt_mapping.items()}
        normalized_prompt_mapping = {}
        for dataset_name in dataset_names:
            prompt = prompt_mapping.get(dataset_name)
            if prompt is None:
                prompt = lower_to_prompt.get(dataset_name.lower())
            if prompt is not None:
                normalized_prompt_mapping[dataset_name] = prompt
        return normalized_prompt_mapping

    def _is_known_split_name(self, dataset_name: str) -> bool:
        return dataset_name in self._get_available_splits("queries")

    def _get_split_name(self, dataset_name: str) -> str:
        if self.dataset_name_to_human_readable is None:
            return dataset_name
        if dataset_name in self.dataset_name_to_human_readable:
            return f"{self.split_prefix}{self.dataset_name_to_human_readable[dataset_name]}"
        lowered = dataset_name.lower()
        if lowered in self.dataset_name_to_human_readable:
            return f"{self.split_prefix}{self.dataset_name_to_human_readable[lowered]}"
        if not self.strict_dataset_name_validation:
            return dataset_name
        if self._is_known_split_name(dataset_name):
            return dataset_name
        raise ValueError(
            f"Dataset '{dataset_name}' does not exist in dataset_name_to_human_readable mapping. "
            f"Available dataset names are: {list(self.dataset_name_to_human_readable.keys())}"
        )

    def _validate_dataset_names(self) -> None:
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")
        if self.dataset_name_to_human_readable is None or not self.strict_dataset_name_validation:
            return

        missing_datasets = []
        for dataset_name in self.dataset_names:
            if dataset_name in self.dataset_name_to_human_readable:
                continue
            if dataset_name.lower() in self.dataset_name_to_human_readable:
                continue
            if self._is_known_split_name(dataset_name):
                continue
            missing_datasets.append(dataset_name)
        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} do not exist in dataset_name_to_human_readable mapping. "
                f"Available dataset names are: {list(self.dataset_name_to_human_readable.keys())}"
            )

    def _get_required_subset_names_for_split_validation(self) -> list[str]:
        return ["corpus", "queries", "qrels"]

    def _validate_mapping_splits(self) -> None:
        if self.dataset_name_to_human_readable is None:
            return
        for dataset_name in self.dataset_names:
            split_name = self._get_split_name(dataset_name)
            for subset_name in self._get_required_subset_names_for_split_validation():
                self._validate_split_exists(dataset_name, subset_name, split_name)

    def _get_available_splits(self, subset: str) -> list[str]:
        if subset in self._subset_to_split_names_cache:
            return self._subset_to_split_names_cache[subset]
        if not is_datasets_available():
            raise ValueError(f"datasets is not available. Please install it to use the {type(self).__name__}.")
        from datasets import get_dataset_split_names

        try:
            split_names = get_dataset_split_names(self.dataset_id, subset)
        except Exception as exc:
            raise ValueError(
                f"Could not list split names for subset '{subset}' from dataset '{self.dataset_id}'."
            ) from exc

        if not split_names:
            raise ValueError(f"No split names were found for subset '{subset}' in dataset '{self.dataset_id}'.")
        self._subset_to_split_names_cache[subset] = list(split_names)
        return self._subset_to_split_names_cache[subset]

    def _validate_split_exists(self, dataset_name: str, subset: str, split_name: str) -> None:
        available_splits = self._get_available_splits(subset)
        if split_name not in available_splits:
            raise ValueError(
                f"Dataset '{dataset_name}' maps to split '{split_name}', but it does not exist in subset '{subset}' "
                f"for dataset '{self.dataset_id}'. Available splits: {available_splits}"
            )

    def _get_generic_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "dataset_name_to_human_readable": self.dataset_name_to_human_readable,
            "split_prefix": self.split_prefix,
            "strict_dataset_name_validation": self.strict_dataset_name_validation,
            "auto_expand_splits_when_dataset_names_none": self.auto_expand_splits_when_dataset_names_none,
        }
        if self._configured_name is not None:
            config_dict["name"] = self._configured_name
        for key in ["truncate_dim", "query_prompts", "corpus_prompts"]:
            value = getattr(self, key, None)
            if value is not None:
                config_dict[key] = value
        return config_dict


class _GenericCrossEncoderNanoMixin(_GenericNanoDatasetMixin):
    def _initialize_generic_cross_encoder_state(
        self,
        *,
        dataset_id: str,
        dataset_name_to_human_readable: Mapping[str, str] | None,
        split_prefix: str,
        strict_dataset_name_validation: bool,
        auto_expand_splits_when_dataset_names_none: bool,
        name: str | None,
    ) -> None:
        self._initialize_generic_nano_state(
            dataset_id=dataset_id,
            dataset_name_to_human_readable=dataset_name_to_human_readable,
            split_prefix=split_prefix,
            strict_dataset_name_validation=strict_dataset_name_validation,
            auto_expand_splits_when_dataset_names_none=auto_expand_splits_when_dataset_names_none,
            name=name,
        )

    def _get_required_subset_names_for_split_validation(self) -> list[str]:
        return [*super()._get_required_subset_names_for_split_validation(), "bm25"]

    def _validate_retrieval_references(
        self,
        dataset_name: str,
        split_name: str,
        query_mapping: dict[str, str],
        corpus_mapping: dict[str, str],
        qrels_mapping: dict[str, set[str]],
        retrieved: Any,
    ) -> None:
        missing_query_ids_in_qrels = [query_id for query_id in qrels_mapping if query_id not in query_mapping]
        missing_positive_ids = sorted(
            {
                corpus_id
                for corpus_ids in qrels_mapping.values()
                for corpus_id in corpus_ids
                if corpus_id not in corpus_mapping
            }
        )

        missing_query_ids_in_candidates: set[str] = set()
        missing_qrels_for_candidates: set[str] = set()
        missing_retrieved_ids: set[str] = set()
        for sample in retrieved:
            query_id = sample["query-id"]
            if query_id not in query_mapping:
                missing_query_ids_in_candidates.add(query_id)
            if query_id not in qrels_mapping:
                missing_qrels_for_candidates.add(query_id)
            for document_id in sample["corpus-ids"]:
                if document_id not in corpus_mapping:
                    missing_retrieved_ids.add(document_id)

        if any(
            [
                missing_query_ids_in_qrels,
                missing_positive_ids,
                missing_query_ids_in_candidates,
                missing_qrels_for_candidates,
                missing_retrieved_ids,
            ]
        ):
            error_details: list[str] = []
            if missing_query_ids_in_qrels:
                error_details.append(f"qrels references unknown query IDs: {sorted(missing_query_ids_in_qrels)[:5]}")
            if missing_positive_ids:
                error_details.append(f"qrels references unknown corpus IDs: {missing_positive_ids[:5]}")
            if missing_query_ids_in_candidates:
                error_details.append(
                    f"candidate subset references unknown query IDs: {sorted(missing_query_ids_in_candidates)[:5]}"
                )
            if missing_qrels_for_candidates:
                error_details.append(
                    f"candidate subset contains queries missing in qrels: {sorted(missing_qrels_for_candidates)[:5]}"
                )
            if missing_retrieved_ids:
                error_details.append(
                    f"candidate subset references unknown corpus IDs: {sorted(missing_retrieved_ids)[:5]}"
                )
            raise ValueError(
                f"Inconsistent IDs found for dataset '{dataset_name}' split '{split_name}' in '{self.dataset_id}'. "
                + " | ".join(error_details)
            )

    def _get_generic_cross_encoder_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "dataset_name_to_human_readable": self.dataset_name_to_human_readable,
            "split_prefix": self.split_prefix,
            "strict_dataset_name_validation": self.strict_dataset_name_validation,
            "auto_expand_splits_when_dataset_names_none": self.auto_expand_splits_when_dataset_names_none,
        }
        if self._configured_name is not None:
            config_dict["name"] = self._configured_name
        return config_dict
