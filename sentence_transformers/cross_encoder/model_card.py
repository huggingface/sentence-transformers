from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import ModelCard

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.model import CrossEncoder


class CrossEncoderModelCardCallback(BaseModelCardCallback):
    def __init__(self, default_args_dict: dict[str, Any]) -> None:
        super().__init__(default_args_dict)


@dataclass
class CrossEncoderModelCardData(BaseModelCardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "CrossEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/ce-mpnet-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search and paraphrase mining".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "cross-encoder"].
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
            Defaults to False.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = CrossEncoder(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=CrossEncoderModelCardData(
        ...         model_id="tomaarsen/ce-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    task_name: str = field(default=None)
    tags: list[str] | None = field(
        default_factory=lambda: [
            "sentence-transformers",
            "cross-encoder",
            "reranker",
        ]
    )

    # Automatically filled by `CrossEncoderModelCardCallback` and the Trainer directly
    predict_example: list[list[str]] | None = field(default=None, init=False)
    ir_model: bool | None = field(default=True, init=False, repr=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)

    # Passed via `register_model` only
    model: CrossEncoder | None = field(default=None, init=False, repr=False)

    def set_widget_examples(self, dataset: Dataset | DatasetDict) -> None:
        """
        We don't set widget examples, but only load the prediction example.
        This is because the Hugging Face Hub doesn't currently have a Sentence Ranking
        or Text Classification widget that accepts pairs, which is what CrossEncoder
        models require.
        """
        if isinstance(dataset, DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            # We can't set widget examples from an IterableDataset without losing data
            return

        if len(dataset) == 0:
            return

        columns = [
            column
            for column, feature in dataset.features.items()
            if (isinstance(feature, Value) and feature.dtype in {"string", "large_string"})
            or (
                hasattr(feature, "feature")
                and isinstance(feature.feature, Value)
                and feature.feature.dtype in {"string", "large_string"}
            )
        ]
        if len(columns) < 2:
            return

        query_column = columns[0]
        answer_column = columns[1]

        query_type = type(dataset[0][query_column])
        answer_type = type(dataset[0][answer_column])

        queries = dataset[:5][query_column]
        answers = dataset[:5][answer_column]

        # If the response is a list, then the first query-answer is a nice example
        if answer_type is list:
            answers = answers[0][:5]
            queries = [queries[0]] * len(answers)

        if query_type is str:
            self.predict_example = [[query, response] for query, response in zip(queries, answers)]

    def register_model(self, model) -> None:
        super().register_model(model)

        if self.task_name is None:
            self.task_name = (
                "text reranking and semantic search" if model.num_labels == 1 else "text pair classification"
            )
        if self.pipeline_tag is None:
            self.pipeline_tag = "text-ranking" if model.num_labels == 1 else "text-classification"

    # def tokenize(self, text: str | list[str], **kwargs) -> dict[str, Any]:
    #     return self.model.tokenizer(text)

    def run_usage_snippet(self) -> dict[str, Any]:
        # At the moment, we don't run the usage snippet for CrossEncoder models,
        # although we could.
        return

    def get_model_specific_metadata(self) -> dict[str, Any]:
        return {
            "model_max_length": self.model.max_length,
            "model_num_labels": self.model.num_labels,
        }


def generate_model_card(model: CrossEncoder) -> str:
    model_card = ModelCard.from_template(
        card_data=model.model_card_data, template_path=model.model_card_data.template_path, hf_emoji="🤗"
    )
    return model_card.content
