from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, UserDict, defaultdict
from copy import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from platform import python_version
from pprint import pformat
from textwrap import indent
from typing import TYPE_CHECKING, Any, Literal

import torch
import transformers
from huggingface_hub import CardData, ModelCard
from huggingface_hub import dataset_info as get_dataset_info
from huggingface_hub import model_info as get_model_info
from huggingface_hub.repocard_data import EvalResult, eval_results_to_model_index
from huggingface_hub.utils import yaml_dump
from torch import nn
from tqdm.autonotebook import tqdm
from transformers import TrainerCallback
from transformers.integrations import CodeCarbonCallback
from transformers.modelcard import make_markdown_table
from transformers.trainer_callback import TrainerControl, TrainerState

from sentence_transformers import __version__ as sentence_transformers_version
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.util import fullname, is_accelerate_available, is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.base.evaluation.SentenceEvaluator import SentenceEvaluator
    from sentence_transformers.base.model import BaseModel
    from sentence_transformers.base.trainer import BaseTrainer


class BaseModelCardCallback(TrainerCallback):
    def __init__(self, default_args_dict: dict[str, Any]) -> None:
        super().__init__()
        self.default_args_dict = default_args_dict

    def on_init_end(
        self,
        args: BaseTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: BaseModel,
        trainer: BaseTrainer,
        **kwargs,
    ) -> None:
        model.model_card_data.add_tags("generated_from_trainer")

        # Try to set the code carbon callback if it exists
        callbacks = [
            callback for callback in trainer.callback_handler.callbacks if isinstance(callback, CodeCarbonCallback)
        ]
        if callbacks:
            model.model_card_data.code_carbon_callback = callbacks[0]

        # Try to infer the dataset "name", "id" and "revision" from the dataset cache files
        if trainer.train_dataset:
            model.model_card_data.train_datasets = model.model_card_data.extract_dataset_metadata(
                trainer.train_dataset, model.model_card_data.train_datasets, trainer.loss, "train"
            )

        if trainer.eval_dataset:
            model.model_card_data.eval_datasets = model.model_card_data.extract_dataset_metadata(
                trainer.eval_dataset, model.model_card_data.eval_datasets, trainer.loss, "eval"
            )

        losses = get_losses(trainer.loss)

        model.model_card_data.set_losses(losses)

        # Extract some meaningful examples from the evaluation or training dataset to showcase the performance
        if not model.model_card_data.widget and (dataset := trainer.eval_dataset or trainer.train_dataset):
            model.model_card_data.set_widget_examples(dataset)

    def on_train_begin(
        self,
        args: BaseTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: BaseModel,
        **kwargs,
    ) -> None:
        ignore_keys = {
            "output_dir",
            "logging_dir",
            "logging_strategy",
            "logging_first_step",
            "logging_steps",
            "evaluation_strategy",
            "eval_steps",
            "eval_delay",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "metric_for_best_model",
            "greater_is_better",
            "report_to",
            "samples_per_label",
            "show_progress_bar",
            "do_train",
            "do_eval",
            "do_test",
            "run_name",
            "hub_token",
            "push_to_hub_token",
        }
        args_dict = args.to_dict()
        model.model_card_data.all_hyperparameters = {
            key: value for key, value in args_dict.items() if key not in ignore_keys
        }
        model.model_card_data.non_default_hyperparameters = {
            key: value
            for key, value in args_dict.items()
            if key not in ignore_keys and key in self.default_args_dict and value != self.default_args_dict[key]
        }

    def on_evaluate(
        self,
        args: BaseTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: BaseModel,
        metrics: dict[str, float],
        **kwargs,
    ) -> None:
        loss_dict = {
            " ".join(key.split("_")[1:]): metrics[key]
            for key in metrics
            if key.startswith("eval_") and key.endswith("_loss")
        }
        if len(loss_dict) == 1 and "loss" in loss_dict:
            loss_dict = {"Validation Loss": loss_dict["loss"]}
        if (
            model.model_card_data.training_logs
            and model.model_card_data.training_logs[-1]["Step"] == state.global_step
        ):
            model.model_card_data.training_logs[-1].update(loss_dict)
        else:
            model.model_card_data.training_logs.append(
                {
                    "Epoch": state.epoch,
                    "Step": state.global_step,
                    **loss_dict,
                }
            )

    def on_log(
        self,
        args: BaseTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: BaseModel,
        logs: dict[str, float],
        **kwargs,
    ) -> None:
        keys = {"loss"} & set(logs)
        if keys:
            if (
                model.model_card_data.training_logs
                and model.model_card_data.training_logs[-1]["Step"] == state.global_step
            ):
                model.model_card_data.training_logs[-1]["Training Loss"] = logs[keys.pop()]
            else:
                model.model_card_data.training_logs.append(
                    {
                        "Epoch": state.epoch,
                        "Step": state.global_step,
                        "Training Loss": logs[keys.pop()],
                    }
                )


YAML_FIELDS = [
    "language",
    "license",
    "library_name",
    "tags",
    "datasets",
    "metrics",
    "pipeline_tag",
    "widget",
    "model-index",
    "co2_eq_emissions",
    "base_model",
]
IGNORED_FIELDS = ["model", "trainer", "eval_results_dict"]


def get_versions() -> dict[str, Any]:
    versions = {
        "python": python_version(),
        "sentence_transformers": sentence_transformers_version,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    if is_accelerate_available():
        from accelerate import __version__ as accelerate_version

        versions["accelerate"] = accelerate_version
    if is_datasets_available():
        from datasets import __version__ as datasets_version

        versions["datasets"] = datasets_version
    from tokenizers import __version__ as tokenizers_version

    versions["tokenizers"] = tokenizers_version

    return versions


def format_log(value: float | int | str) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def get_losses(loss: nn.Module | dict[nn.Module]) -> list[nn.Module]:
    if isinstance(loss, dict):
        losses = list(loss.values())
    else:
        losses = [loss]
    # Some losses are known to use other losses internally
    # So, verify for `loss` attributes in the losses
    loss_idx = 0
    while loss_idx < len(losses):
        loss = losses[loss_idx]
        if hasattr(loss, "loss") and loss.loss not in losses:
            losses.append(loss.loss)
        if hasattr(loss, "document_regularizer") and loss.document_regularizer not in losses:
            losses.append(loss.document_regularizer)
        if hasattr(loss, "query_regularizer") and loss.query_regularizer not in losses:
            losses.append(loss.query_regularizer)
        loss_idx += 1
    return losses


@dataclass
class BaseModelCardData(CardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SentenceTransformer based on microsoft/mpnet-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/sbert-mpnet-base-allnli".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "sentence-similarity", "feature-extraction"].
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
            Defaults to False.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset,
            and compute their similarities. Defaults to True.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = SentenceTransformer(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=SentenceTransformerModelCardData(
        ...         model_id="tomaarsen/sbert-mpnet-base-allnli",
        ...         train_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         eval_datasets=[{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}],
        ...         license="apache-2.0",
        ...         language="en",
        ...     ),
        ... )
    """

    # Potentially provided by the user
    language: str | list[str] | None = field(default_factory=list)
    license: str | None = None
    model_name: str | None = None
    model_id: str | None = None
    train_datasets: list[dict[str, str]] = field(default_factory=list)
    eval_datasets: list[dict[str, str]] = field(default_factory=list)
    task_name: str = (
        "semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more"
    )
    tags: list[str] | None = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
            "dense",
        ]
    )
    local_files_only: bool = False
    generate_widget_examples: bool = field(default=True)

    # Automatically filled by `SentenceTransformerModelCardCallback` and the Trainer directly
    base_model: str | None = field(default=None, init=False)
    base_model_revision: str | None = field(default=None, init=False)
    non_default_hyperparameters: dict[str, Any] = field(default_factory=dict, init=False)
    all_hyperparameters: dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: dict[SentenceEvaluator, dict[str, Any]] | None = field(default_factory=dict, init=False)
    training_logs: list[dict[str, float]] = field(default_factory=list, init=False)
    widget: list[dict[str, str]] = field(default_factory=list, init=False)
    predict_example: list[str] | None = field(default=None, init=False)
    label_example_list: list[dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: CodeCarbonCallback | None = field(default=None, init=False)
    citations: dict[str, str] = field(default_factory=dict, init=False)
    best_model_step: int | None = field(default=None, init=False)
    datasets: list[str] = field(default_factory=list, init=False, repr=False)
    similarities: str | None = field(default=None, init=False, repr=False)

    # Utility fields
    first_save: bool = field(default=True, init=False)
    widget_step: int = field(default=-1, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default="sentence-similarity", init=False)
    library_name: str = field(default="sentence-transformers", init=False)
    version: dict[str, str] = field(default_factory=get_versions, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)

    # Passed via `register_model` only
    model: BaseModel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # We don't want to save "ignore_metadata_errors" in our Model Card
        if isinstance(self.language, str):
            self.language = [self.language]

        self.train_datasets = self.validate_datasets(self.train_datasets)
        self.eval_datasets = self.validate_datasets(self.eval_datasets)

        if self.model_id and self.model_id.count("/") != 1:
            logger.warning(
                f"The provided {self.model_id!r} model ID should include the organization or user,"
                ' such as "tomaarsen/mpnet-base-nli-matryoshka". Setting `model_id` to None.'
            )
            self.model_id = None

    def validate_datasets(
        self, dataset_list: list[dict[str, Any]], infer_languages: bool | None = None
    ) -> list[dict[str, Any]]:
        """
        Validate (i.e. check if the dataset IDs exist on the Hub) and process a list of dataset dictionaries.

        Args:
            dataset_list (list[dict[str, Any]]): List of dataset metadata dictionaries.
            infer_languages (bool | None, optional): Whether to infer languages from the dataset information.
                If None (default), languages will be inferred only if `self.language` is empty.

        Returns:
            list[dict[str, Any]]: The validated and possibly updated list of dataset dictionaries.
        """
        if infer_languages is None:
            # Infer languages if they're not already defined
            infer_languages = not self.language
        output_dataset_list = []
        for dataset in dataset_list:
            if "name" not in dataset:
                if "id" in dataset:
                    dataset["name"] = dataset["id"]

            if "id" in dataset and not self.local_files_only:
                # Try to determine the language from the dataset on the Hub
                try:
                    info = get_dataset_info(dataset["id"])
                except Exception:
                    logger.warning(
                        f"The dataset `id` {dataset['id']!r} does not exist on the Hub. Setting the `id` to None."
                    )
                    del dataset["id"]
                else:
                    if info.cardData and infer_languages and "language" in info.cardData:
                        dataset_language = info.cardData.get("language")
                        if dataset_language is not None:
                            if isinstance(dataset_language, str):
                                dataset_language = [dataset_language]
                            for language in dataset_language:
                                if language not in self.language:
                                    self.language.append(language)

                    # Track dataset IDs for the metadata
                    if info.id not in self.datasets:
                        self.datasets.append(info.id)

            output_dataset_list.append(dataset)
        return output_dataset_list

    def set_losses(self, losses: list[nn.Module]) -> None:
        citations = {
            "Sentence Transformers": """
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
"""
        }
        for loss in losses:
            try:
                citations[loss.__class__.__name__] = loss.citation
            except Exception:
                pass
        inverted_citations = defaultdict(list)
        for loss, citation in citations.items():
            inverted_citations[citation].append(loss)

        def join_list(losses: list[str]) -> str:
            if len(losses) > 1:
                return ", ".join(losses[:-1]) + " and " + losses[-1]
            return losses[0]

        self.citations = {join_list(losses): citation for citation, losses in inverted_citations.items()}
        self.add_tags([f"loss:{loss}" for loss in {loss.__class__.__name__: loss for loss in losses}])

    def set_best_model_step(self, step: int) -> None:
        self.best_model_step = step

    def set_widget_examples(self, dataset: Dataset | DatasetDict) -> None:
        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            # We can't set widget examples from an IterableDataset without losing data
            return

        if isinstance(dataset, Dataset):
            dataset = DatasetDict(dataset=dataset)

        self.widget = []
        # Pick 5 random datasets to generate widget examples from
        dataset_names = Counter(random.choices(list(dataset.keys()), k=5))
        num_samples_to_check = 1000
        for dataset_name, num_samples in tqdm(
            dataset_names.items(), desc="Computing widget examples", unit="example", leave=False
        ):
            if isinstance(dataset[dataset_name], IterableDataset):
                # We can't set widget examples from an IterableDataset without losing data
                continue

            # Sample 1000 examples from the dataset, sort them by length, and pick the shortest examples as the core
            # examples for the widget
            columns = [
                column
                for column, feature in dataset[dataset_name].features.items()
                if isinstance(feature, dict)
                or (isinstance(feature, Value) and feature.dtype in {"string", "large_string"})
            ]
            str_dataset = dataset[dataset_name].select_columns(columns)
            dataset_size = len(str_dataset)
            if dataset_size == 0:
                continue

            lengths = {}
            for idx, sample in enumerate(
                str_dataset.select(random.sample(range(dataset_size), k=min(num_samples_to_check, dataset_size)))
            ):
                lengths[idx] = sum(len(value) for key, value in sample.items() if key != "dataset_name")

            indices, _ = zip(*sorted(lengths.items(), key=lambda x: x[1]))
            target_indices, backup_indices = indices[:num_samples], list(indices[num_samples:][::-1])

            # We want 4 texts, so we take texts from the backup indices, short texts first
            for idx in target_indices:
                # This is anywhere between 1 and n texts
                sentences = [sentence for key, sentence in str_dataset[idx].items() if key != "dataset_name"]
                while len(sentences) < 4 and backup_indices:
                    backup_idx = backup_indices.pop()
                    backup_sample = [
                        sentence for key, sentence in str_dataset[backup_idx].items() if key != "dataset_name"
                    ]
                    if len(backup_sample) == 1:
                        # If there is only one text in the backup sample, we take it
                        sentences.extend(backup_sample)
                    else:
                        # Otherwise we prefer the 2nd text: the 1st can be another query
                        sentences.append(backup_sample[1])

                if len(sentences) < 4:
                    continue

                # When training with a Router (or Asym) module, you might be using backwards compatible training,
                # i.e. with a dictionary with a mapping of Router keys to texts, so let's grab the texts
                sentences = [
                    list(sentence.values())[0] if isinstance(sentence, dict) else sentence for sentence in sentences
                ]

                if self.pipeline_tag == "sentence-similarity":
                    self.widget.append(
                        {
                            "source_sentence": sentences[0],
                            "sentences": random.sample(sentences[1:], k=len(sentences) - 1),
                        }
                    )
                else:
                    # If we have e.g. feature-extraction, we just want individual sentences
                    self.widget.append({"text": random.choice(sentences)})
                self.predict_example = sentences[:4]

    def set_evaluation_metrics(
        self, evaluator: SentenceEvaluator, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

        self.eval_results_dict[evaluator] = copy(metrics)

        # If the evaluator has a primary metric and we have a trainer, then add the primary metric to the training logs
        if hasattr(evaluator, "primary_metric") and (primary_metrics := evaluator.primary_metric):
            if isinstance(evaluator, SequentialEvaluator):
                primary_metrics = [sub_evaluator.primary_metric for sub_evaluator in evaluator.evaluators]
            elif isinstance(primary_metrics, str):
                primary_metrics = [primary_metrics]

            training_log_metrics = {key: value for key, value in metrics.items() if key in primary_metrics}

            if self.training_logs and self.training_logs[-1]["Step"] == step:
                self.training_logs[-1].update(training_log_metrics)
            else:
                self.training_logs.append(
                    {
                        "Epoch": epoch,
                        "Step": step,
                        **training_log_metrics,
                    }
                )

    def set_label_examples(self, dataset: Dataset) -> None:
        num_examples_per_label = 3
        examples = defaultdict(list)
        finished_labels = set()
        for sample in dataset:
            text = sample["text"]
            label = sample["label"]
            if label not in finished_labels:
                examples[label].append(f"<li>{repr(text)}</li>")
                if len(examples[label]) >= num_examples_per_label:
                    finished_labels.add(label)
            if len(finished_labels) == self.num_classes:
                break
        self.label_example_list = [
            {
                "Label": self.model.labels[label] if self.model.labels and isinstance(label, int) else label,
                "Examples": "<ul>" + "".join(example_set) + "</ul>",
            }
            for label, example_set in examples.items()
        ]

    def infer_datasets(self, dataset: Dataset | DatasetDict, dataset_name: str | None = None) -> list[dict[str, str]]:
        if isinstance(dataset, DatasetDict):
            return [
                dataset
                for dataset_name, sub_dataset in dataset.items()
                for dataset in self.infer_datasets(sub_dataset, dataset_name=dataset_name)
            ]

        # Ignore the dataset name if it is a default name from the FitMixin backwards compatibility
        if dataset_name and re.match(r"_dataset_\d+", dataset_name):
            dataset_name = None

        dataset_output = {
            "name": dataset_name or dataset.info.dataset_name,
            "split": str(dataset.split),
        }
        if dataset.info.splits and dataset.split in dataset.info.splits:
            dataset_output["size"] = dataset.info.splits[dataset.split].num_examples

        # The download checksums seems like a fairly safe way to extract the dataset ID and revision
        # for iterable datasets as well as regular datasets from the Hub
        if checksums := dataset.download_checksums:
            source = list(checksums.keys())[0]
            if source.startswith("hf://datasets/") and "@" in source:
                source_parts = source[len("hf://datasets/") :].split("@")
                dataset_output["id"] = source_parts[0]
                if (revision := source_parts[1].split("/")[0]) and len(revision) == 40:
                    dataset_output["revision"] = revision

        return [dataset_output]

    def preprocess(self, text: str | list[str], **kwargs) -> dict[str, Any]:
        try:
            return self.model.preprocess(text, **kwargs)
        except TypeError:
            # Fallback for backwards compatibility with modules that don't yet support kwargs
            return self.model.preprocess(text)

    def compute_dataset_metrics(
        self,
        dataset: Dataset | IterableDataset | None,
        dataset_info: dict[str, Any],
        loss: dict[str, nn.Module] | nn.Module | None,
    ) -> dict[str, str]:
        """
        Given a dataset, compute the following:
        * Dataset Size
        * Dataset Columns
        * Dataset Stats
            - Strings: min, mean, max word count/token length
            - Integers: Counter() instance
            - Floats: min, mean, max range
            - List: number of elements or min, mean, max number of elements
        * 3 Example samples
        * Loss function name
            - Loss function config
        """
        if not dataset:
            return {}

        if isinstance(dataset, Dataset):
            # Size might already be defined, but `len(dataset)` is more reliable
            dataset_info["size"] = len(dataset)
        dataset_info["columns"] = [f"<code>{column}</code>" for column in dataset.column_names]
        dataset_info["stats"] = {}
        if isinstance(dataset, Dataset):
            for column in dataset.column_names:
                subsection = dataset[:1000][column]
                first = subsection[0]
                if isinstance(first, str):
                    tokenized = self.preprocess(subsection, task="document")
                    if isinstance(tokenized, (dict, UserDict)) and "attention_mask" in tokenized:
                        lengths = tokenized["attention_mask"].sum(dim=1).tolist()
                        suffix = "tokens"
                    else:
                        lengths = [len(sentence) for sentence in subsection]
                        suffix = "characters"
                    dataset_info["stats"][column] = {
                        "dtype": "string",
                        "data": {
                            "min": f"{round(min(lengths), 2)} {suffix}",
                            "mean": f"{round(sum(lengths) / len(lengths), 2)} {suffix}",
                            "max": f"{round(max(lengths), 2)} {suffix}",
                        },
                    }
                elif isinstance(first, (int, bool)):
                    counter = Counter(subsection)
                    dataset_info["stats"][column] = {
                        "dtype": "int",
                        "data": {
                            key: f"{'~' if len(counter) > 1 else ''}{counter[key] / len(subsection):.2%}"
                            for key in sorted(counter)
                        },
                    }
                elif isinstance(first, float):
                    dataset_info["stats"][column] = {
                        "dtype": "float",
                        "data": {
                            "min": round(min(subsection), 2),
                            "mean": round(sum(subsection) / len(subsection), 2),
                            "max": round(max(subsection), 2),
                        },
                    }
                elif isinstance(first, list):
                    counter = Counter([len(lst) for lst in subsection])
                    if len(counter) == 1:
                        dataset_info["stats"][column] = {
                            "dtype": "list",
                            "data": {
                                "size": f"{len(first)} elements",
                            },
                        }
                    else:
                        dataset_info["stats"][column] = {
                            "dtype": "list",
                            "data": {
                                "min": f"{min(counter)} elements",
                                "mean": f"{sum(counter) / len(counter):.2f} elements",
                                "max": f"{max(counter)} elements",
                            },
                        }
                else:
                    dataset_info["stats"][column] = {"dtype": fullname(first), "data": {}}

            def to_html_list(data: dict):
                return "<ul><li>" + "</li><li>".join(f"{key}: {value}" for key, value in data.items()) + "</li></ul>"

            stats_lines = [
                {"": "type", **{key: value["dtype"] for key, value in dataset_info["stats"].items()}},
                {"": "details", **{key: to_html_list(value["data"]) for key, value in dataset_info["stats"].items()}},
            ]
            dataset_info["stats_table"] = indent(make_markdown_table(stats_lines).replace("-:|", "--|"), "  ")

            dataset_info["examples"] = dataset[:3]
            num_samples = len(dataset_info["examples"][list(dataset_info["examples"])[0]])
            examples_lines = []
            for sample_idx in range(num_samples):
                columns = {}
                for column in dataset.column_names:
                    value = dataset_info["examples"][column][sample_idx]
                    # If the value is a long list, truncate it
                    if isinstance(value, list) and len(value) > 5:
                        value = str(value[:5])[:-1] + ", ...]"
                    # If the value is a really long string, truncate it
                    if isinstance(value, str) and len(value) > 1000:
                        value = value[:1000] + "..."
                    # Avoid newlines and | in the table
                    value = str(value).replace("\n", "<br>").replace("|", "\\|")
                    columns[column] = f"<code>{value}</code>"
                examples_lines.append(columns)
            dataset_info["examples_table"] = indent(make_markdown_table(examples_lines).replace("-:|", "--|"), "  ")

        dataset_info["loss"] = {
            "fullname": fullname(loss),
        }
        if hasattr(loss, "get_config_dict"):
            config = loss.get_config_dict()

            def format_config_value(value: Any) -> str:
                if not isinstance(value, nn.Module):
                    return value
                module_name = value.__class__.__name__
                module_args_str = []

                # E.g. SentenceTransformer, SparseEncoder, etc.
                if hasattr(value, "model_card_data") and hasattr(value.model_card_data, "base_model"):
                    module_args_str.append(repr(value.model_card_data.base_model))
                if hasattr(value, "trust_remote_code") and value.trust_remote_code:
                    module_args_str.append("trust_remote_code=True")
                # E.g. MultipleNegativesRankingLoss, CosineSimilarityLoss, etc.
                if hasattr(value, "get_config_dict"):
                    for key, val in value.get_config_dict().items():
                        module_args_str.append(f"{key}={repr(val)}")

                if module_args_str:
                    return f"{module_name}({', '.join(module_args_str)})"
                return module_name

            config = {key: format_config_value(value) for key, value in config.items()}

            try:
                str_config = json.dumps(config, indent=4)
            except TypeError:
                str_config = pformat(config, indent=4)
            dataset_info["loss"]["config_code"] = indent(f"```json\n{str_config}\n```", "  ")
        return dataset_info

    def extract_dataset_metadata(
        self,
        dataset: Dataset | DatasetDict,
        dataset_metadata: list[dict[str, Any]],
        loss: nn.Module | dict[str, nn.Module],
        dataset_type: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        if dataset:
            if dataset_metadata and (
                (isinstance(dataset, DatasetDict) and len(dataset_metadata) != len(dataset))
                or (isinstance(dataset, Dataset) and len(dataset_metadata) != 1)
            ):
                logger.warning(
                    f"The number of `{dataset_type}_datasets` in the model card data does not match the number of {dataset_type} datasets in the Trainer. "
                    f"Removing the provided `{dataset_type}_datasets` from the model card data."
                )
                dataset_metadata = []

            if not dataset_metadata:
                dataset_metadata = self.infer_datasets(dataset)

            if isinstance(dataset, DatasetDict):
                dataset_metadata = [
                    self.compute_dataset_metrics(
                        dataset_value,
                        dataset_info,
                        loss[dataset_name] if isinstance(loss, dict) else loss,
                    )
                    for dataset_name, dataset_value, dataset_info in zip(
                        dataset.keys(), dataset.values(), dataset_metadata
                    )
                ]
            else:
                dataset_metadata = [self.compute_dataset_metrics(dataset, dataset_metadata[0], loss)]

        # Try to get the number of training samples
        if dataset_type == "train":
            num_training_samples = sum([metadata.get("size", 0) for metadata in dataset_metadata])
            if num_training_samples:
                self.add_tags(f"dataset_size:{num_training_samples}")

        return self.validate_datasets(dataset_metadata)

    def register_model(self, model: BaseModel) -> None:
        self.model = model

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def set_base_model(self, model_id: str, revision: str | None = None) -> None:
        # We only set the base model if we can verify that it exists on the Hub
        if self.local_files_only:
            # Don't try to get the model info if we are not allowed to access the Hub
            return False
        try:
            model_info = get_model_info(model_id)
        except Exception:
            # Getting the model info can fail for many reasons: model does not exist, no internet, outage, etc.
            return False
        self.base_model = model_info.id
        if revision is None or revision == "main":
            revision = model_info.sha
        self.base_model_revision = revision
        return True

    def set_language(self, language: str | list[str]) -> None:
        if isinstance(language, str):
            language = [language]
        self.language = language

    def set_license(self, license: str) -> None:
        self.license = license

    def add_tags(self, tags: str | list[str]) -> None:
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            if tag not in self.tags:
                self.tags.append(tag)

    def try_to_set_base_model(self) -> None:
        if (transformers_model := self.model.transformers_model) is not None:
            base_model = transformers_model.config._name_or_path
            base_model_path = Path(base_model)
            # Sometimes the name_or_path ends exactly with the model_id, e.g.
            # "C:\\Users\\tom/.cache\\torch\\sentence_transformers\\BAAI_bge-small-en-v1.5\\"
            candidate_model_ids = ["/".join(base_model_path.parts[-2:])]
            # Sometimes the name_or_path its final part contains the full model_id, with "/" replaced with a "_", e.g.
            # "/root/.cache/torch/sentence_transformers/sentence-transformers_all-mpnet-base-v2/"
            # In that case, we take the last part, split on _, and try all combinations
            # e.g. "a_b_c_d" -> ['a/b_c_d', 'a_b/c_d', 'a_b_c/d']
            splits = base_model_path.name.split("_")
            candidate_model_ids += [
                "_".join(splits[:idx]) + "/" + "_".join(splits[idx:]) for idx in range(1, len(splits))
            ]
            for model_id in candidate_model_ids:
                if self.set_base_model(model_id):
                    break

    def format_eval_metrics(self) -> dict[str, Any]:
        """Format the evaluation metrics for the model card.

        The following keys will be returned:
        - eval_metrics: A list of dictionaries containing the class name, description, dataset name, and a markdown table
          This is used to display the evaluation metrics in the model card.
        - metrics: A list of all metric keys. This is used in the model card metadata.
        - model-index: A list of dictionaries containing the task name, task type, dataset type, dataset name, metric name,
          metric type, and metric value. This is used to display the evaluation metrics in the model card metadata.
        """
        eval_metrics = []
        all_metrics = {}
        eval_results = []
        for evaluator, metrics in self.eval_results_dict.items():
            name = getattr(evaluator, "name", None)
            primary_metric = getattr(evaluator, "primary_metric", None)
            if name and all(key.startswith(name + "_") for key in metrics.keys()):
                metrics = {key[len(name) + 1 :]: value for key, value in metrics.items()}
                if primary_metric and primary_metric.startswith(name + "_"):
                    primary_metric = primary_metric[len(name) + 1 :]

            def try_to_pure_python(value: Any) -> Any:
                """Try to convert a value from a Numpy or Torch scalar to pure Python, if not already pure Python"""
                try:
                    if hasattr(value, "dtype"):
                        return value.item()
                except Exception:
                    pass
                return value

            metrics = {key: try_to_pure_python(value) for key, value in metrics.items()}

            table_lines = [
                {
                    "Metric": f"**{metric_key}**" if metric_key == primary_metric else metric_key,
                    "Value": f"**{format_log(metric_value)}**"
                    if metric_key == primary_metric
                    else format_log(metric_value),
                }
                for metric_key, metric_value in metrics.items()
            ]

            # E.g. "Binary Classification" or "Semantic Similarity"
            description = evaluator.description
            dataset_name = getattr(evaluator, "name", None)
            config_code = ""
            if hasattr(evaluator, "get_config_dict") and (config := evaluator.get_config_dict()):
                try:
                    str_config = json.dumps(config, indent=4)
                except TypeError:
                    str_config = str(config)
                config_code = indent(f"```json\n{str_config}\n```", "  ")

            eval_metrics.append(
                {
                    "class_name": fullname(evaluator),
                    "description": description,
                    "dataset_name": dataset_name,
                    "table_lines": table_lines,
                    "config_code": config_code,
                }
            )

            def try_to_float(metric_value):
                try:
                    return float(metric_value)
                except Exception:
                    pass

                if isinstance(metric_value, str) and " " in metric_value:
                    return try_to_float(metric_value.split()[0])

                return None

            eval_results.extend(
                [
                    EvalResult(
                        task_name=description,
                        task_type=description.lower().replace(" ", "-"),
                        dataset_type=dataset_name or "unknown",
                        dataset_name=dataset_name.replace("_", " ").replace("-", " ") if dataset_name else "Unknown",
                        metric_name=metric_key.replace("_", " ").title(),
                        metric_type=metric_key,
                        metric_value=metric_value_float,
                    )
                    for metric_key, metric_value in metrics.items()
                    if (metric_value_float := try_to_float(metric_value)) is not None
                ]
            )
            all_metrics.update(metrics)

        # Group eval_metrics together by class name and table_lines metrics
        grouped_eval_metrics = []
        for eval_metric in eval_metrics:
            eval_metric_mapping = {line["Metric"]: line["Value"] for line in eval_metric["table_lines"]}
            eval_metric_metrics = set(eval_metric_mapping)
            for grouped_eval_metric in grouped_eval_metrics:
                grouped_eval_metric_metrics = set(line["Metric"] for line in grouped_eval_metric["table_lines"])
                if (
                    eval_metric["class_name"] == grouped_eval_metric["class_name"]
                    and eval_metric_metrics == grouped_eval_metric_metrics
                    and eval_metric["dataset_name"] != grouped_eval_metric["dataset_name"]
                    and eval_metric["config_code"] == grouped_eval_metric["config_code"]
                ):
                    # Add the evaluation results to the existing grouped evaluation metric
                    for line in grouped_eval_metric["table_lines"]:
                        if "Value" in line:
                            line[grouped_eval_metric["dataset_name"]] = line.pop("Value")

                        line[eval_metric["dataset_name"]] = eval_metric_mapping[line["Metric"]]

                    if not isinstance(grouped_eval_metric["dataset_name"], list):
                        grouped_eval_metric["dataset_name"] = [grouped_eval_metric["dataset_name"]]
                    grouped_eval_metric["dataset_name"].append(eval_metric["dataset_name"])
                    break
            else:
                grouped_eval_metrics.append(eval_metric)

        for grouped_eval_metric in grouped_eval_metrics:
            grouped_eval_metric["table"] = make_markdown_table(grouped_eval_metric.pop("table_lines")).replace(
                "-:|", "--|"
            )

        return {
            "eval_metrics": grouped_eval_metrics,
            "metrics": list(all_metrics.keys()),
            "model-index": eval_results_to_model_index(self.model_name, eval_results),
        }

    def format_training_logs(self):
        # Get the keys from all evaluation lines
        eval_lines_keys = []
        for lines in self.training_logs:
            for key in lines.keys():
                if key not in eval_lines_keys:
                    eval_lines_keys.append(key)

        # Sort the metric columns: Epoch, Step, Training Loss, Validation Loss, Evaluator results
        def sort_metrics(key: str) -> str:
            if key == "Epoch":
                return 0
            if key == "Step":
                return 1
            if key == "Training Loss":
                return 2
            if key == "Validation Loss":
                return 3
            if key.endswith("loss"):
                return 4
            return eval_lines_keys.index(key) + 5

        sorted_eval_lines_keys = sorted(eval_lines_keys, key=sort_metrics)
        training_logs = [
            {
                key: f"**{format_log(line[key]) if key in line else '-'}**"
                if line["Step"] == self.best_model_step
                else line.get(key, "-")
                for key in sorted_eval_lines_keys
            }
            for line in self.training_logs
        ]
        eval_lines = make_markdown_table(training_logs)
        return {
            "eval_lines": eval_lines,
            "explain_bold_in_eval": "**" in eval_lines,
        }

    def run_usage_snippet(self) -> dict[str, Any]:
        if self.predict_example is None:
            self.predict_example = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]

        if not self.generate_widget_examples:
            return

        self.predict_example = self.predict_example[:3]  # Limit to 3 examples for standard similarity
        embeddings = self.model.encode(self.predict_example, convert_to_tensor=True, show_progress_bar=False)
        similarity = self.model.similarity(embeddings, embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def get_codecarbon_data(self) -> dict[Literal["co2_eq_emissions"], dict[str, Any]]:
        emissions_data = self.code_carbon_callback.tracker._prepare_emissions_data()
        results = {
            "co2_eq_emissions": {
                # * 1000 to convert kg to g
                "emissions": float(emissions_data.emissions) * 1000,
                "energy_consumed": float(emissions_data.energy_consumed),
                "source": "codecarbon",
                "training_type": "fine-tuning",
                "on_cloud": emissions_data.on_cloud == "Y",
                "cpu_model": emissions_data.cpu_model,
                "ram_total_size": emissions_data.ram_total_size,
                "hours_used": round(emissions_data.duration / 3600, 3),
            }
        }
        if emissions_data.gpu_model:
            results["co2_eq_emissions"]["hardware_used"] = emissions_data.gpu_model
        return results

    def get_model_specific_metadata(self) -> dict[str, Any]:
        similarity_fn_name = "Cosine Similarity"
        if self.model.similarity_fn_name:
            similarity_fn_name = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        return {
            "model_max_length": self.model.get_max_seq_length(),
            "output_dimensionality": self.model.get_sentence_embedding_dimension(),
            "model_string": str(self.model),
            "similarity_fn_name": similarity_fn_name,
        }

    def get_default_model_name(self) -> None:
        if self.base_model:
            return f"{self.model.__class__.__name__} based on {self.base_model}"
        else:
            return self.model.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        # Try to set the base model
        if self.first_save and not self.base_model:
            try:
                self.try_to_set_base_model()
            except Exception:
                pass

        # Set the model name
        if not self.model_name:
            self.model_name = self.get_default_model_name()

        # Compute the similarity scores for the usage snippet
        try:
            self.run_usage_snippet()
        except Exception as exc:
            logger.warning(f"Error while computing usage snippet output: {exc}")

        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Compute required formats from the (usually post-training) evaluation data
        if self.eval_results_dict:
            try:
                super_dict.update(self.format_eval_metrics())
            except Exception as exc:
                logger.warning(f"Error while formatting evaluation metrics: {exc}")
                raise exc

        # Compute required formats for the during-training evaluation data
        if self.training_logs:
            try:
                super_dict.update(self.format_training_logs())
            except Exception as exc:
                logger.warning(f"Error while formatting training logs: {exc}")

        super_dict["hide_eval_lines"] = len(self.training_logs) > 100

        # Try to add the code carbon callback data
        if (
            self.code_carbon_callback
            and self.code_carbon_callback.tracker
            and self.code_carbon_callback.tracker._start_time is not None
        ):
            super_dict.update(self.get_codecarbon_data())

        # Add some additional metadata stored in the model itself
        super_dict.update(self.get_model_specific_metadata())
        self.first_save = False

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)
        return super_dict

    def to_yaml(self, line_break=None) -> str:
        return yaml_dump(
            {key: value for key, value in self.to_dict().items() if key in YAML_FIELDS and value not in (None, [])},
            sort_keys=False,
            line_break=line_break,
        ).strip()


def generate_model_card(model: BaseModel) -> str:
    model_card = ModelCard.from_template(
        card_data=model.model_card_data, template_path=model.model_card_data.template_path, hf_emoji="🤗"
    )
    return model_card.content
