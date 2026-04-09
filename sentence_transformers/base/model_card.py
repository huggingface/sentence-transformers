from __future__ import annotations

import json
import logging
import os
import random
import re
import time
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
from sentence_transformers.base.modality import format_modality
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.util import fullname, is_accelerate_available, is_datasets_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

    try:
        from datasets import Image as ImageFeature
    except ImportError:
        ImageFeature = None
    try:
        from datasets import Audio as AudioFeature
    except ImportError:
        AudioFeature = None
    try:
        from datasets import Video as VideoFeature
    except ImportError:
        VideoFeature = None

try:
    from PIL.Image import Image as PILImage
except ImportError:
    PILImage = None

try:
    from torchcodec.decoders import AudioDecoder
except (ImportError, OSError):
    AudioDecoder = None  # type: ignore[assignment,misc]

try:
    from torchcodec.decoders import VideoDecoder
except (ImportError, OSError):
    VideoDecoder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
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
        model.model_card_data._training_start_time = time.time()

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
        if "eval_runtime" in metrics:
            model.model_card_data.evaluation_duration += metrics["eval_runtime"]

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
        if "loss" in logs:
            if (
                model.model_card_data.training_logs
                and model.model_card_data.training_logs[-1]["Step"] == state.global_step
            ):
                model.model_card_data.training_logs[-1]["Training Loss"] = logs["loss"]
            else:
                model.model_card_data.training_logs.append(
                    {
                        "Epoch": state.epoch,
                        "Step": state.global_step,
                        "Training Loss": logs["loss"],
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
IGNORED_FIELDS = [
    "model",
    "trainer",
    "eval_results_dict",
    "save_dir",
    "usage_examples_display",
    "_asset_cache",
    "_cached_dict",
    "_training_start_time",
    "evaluation_duration",
]


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


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string, e.g. "23 minutes" or "1.6 hours"."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"
    hours = seconds / 3600
    if hours < 24:
        return f"{hours:.1f} hours"
    days = seconds / 86400
    return f"{days:.1f} days"


def format_log(value: float | int | str) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def get_losses(loss: nn.Module | dict[str, nn.Module]) -> list[nn.Module]:
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
        model_name (`Optional[str]`): The pretty name of the model.
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub.
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on.
        tags (`Optional[List[str]]`): A list of tags for the model.
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
            Defaults to False.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset,
            and compute their similarities. Defaults to True.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.
    """

    # Potentially provided by the user
    language: str | list[str] | None = field(default_factory=list)
    license: str | None = None
    model_name: str | None = None
    model_id: str | None = None
    train_datasets: list[dict[str, str]] = field(default_factory=list)
    eval_datasets: list[dict[str, str]] = field(default_factory=list)
    task_name: str | None = "retrieval"
    tags: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
        ]
    )
    local_files_only: bool = False
    generate_widget_examples: bool = field(default=True)

    # Automatically filled by `SentenceTransformerModelCardCallback` and the Trainer directly
    base_model: str | None = field(default=None, init=False)
    base_model_revision: str | None = field(default=None, init=False)
    non_default_hyperparameters: dict[str, Any] = field(default_factory=dict, init=False)
    all_hyperparameters: dict[str, Any] = field(default_factory=dict, init=False)
    eval_results_dict: dict[BaseEvaluator, dict[str, Any]] | None = field(default_factory=dict, init=False)
    training_logs: list[dict[str, float]] = field(default_factory=list, init=False)
    widget: list[dict[str, str]] = field(default_factory=list, init=False)
    usage_examples: list | None = field(default=None, init=False)
    usage_examples_display: list | None = field(default=None, init=False, repr=False)
    label_example_list: list[dict[str, str]] = field(default_factory=list, init=False)
    code_carbon_callback: CodeCarbonCallback | None = field(default=None, init=False)
    _training_start_time: float | None = field(default=None, init=False)
    evaluation_duration: float = field(default=0.0, init=False)
    citations: dict[str, str] = field(default_factory=dict, init=False)
    best_model_step: int | None = field(default=None, init=False)
    ir_model: bool | None = field(default=None, init=False, repr=False)
    datasets: list[str] = field(default_factory=list, init=False, repr=False)
    similarities: str | None = field(default=None, init=False, repr=False)

    # Utility fields
    first_save: bool = field(default=True, init=False)
    widget_step: int = field(default=-1, init=False)
    save_dir: str | None = field(default=None, init=False, repr=False)
    _asset_cache: dict = field(default_factory=dict, init=False, repr=False)
    _cached_dict: dict | None = field(default=None, init=False, repr=False)

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
                self.usage_examples = sentences[:4]

        # If the model supports non-text modalities, set multimodal usage_examples
        if self.model and any(m not in ("text", "message") for m in self.model.modalities):
            self._set_multimodal_usage_examples(dataset)

    def _set_multimodal_usage_examples(self, dataset: DatasetDict) -> None:
        """Override :attr:`usage_examples` with multimodal inputs when the model supports non-text modalities.

        Respects the distinction between models that support modalities independently (e.g. CLIP
        supports text OR image, but not combined) vs models that support combined modalities
        (e.g. BLIP supports text+image together via a tuple modality ``("image", "text")``).

        - If the model has a **tuple modality** matching the dataset columns, build multimodal dicts
          (e.g. ``{"text": "...", "image": <PIL.Image>}``).
        - If the model only supports individual non-text modalities (no matching tuple), pick the
          **first non-text modality** and show single-modality examples.
        """
        sub_dataset = next(iter(dataset.values()))
        if isinstance(sub_dataset, IterableDataset) or len(sub_dataset) == 0:
            return

        # For IR models, source the query from the first column and documents from the second column
        if self.ir_model:
            columns = [col for col in sub_dataset.column_names if col != "dataset_name"]
            if len(columns) >= 2:
                query_col = columns[0]
                doc_col = columns[1]

                query = sub_dataset[0][query_col]

                documents = []
                seen_hashes = set()
                for i in range(min(100, len(sub_dataset))):
                    value = sub_dataset[i][doc_col]
                    content_hash = self._hash_asset(value) if not isinstance(value, str) else hash(value)
                    if content_hash is not None and content_hash in seen_hashes:
                        continue
                    if content_hash is not None:
                        seen_hashes.add(content_hash)
                    documents.append(value)
                    if len(documents) >= 3:
                        break

                self.usage_examples = [query] + documents
                return

        # Classify dataset columns by modality
        column_modalities: dict[str, str] = {}
        for column, feature in sub_dataset.features.items():
            if column == "dataset_name":
                continue
            if isinstance(feature, Value) and feature.dtype in {"string", "large_string"}:
                column_modalities[column] = "text"
            elif ImageFeature and isinstance(feature, ImageFeature):
                column_modalities[column] = "image"
            elif AudioFeature and isinstance(feature, AudioFeature):
                column_modalities[column] = "audio"
            elif VideoFeature and isinstance(feature, VideoFeature):
                column_modalities[column] = "video"

        available_modalities = set(column_modalities.values())

        # Check if the model has a tuple modality whose parts all match available columns.
        # E.g. BLIP has ("image", "text") and the dataset has both image and text columns.
        combined_modality: tuple | None = None
        for modality in self.model.modalities:
            if isinstance(modality, tuple) and all(part in available_modalities for part in modality):
                combined_modality = modality
                break

        if combined_modality:
            # Build multimodal dicts using the first column per modality in the tuple
            selected_columns: dict[str, str] = {}
            for part in combined_modality:
                for column, mod in column_modalities.items():
                    if mod == part and part not in selected_columns:
                        selected_columns[part] = column
                        break

            num_examples = min(3, len(sub_dataset))
            usage_examples = []
            for i in range(num_examples):
                sample = sub_dataset[i]
                usage_examples.append({mod: sample[col] for mod, col in selected_columns.items()})
            self.usage_examples = usage_examples
            return

        # No combined modality: pick the first non-text modality that both the model
        # and the dataset support, and show deduplicated single-modality examples.
        for modality in self.model.modalities:
            if isinstance(modality, str) and modality not in ("text", "message") and modality in available_modalities:
                col = next(c for c, m in column_modalities.items() if m == modality)
                examples = []
                seen_hashes = set()
                for i in range(min(100, len(sub_dataset))):
                    value = sub_dataset[i][col]
                    content_hash = self._hash_asset(value) if not isinstance(value, str) else hash(value)
                    if content_hash is not None and content_hash in seen_hashes:
                        continue
                    if content_hash is not None:
                        seen_hashes.add(content_hash)
                    examples.append(value)
                    if len(examples) >= 3:
                        break
                self.usage_examples = examples
                return

    def save_usage_example_assets(self) -> None:
        """Save non-text items in :attr:`usage_examples` as files in an ``assets/`` subdirectory.

        After saving, :attr:`usage_examples_display` is set with the same structure as
        :attr:`usage_examples` but with relative file paths (e.g. ``"assets/image_0.jpg"``)
        replacing raw data (PIL images, audio dicts, etc.). Text strings are kept as-is.

        This is called during save, after :meth:`run_usage_snippet` encodes the original data.
        :meth:`generate_usage_snippet` then uses :attr:`usage_examples_display` for the code block.
        """
        if not self.save_dir or not self.usage_examples:
            return

        # Quick check: if everything is already text (strings or list-of-strings), nothing to save
        if all(
            isinstance(item, str) or (isinstance(item, list) and all(isinstance(x, str) for x in item))
            for item in self.usage_examples
        ):
            return

        assets_dir = os.path.join(self.save_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        counter = 0
        display = []

        for item in self.usage_examples:
            if isinstance(item, str):
                display.append(item)
            elif isinstance(item, list) and all(isinstance(x, str) for x in item):
                display.append(item)
            elif isinstance(item, list):
                # Mixed-type list, e.g. CrossEncoder pair [PIL.Image, "text"]
                display_list = []
                for elem in item:
                    if isinstance(elem, str):
                        display_list.append(elem)
                    else:
                        rel_path = self._save_asset(elem, assets_dir, counter)
                        if rel_path:
                            counter += 1
                            display_list.append(rel_path)
                        else:
                            display_list.append(f"<{type(elem).__name__}>")
                display.append(display_list)
            elif isinstance(item, dict) and not self._is_typed_media_dict(item):
                # Multimodal input dict, e.g. {"text": "...", "image": <PIL.Image>}
                display_dict = {}
                for key, value in item.items():
                    if isinstance(value, str):
                        display_dict[key] = value
                    else:
                        rel_path = self._save_asset(value, assets_dir, counter)
                        if rel_path:
                            counter += 1
                            display_dict[key] = rel_path
                        else:
                            display_dict[key] = f"<{type(value).__name__}>"
                display.append(display_dict)
            else:
                # Single non-text item (PIL Image, AudioDict, VideoDict, array, ...)
                rel_path = self._save_asset(item, assets_dir, counter)
                if rel_path:
                    counter += 1
                    display.append(rel_path)
                else:
                    display.append(f"<{type(item).__name__}>")

        self.usage_examples_display = display

    @staticmethod
    def _is_typed_media_dict(item: dict) -> bool:
        """Check if a dict is an AudioDict or VideoDict (vs a multimodal input dict)."""
        return isinstance(item, dict) and "array" in item and ("sampling_rate" in item or "video_metadata" in item)

    @staticmethod
    def _hash_asset(value: Any) -> int | None:
        """Compute a content hash for an asset value, or None if the type is not supported."""
        # AudioDecoder supports dict-like access but is not a dict
        if AudioDecoder is not None and isinstance(value, AudioDecoder):
            value = {"array": value["array"], "sampling_rate": value["sampling_rate"]}
        # VideoDecoder: hash based on metadata (source path, duration, resolution)
        if VideoDecoder is not None and isinstance(value, VideoDecoder):
            m = value.metadata
            return hash((getattr(m, "path", None), m.duration_seconds, m.width, m.height, m.num_frames))
        if PILImage and isinstance(value, PILImage):
            return hash((value.tobytes(), value.size, value.mode))
        if isinstance(value, dict) and "array" in value:
            import numpy as np

            array = value["array"]
            if isinstance(array, torch.Tensor):
                array = array.cpu().numpy()
            if isinstance(array, np.ndarray):
                return hash((array.tobytes(), array.shape, value.get("sampling_rate")))
        return None

    def _save_asset(self, value: Any, assets_dir: str, idx: int, prefix: str = "") -> str | None:
        """Save a non-text value to the assets directory, deduplicating by content hash.

        Args:
            prefix: Prepended to filenames, e.g. ``"example_"`` → ``"example_image_0.jpg"``.

        Returns the relative path (e.g. ``"assets/image_0.jpg"``) on success, or ``None`` on failure.
        """
        # Compute hash before conversion so VideoDecoder/AudioDecoder hashing is consistent
        content_hash = self._hash_asset(value)
        if content_hash is not None and content_hash in self._asset_cache:
            return self._asset_cache[content_hash]

        # AudioDecoder supports dict-like access but is not a dict
        if AudioDecoder is not None and isinstance(value, AudioDecoder):
            value = {"array": value["array"], "sampling_rate": value["sampling_rate"]}
        # VideoDecoder: copy source file directly (avoids frame decoding issues) or decode
        if VideoDecoder is not None and isinstance(value, VideoDecoder):
            source_path = getattr(value.metadata, "path", None)
            if source_path and os.path.isfile(source_path):
                import shutil

                ext = os.path.splitext(source_path)[1] or ".mp4"
                filename = f"{prefix}video_{idx}{ext}"
                rel_path = f"assets/{filename}"
                os.makedirs(assets_dir, exist_ok=True)
                shutil.copy2(source_path, os.path.join(assets_dir, filename))
                if content_hash is not None:
                    self._asset_cache[content_hash] = rel_path
                return rel_path
            value = self._video_decoder_to_dict(value)

        os.makedirs(assets_dir, exist_ok=True)

        # PIL Image
        if PILImage and isinstance(value, PILImage):
            filename = f"{prefix}image_{idx}.jpg"
            rel_path = f"assets/{filename}"
            value.convert("RGB").save(os.path.join(assets_dir, filename))
            if content_hash is not None:
                self._asset_cache[content_hash] = rel_path
            return rel_path

        # AudioDict: {"array": ..., "sampling_rate": ...}
        if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
            try:
                import torchaudio

                array = value["array"]
                if not isinstance(array, torch.Tensor):
                    array = torch.as_tensor(array)
                if array.ndim == 1:
                    array = array.unsqueeze(0)  # (1, num_samples) for torchaudio
                filename = f"{prefix}audio_{idx}.wav"
                rel_path = f"assets/{filename}"
                torchaudio.save(os.path.join(assets_dir, filename), array.float().cpu(), value["sampling_rate"])
                if content_hash is not None:
                    self._asset_cache[content_hash] = rel_path
                return rel_path
            except Exception:
                pass

        # VideoDict: {"array": ..., "video_metadata": ...} - save as mp4
        if isinstance(value, dict) and "array" in value and "video_metadata" in value:
            try:
                import av

                array = value["array"]
                if not isinstance(array, torch.Tensor):
                    array = torch.as_tensor(array)
                if array.ndim == 5:
                    array = array[0]
                # Ensure (T, H, W, C) uint8 for av
                if array.ndim == 4 and array.shape[1] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                    # (T, C, H, W) -> (T, H, W, C)
                    array = array.permute(0, 2, 3, 1)
                if array.dtype != torch.uint8:
                    if array.is_floating_point() and array.max() <= 1.0:
                        array = (array * 255).clamp(0, 255).to(torch.uint8)
                    else:
                        array = array.clamp(0, 255).to(torch.uint8)

                fps = value.get("video_metadata", {}).get("fps", 24)
                filename = f"{prefix}video_{idx}.mp4"
                rel_path = f"assets/{filename}"
                filepath = os.path.join(assets_dir, filename)
                frames = array.cpu().numpy()
                height, width = frames.shape[1], frames.shape[2]
                with av.open(filepath, mode="w") as container:
                    stream = container.add_stream("h264", rate=round(fps))
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = "yuv420p"
                    for frame_data in frames:
                        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                if content_hash is not None:
                    self._asset_cache[content_hash] = rel_path
                return rel_path
            except Exception:
                pass

        logger.warning(f"Could not save predict example asset of type {type(value).__name__}")
        return None

    def set_evaluation_metrics(
        self, evaluator: BaseEvaluator, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        from sentence_transformers.base.evaluation import SequentialEvaluator

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
                inferred_dataset
                for dataset_name, sub_dataset in dataset.items()
                for inferred_dataset in self.infer_datasets(sub_dataset, dataset_name=dataset_name)
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
            # dataset[0].keys() reflects post-transform columns if set_transform is used
            dataset_columns = [column for column in dataset[0].keys() if column != "dataset_name"]
        else:
            dataset_columns = dataset.column_names

        dataset_info["columns"] = [f"<code>{column}</code>" for column in dataset_columns]
        dataset_info["stats"] = {}
        if isinstance(dataset, Dataset):
            for column in dataset_columns:
                subsection = dataset[:1000][column]
                first = subsection[0]
                if isinstance(first, str):
                    tokenized = self.model.preprocess(subsection, task="document")
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
                    # Handle non-text types: PIL Images, Audio dicts, etc.
                    if PILImage and isinstance(first, PILImage):
                        widths = [img.width for img in subsection if isinstance(img, PILImage)]
                        heights = [img.height for img in subsection if isinstance(img, PILImage)]
                        dataset_info["stats"][column] = {
                            "dtype": "image",
                            "data": {
                                "min": f"{min(widths)}x{min(heights)} px",
                                "mean": f"{sum(widths) // len(widths)}x{sum(heights) // len(heights)} px",
                                "max": f"{max(widths)}x{max(heights)} px",
                            },
                        }
                    elif (isinstance(first, dict) and "array" in first and "sampling_rate" in first) or (
                        AudioDecoder is not None and isinstance(first, AudioDecoder)
                    ):
                        durations = [
                            len(d["array"]) / d["sampling_rate"]
                            for d in subsection
                            if (isinstance(d, dict) and "array" in d and "sampling_rate" in d)
                            or (AudioDecoder is not None and isinstance(d, AudioDecoder))
                        ]
                        if durations:
                            dataset_info["stats"][column] = {
                                "dtype": "audio",
                                "data": {
                                    "min": f"{min(durations):.2f}s",
                                    "mean": f"{sum(durations) / len(durations):.2f}s",
                                    "max": f"{max(durations):.2f}s",
                                    "sampling_rate": f"{first['sampling_rate']} Hz",
                                },
                            }
                        else:
                            dataset_info["stats"][column] = {"dtype": "audio", "data": {}}
                    elif VideoDecoder is not None and isinstance(first, VideoDecoder):
                        durations = [
                            v.metadata.duration_seconds
                            for v in subsection
                            if VideoDecoder is not None and isinstance(v, VideoDecoder)
                        ]
                        widths = [
                            v.metadata.width
                            for v in subsection
                            if VideoDecoder is not None and isinstance(v, VideoDecoder)
                        ]
                        heights = [
                            v.metadata.height
                            for v in subsection
                            if VideoDecoder is not None and isinstance(v, VideoDecoder)
                        ]
                        if durations:
                            dataset_info["stats"][column] = {
                                "dtype": "video",
                                "data": {
                                    "min": f"{min(durations):.2f}s, {min(widths)}x{min(heights)} px",
                                    "mean": f"{sum(durations) / len(durations):.2f}s, {sum(widths) // len(widths)}x{sum(heights) // len(heights)} px",
                                    "max": f"{max(durations):.2f}s, {max(widths)}x{max(heights)} px",
                                    "fps": f"{first.metadata.average_fps:.0f}",
                                },
                            }
                        else:
                            dataset_info["stats"][column] = {"dtype": "video", "data": {}}
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
            dataset_info["_example_columns"] = dataset_columns
            dataset_info["examples_table"], _ = self._render_examples_table(dataset_info)

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

        # Try to detect IR model from dataset columns
        if dataset and dataset_type == "train" and self.ir_model is None:
            if isinstance(dataset, dict):
                column_names = set(column for sub_dataset in dataset.values() for column in sub_dataset.column_names)
            else:
                column_names = set(dataset.column_names)
            if {"query", "question"} & column_names:
                self.ir_model = True

        return self.validate_datasets(dataset_metadata)

    def register_model(self, model: BaseModel) -> None:
        self.model = model

        if self.ir_model is not None:
            return

        from sentence_transformers.base.modules import Router

        if Router in [module.__class__ for module in model.children()]:
            self.ir_model = True
            return

        for ir_prompt_name in ["query", "document", "passage", "corpus"]:
            if ir_prompt_name in model.prompts and len(model.prompts[ir_prompt_name]) > 0:
                self.ir_model = True
                return

    def set_model_id(self, model_id: str) -> None:
        self.model_id = model_id

    def set_base_model(self, model_id: str, revision: str | None = None) -> bool:
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

    @staticmethod
    def _video_decoder_to_dict(value: Any) -> dict[str, Any]:
        """Convert a ``VideoDecoder`` to a ``VideoDict`` by extracting all frames.

        Tries multiple strategies to extract frames:
        1. ``get_frames_at`` on the original decoder (random-access batch)
        2. Recreate a fresh decoder from source path and retry
        3. Collect frames one-by-one via ``decoder[i]`` (random-access seek)
        """
        fps = value.metadata.average_fps
        path = getattr(value.metadata, "path", None)

        def _make_metadata(num_decoded_frames: int) -> dict[str, Any]:
            return {
                "fps": fps,
                "total_num_frames": value.metadata.num_frames,
                "frames_indices": list(range(num_decoded_frames)),
            }

        def _try_decode(decoder) -> dict[str, Any] | None:
            n = len(decoder)
            # Try batch decode with get_frames_at
            for num_frames in (n, n - 1):
                try:
                    frames = decoder.get_frames_at(list(range(num_frames)))
                    return {
                        "array": frames.data,
                        "video_metadata": _make_metadata(frames.data.shape[0]),
                    }
                except Exception:
                    continue

            # Fall back to collecting frames one-by-one via random access
            collected = []
            for i in range(n):
                try:
                    collected.append(decoder[i])
                except Exception:
                    break
            if collected:
                return {
                    "array": torch.stack(collected),
                    "video_metadata": _make_metadata(len(collected)),
                }

            return None

        # Try the original decoder first
        result = _try_decode(value)
        if result is not None:
            return result

        # Recreate a fresh decoder from the source path
        if path:
            try:
                fresh = VideoDecoder(path)
                result = _try_decode(fresh)
                if result is not None:
                    return result
            except Exception:
                pass

        raise RuntimeError("Could not decode frames from VideoDecoder")

    @staticmethod
    def _prepare_for_inference(value: Any) -> Any:
        """Convert a value to a format suitable for model inference.

        ``VideoDecoder`` objects are converted to a ``VideoDict`` via :meth:`_video_decoder_to_dict`.
        All other values are returned as-is.
        """
        if VideoDecoder is not None and isinstance(value, VideoDecoder):
            return BaseModelCardData._video_decoder_to_dict(value)
        if isinstance(value, dict):
            return {k: BaseModelCardData._prepare_for_inference(v) for k, v in value.items()}
        return value

    def run_usage_snippet(self) -> None:
        if self.usage_examples is None:
            self.usage_examples = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]

    def generate_usage_snippet(self) -> str:
        """Generate the Python usage code snippet for the model card.

        Returns the code block (including \\`\\`\\` delimiters) showing how to use this model.
        Called after :meth:`run_usage_snippet` has set :attr:`usage_examples` and :attr:`similarities`.

        Subclasses can override this to generate snippets for different model types (e.g. IR models,
        cross-encoders) or multimodal inputs.
        """
        # Use display version (with file paths) if available, otherwise original usage_examples
        display = self.usage_examples_display or self.usage_examples
        if not display:
            return self._generate_text_snippet(None)

        # Check the *original* usage_examples for modality detection, since display converts
        # non-text items (PIL images, audio dicts, etc.) to file path strings.
        source = self.usage_examples or display
        has_non_text = any(not isinstance(item, (str, list)) for item in source)
        if has_non_text:
            return self._generate_non_text_snippet(display)

        return self._generate_text_snippet(display)

    def _generate_text_snippet(self, display: list[str] | None) -> str:
        """Generate a text-only usage snippet."""
        model_class = getattr(self, "_snippet_model_class", "SentenceTransformer")
        default_model_id = getattr(self, "_snippet_default_model_id", "sentence_transformers_model_id")
        model_id = self.model_id or default_model_id
        examples = display or [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
        ]
        output_dim = self._get_snippet_output_dimensionality()

        lines = [
            f"from sentence_transformers import {model_class}",
            "",
            "# Download from the 🤗 Hub",
            f'model = {model_class}("{model_id}")',
            "# Run inference",
            "sentences = [",
        ]
        for text in examples:
            lines.append(f"    {text!r},")
        lines.extend(
            [
                "]",
                "embeddings = model.encode(sentences)",
                "print(embeddings.shape)",
                f"# [{len(examples)}, {output_dim}]",
                "",
                "# Get the similarity scores for the embeddings",
                "similarities = model.similarity(embeddings, embeddings)",
            ]
        )
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(
                [
                    "print(similarities.shape)",
                    f"# [{len(examples)}, {len(examples)}]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"

    def _generate_non_text_snippet(self, display: list[str | dict[str, str]]) -> str:
        """Generate a usage snippet for non-text inputs (multimodal dicts or single-modality items)."""
        model_class = getattr(self, "_snippet_model_class", "SentenceTransformer")
        default_model_id = getattr(self, "_snippet_default_model_id", "sentence_transformers_model_id")
        model_id = self.model_id or default_model_id
        output_dim = self._get_snippet_output_dimensionality()

        lines = [
            f"from sentence_transformers import {model_class}",
            "",
            "# Download from the 🤗 Hub",
            f'model = {model_class}("{model_id}")',
            "# Run inference",
            "inputs = [",
        ]
        for item in display:
            lines.append(f"    {self._format_snippet_value(item)},")
        lines.extend(
            [
                "]",
                "embeddings = model.encode(inputs)",
                "print(embeddings.shape)",
                f"# [{len(display)}, {output_dim}]",
                "",
                "# Get the similarity scores for the embeddings",
                "similarities = model.similarity(embeddings, embeddings)",
            ]
        )
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(
                [
                    "print(similarities.shape)",
                    f"# [{len(display)}, {len(display)}]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"

    def _render_examples_table(self, dataset_info: dict, asset_counter: int = 0) -> tuple[str, int]:
        """Render the examples table for a dataset, saving non-text values as assets when possible.

        Returns:
            A tuple of ``(rendered_table, new_asset_counter)``. The counter should be passed to
            subsequent calls to avoid filename collisions across datasets.
        """
        dataset_columns = dataset_info["_example_columns"]
        if not dataset_info["examples"]:
            return "", asset_counter
        num_samples = len(dataset_info["examples"][list(dataset_info["examples"])[0]])
        examples_lines = []
        for sample_idx in range(num_samples):
            columns = {}
            for column in dataset_columns:
                value = dataset_info["examples"][column][sample_idx]
                cell, asset_counter = self._format_and_save_example(value, asset_counter)
                columns[column] = cell
            examples_lines.append(columns)
        return indent(make_markdown_table(examples_lines).replace("-:|", "--|"), "  "), asset_counter

    def _format_and_save_example(self, value: Any, counter: int) -> tuple[str, int]:
        """Format a dataset example value for the model card table, saving non-text values as assets.

        Delegates the actual file I/O to :meth:`_save_asset` and wraps the resulting path in the
        appropriate HTML tag via :meth:`_example_asset_html`.

        Returns:
            A tuple of ``(html_cell_content, new_counter)``.
        """
        is_media = (
            (PILImage and isinstance(value, PILImage))
            or (AudioDecoder is not None and isinstance(value, AudioDecoder))
            or (isinstance(value, dict) and "array" in value and "sampling_rate" in value)
            or (VideoDecoder is not None and isinstance(value, VideoDecoder))
        )
        if not is_media or not self.save_dir:
            return f"<code>{self._format_example_value(value)}</code>", counter

        # Check cache, cached assets must not increment the counter
        content_hash = self._hash_asset(value)
        if content_hash is not None and content_hash in self._asset_cache:
            return self._example_asset_html(value, self._asset_cache[content_hash]), counter

        assets_dir = os.path.join(self.save_dir, "assets")
        rel_path = self._save_asset(value, assets_dir, counter, prefix="example_")
        if rel_path:
            return self._example_asset_html(value, rel_path), counter + 1

        return f"<code>{self._format_example_value(value)}</code>", counter

    def _example_asset_html(self, value: Any, rel_path: str) -> str:
        """Generate an inline HTML tag for a saved example asset."""
        if PILImage and isinstance(value, PILImage):
            return f'<img src="{rel_path}" width="200">'
        # AudioDecoder supports dict-like access, so both AudioDecoder and audio dicts work here
        if (AudioDecoder is not None and isinstance(value, AudioDecoder)) or (
            isinstance(value, dict) and "array" in value and "sampling_rate" in value
        ):
            duration = len(value["array"]) / value["sampling_rate"]
            return f'<audio controls src="{rel_path}"><code>&lt;audio {duration:.2f}s&gt;</code></audio>'
        if VideoDecoder is not None and isinstance(value, VideoDecoder):
            m = value.metadata
            return (
                f'<video controls width="200" src="{rel_path}">'
                f"<code>&lt;video {m.duration_seconds:.2f}s&gt;</code></video>"
            )
        return f"<code>{self._format_example_value(value)}</code>"

    @staticmethod
    def _format_example_value(value: Any) -> str:
        """Format a dataset example value for the model card examples table."""
        # AudioDecoder supports dict-like access but is not a dict
        if AudioDecoder is not None and isinstance(value, AudioDecoder):
            value = {"array": value["array"], "sampling_rate": value["sampling_rate"]}
        if VideoDecoder is not None and isinstance(value, VideoDecoder):
            m = value.metadata
            return f"&lt;video {m.duration_seconds:.2f}s {m.width}x{m.height} @ {m.average_fps:.0f}fps&gt;"
        if PILImage and isinstance(value, PILImage):
            return f"&lt;image {value.width}x{value.height}&gt;"
        if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
            duration = len(value["array"]) / value["sampling_rate"]
            return f"&lt;audio {duration:.2f}s @ {value['sampling_rate']} Hz&gt;"
        if isinstance(value, dict) and "array" in value and "video_metadata" in value:
            return "&lt;video&gt;"
        if isinstance(value, list) and len(value) > 5:
            return str(value[:5])[:-1] + ", ...]"
        if isinstance(value, str) and len(value) > 1000:
            return value[:1000] + "..."
        result = str(value).replace("\n", "<br>").replace("|", "\\|")
        return result

    def _format_snippet_value(self, value: Any) -> str:
        """Format a value for inclusion in a code snippet.

        Strings are shown as repr (quoted), and asset paths are converted to Hub URLs
        when model_id is available. Dicts and lists are formatted recursively so that
        nested asset paths also get URL conversion.
        """
        if isinstance(value, str) and value.startswith("assets/"):
            return repr(self._asset_path_to_url(value))
        if isinstance(value, dict):
            parts = ", ".join(f"{k!r}: {self._format_snippet_value(v)}" for k, v in value.items())
            return f"{{{parts}}}"
        if isinstance(value, list):
            elems = ", ".join(self._format_snippet_value(v) for v in value)
            return f"[{elems}]"
        return repr(value)

    def _asset_path_to_url(self, relative_path: str) -> str:
        """Convert a relative asset path to a Hub URL if model_id is available, otherwise keep relative."""
        if self.model_id:
            return f"https://huggingface.co/{self.model_id}/resolve/main/{relative_path}"
        return relative_path

    def _get_snippet_output_dimensionality(self) -> int | str:
        if self.model:
            try:
                return self.model.get_embedding_dimension()
            except Exception:
                pass
        return "?"

    def get_codecarbon_data(self) -> dict[str, Any]:
        emissions_data = self.code_carbon_callback.tracker._prepare_emissions_data()
        co2_eq_emissions: dict[str, Any] = {
            # * 1000 to convert kg to g
            "emissions": float(emissions_data.emissions) * 1000,
            "energy_consumed": float(emissions_data.energy_consumed),
            "source": "codecarbon",
            "training_type": "fine-tuning",
            "on_cloud": emissions_data.on_cloud == "Y",
            "cpu_model": emissions_data.cpu_model,
            "ram_total_size": emissions_data.ram_total_size,
        }
        if emissions_data.gpu_model:
            co2_eq_emissions["hardware_used"] = emissions_data.gpu_model
        return {"co2_eq_emissions": co2_eq_emissions}

    def get_training_duration_data(self) -> dict[str, str | None]:
        if self._training_start_time is None:
            return {"training_time": None, "evaluation_time": None, "total_time": None}
        total_duration = time.time() - self._training_start_time
        training_duration = total_duration - self.evaluation_duration
        return {
            "training_time": format_duration(training_duration),
            "evaluation_time": format_duration(self.evaluation_duration) if self.evaluation_duration else None,
            "total_time": format_duration(total_duration) if self.evaluation_duration else None,
        }

    def get_model_specific_metadata(self) -> dict[str, Any]:
        if self.model is None:
            return {}
        supported_modalities = [format_modality(m).title() for m in self.model.modalities]
        return {
            "model_max_length": self.model.max_seq_length,
            "model_string": str(self.model),
            "supported_modalities": supported_modalities,
        }

    def get_default_model_name(self) -> str:
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

        # Clear asset cache so deduplication works within a single save but doesn't persist stale paths
        self._asset_cache = {}

        # Save non-text predict example assets (images, audio, etc.) to the assets/ directory.
        # Must run after run_usage_snippet() (which encodes the original data) and before
        # generate_usage_snippet() (which needs the file paths).
        try:
            self.save_usage_example_assets()
        except Exception as exc:
            logger.warning(f"Error while saving usage example assets: {exc}")

        # Re-render dataset example tables now that save_dir is available for asset saving.
        # The tables were first rendered during Trainer init (compute_dataset_metrics) when
        # save_dir was not yet set, so non-text values got placeholder text instead of saved files.
        if self.save_dir:
            example_asset_counter = 0
            for dataset_list in (self.train_datasets, self.eval_datasets):
                for dataset_info in dataset_list:
                    if "examples" in dataset_info and "_example_columns" in dataset_info:
                        try:
                            dataset_info["examples_table"], example_asset_counter = self._render_examples_table(
                                dataset_info, example_asset_counter
                            )
                        except Exception as exc:
                            logger.warning(f"Error while re-rendering examples table: {exc}")

        super_dict = {field.name: getattr(self, field.name) for field in fields(self)}

        # Generate the usage snippet code block
        try:
            super_dict["usage_snippet"] = self.generate_usage_snippet()
        except Exception as exc:
            logger.warning(f"Error while generating usage snippet: {exc}")
            super_dict["usage_snippet"] = ""

        # Compute required formats from the (usually post-training) evaluation data
        if self.eval_results_dict:
            try:
                super_dict.update(self.format_eval_metrics())
            except Exception as exc:
                logger.warning(f"Error while formatting evaluation metrics: {exc}")

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

        super_dict.update(self.get_training_duration_data())

        # Add some additional metadata stored in the model itself
        super_dict.update(self.get_model_specific_metadata())
        self.first_save = False

        for key in IGNORED_FIELDS:
            super_dict.pop(key, None)

        # Cache result so that to_yaml() can reuse it without re-running expensive
        # operations like usage snippet generation and dataset example rendering.
        # This matters because huggingface_hub's ModelCard.from_template() calls
        # to_dict() and then to_yaml() back-to-back.
        self._cached_dict = super_dict

        return super_dict

    def to_yaml(self, line_break=None) -> str:
        if self._cached_dict is not None:
            data = self._cached_dict
            self._cached_dict = None
        else:
            data = self.to_dict()
        return yaml_dump(
            {key: value for key, value in data.items() if key in YAML_FIELDS and value not in (None, [])},
            sort_keys=False,
            line_break=line_break,
        ).strip()


def generate_model_card(model: BaseModel) -> str:
    model_card = ModelCard.from_template(
        card_data=model.model_card_data, template_path=model.model_card_data.template_path, hf_emoji="🤗"
    )
    content = model_card.content

    # Replace relative asset paths with absolute Hub URLs so that images, audio, and
    # video files render correctly when the README is viewed on the Hugging Face Hub.
    model_id = getattr(model.model_card_data, "model_id", None)
    if model_id:
        base_url = f"https://huggingface.co/{model_id}/resolve/main/"
        content = content.replace('src="assets/', f'src="{base_url}assets/')

    return content
