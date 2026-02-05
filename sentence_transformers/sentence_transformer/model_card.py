from __future__ import annotations

from dataclasses import field
from typing import Any

import torch
from typing_extensions import deprecated

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.base.models import Router
from sentence_transformers.sentence_transformer.models import StaticEmbedding


class SentenceTransformerModelCardCallback(BaseModelCardCallback):
    def on_log(self, args, state, control, model, logs, **kwargs):
        super().on_log(args, state, control, model, logs, **kwargs)

        # Set the ir_model flag so we can generate the model card with the encode_query/encode_document methods
        keys = {"loss"} & set(logs)
        if model.model_card_data.ir_model is None:
            for key in keys:
                if "ndcg" in key:
                    model.model_card_data.ir_model = True


class SentenceTransformerModelCardData(BaseModelCardData):
    ir_model: bool | None = field(default=None, init=False, repr=False)

    def try_to_set_base_model(self):
        super().try_to_set_base_model()
        if isinstance(self.model[0], StaticEmbedding) and self.base_model is None:
            if self.model[0].base_model:
                self.set_base_model(self.model[0].base_model)

    def register_model(self, model):
        super().register_model(model)

        if self.ir_model is not None:
            return

        if Router in [module.__class__ for module in model.children()]:
            self.ir_model = True
            return

        for ir_prompt_name in ["query", "document", "passage", "corpus"]:
            if ir_prompt_name in model.prompts and len(model.prompts[ir_prompt_name]) > 0:
                self.ir_model = True
                return

    def extract_dataset_metadata(self, dataset, dataset_metadata, loss, dataset_type):
        validated_datasets = super().extract_dataset_metadata(dataset, dataset_metadata, loss, dataset_type)
        if dataset_type == "train":
            if self.ir_model is None:
                if isinstance(dataset, dict):
                    column_names = set(
                        column for sub_dataset in dataset.values() for column in sub_dataset.column_names
                    )
                else:
                    column_names = set(dataset.column_names)
                if {"query", "question"} & column_names:
                    self.ir_model = True
        return validated_datasets

    def run_usage_snippet(self) -> dict[str, Any]:
        if self.predict_example is None:
            if self.ir_model:
                self.predict_example = [
                    "Which planet is known as the Red Planet?",
                    "Venus is often called Earth's twin because of its similar size and proximity.",
                    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
                ]
            else:
                self.predict_example = [
                    "The weather is lovely today.",
                    "It's so sunny outside!",
                    "He drove to the stadium.",
                ]

        if not self.generate_widget_examples:
            return

        if self.ir_model:
            query_embeddings = self.model.encode_query(
                self.predict_example[0], convert_to_tensor=True, show_progress_bar=False
            )
            document_embeddings = self.model.encode_document(
                self.predict_example[1:], convert_to_tensor=True, show_progress_bar=False
            )
            similarity = self.model.similarity(query_embeddings, document_embeddings)
        else:
            self.predict_example = self.predict_example[:3]  # Limit to 3 examples for standard similarity
            embeddings = self.model.encode(self.predict_example, convert_to_tensor=True, show_progress_bar=False)
            similarity = self.model.similarity(embeddings, embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())


@deprecated(
    "The `ModelCardCallback` has been renamed to `SentenceTransformerModelCardCallback` and the former is now deprecated. Please use `SentenceTransformerModelCardCallback` instead."
)
class ModelCardCallback(SentenceTransformerModelCardCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
