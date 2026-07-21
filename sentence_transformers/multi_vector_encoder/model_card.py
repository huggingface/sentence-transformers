from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.base.modules import Transformer

if TYPE_CHECKING:
    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder

logger = logging.getLogger(__name__)


class MultiVectorEncoderModelCardCallback(BaseModelCardCallback):
    pass


@dataclass
class MultiVectorEncoderModelCardData(BaseModelCardData):
    """A dataclass storing data used in the model card for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g.
            "MultiVectorEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/mve-modernbert-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training
            datasets, e.g. ``[{"name": "MS MARCO", "id": "microsoft/ms_marco"}]``.
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation
            datasets.
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search with late interaction".
        tags (`Optional[List[str]]`): A list of tags for the model, e.g.
            ``["sentence-transformers", "multi-vector", "colbert", "late-interaction"]``.
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.
    """

    _snippet_model_class = "MultiVectorEncoder"
    _snippet_default_model_id = "multi_vector_encoder_model_id"

    task_name: str | None = None
    tags: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "multi-vector",
            "colbert",
            "late-interaction",
        ]
    )

    usage_examples: list[list[str]] | None = field(default=None, init=False)
    usage_query_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    usage_document_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)
    model_type: str = field(default="Multi-Vector Encoder", init=False, repr=False)

    model: MultiVectorEncoder | None = field(default=None, init=False, repr=False)

    def register_model(self, model: MultiVectorEncoder) -> None:
        super().register_model(model)

        if self.task_name is None:
            self.task_name = "semantic search with late interaction"
        if self.pipeline_tag is None:
            self.pipeline_tag = "feature-extraction"
        # Late interaction is always retrieval: the base then sources widget examples positionally,
        # query from the first dataset column and documents from the second.
        if self.ir_model is None:
            self.ir_model = True

    def get_model_specific_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_specific_metadata()
        # query_length / document_length live on the backbone Transformer, not on the model.
        transformer = next((module for module in self.model if isinstance(module, Transformer)), None)
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": self.model.similarity_fn_name,
                "query_length": getattr(transformer, "query_length", None),
                "document_length": getattr(transformer, "document_length", None),
            }
        )
        return metadata

    def run_usage_snippet(self) -> None:
        if self.usage_examples is None:
            self.usage_examples = [
                "Which planet is known as the Red Planet?",
                "Venus is often called Earth's twin because of its similar size and proximity.",
                "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
            ]

        if not self.generate_widget_examples:
            return

        self.usage_examples = self.usage_examples[:4]
        prepared_examples = [self._prepare_for_inference(item) for item in self.usage_examples]
        # Mirror the positional queries / documents split of the generated snippet (first example is
        # the query, the rest are documents) so the printed similarity matrix matches what a user
        # running the snippet sees.
        if len(prepared_examples) >= 2:
            queries = prepared_examples[:1]
            documents = prepared_examples[1:]
        else:
            queries = documents = prepared_examples
        query_embeddings = self.model.encode_query(queries, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = self.model.encode_document(documents, convert_to_tensor=True, show_progress_bar=False)
        self.usage_query_shape = tuple(query_embeddings[0].shape)
        self.usage_document_shape = tuple(doc_embeddings[0].shape)
        similarity = self.model.similarity(query_embeddings, doc_embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def generate_usage_snippet(self) -> str:
        # MV encode() returns variable-length per-token lists (no stacked `.shape`), and late
        # interaction is always retrieval, so the snippet splits positionally into queries /
        # documents, matching how the base sources IR usage examples (query from the first dataset
        # column, documents from the second). Text and image documents render identically.
        display = self.usage_examples_display or self.usage_examples
        examples = display or [
            "Which planet is known as the Red Planet?",
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]
        model_class = self._snippet_model_class
        model_id = self.model_id or self._snippet_default_model_id
        output_dim = self._get_snippet_output_dimensionality()

        if len(examples) >= 2:
            queries = examples[:1]
            documents = examples[1:]
        else:
            queries = documents = examples
        query_shape = self.usage_query_shape or f"(num_query_tokens, {output_dim})"
        document_shape = self.usage_document_shape or f"(num_document_tokens, {output_dim})"

        lines = [
            f"from sentence_transformers import {model_class}",
            "",
            "# Download from the 🤗 Hub",
            f'model = {model_class}("{model_id}")',
            "# Run inference: each input becomes a sequence of per-token vectors (variable length).",
            "queries = [",
            *(f"    {self._format_snippet_value(item)}," for item in queries),
            "]",
            "documents = [",
            *(f"    {self._format_snippet_value(item)}," for item in documents),
            "]",
            "query_embeddings = model.encode_query(queries)",
            "document_embeddings = model.encode_document(documents)",
            "print(query_embeddings[0].shape, document_embeddings[0].shape)",
            f"# {query_shape} {document_shape}",
            "",
            "# Get the MaxSim similarity scores",
            "similarities = model.similarity(query_embeddings, document_embeddings)",
        ]
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(["print(similarities.shape)", f"# [{len(queries)}, {len(documents)}]"])

        return "```python\n" + "\n".join(lines) + "\n```"

    def get_default_model_name(self) -> str:
        return self.model_type
