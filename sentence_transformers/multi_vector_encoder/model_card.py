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

    def get_model_specific_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_specific_metadata()
        # query_length / document_length live on the backbone Transformer, not on the model.
        transformer = next((module for module in self.model if isinstance(module, Transformer)), None)
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": "maxsim",
                "query_length": getattr(transformer, "query_length", None),
                "document_length": getattr(transformer, "document_length", None),
            }
        )
        return metadata

    def run_usage_snippet(self) -> None:
        super().run_usage_snippet()

        if not self.generate_widget_examples:
            return

        self.usage_examples = self.usage_examples[:3]
        prepared_examples = [self._prepare_for_inference(item) for item in self.usage_examples]
        query_embeddings = self.model.encode_query(prepared_examples, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = self.model.encode_document(prepared_examples, convert_to_tensor=True, show_progress_bar=False)
        similarity = self.model.similarity(query_embeddings, doc_embeddings)

        with torch._tensor_str.printoptions(precision=4, sci_mode=False):
            self.similarities = "\n".join(f"# {line}" for line in str(similarity.cpu()).splitlines())

    def generate_usage_snippet(self) -> str:
        # MV encode() returns variable-length per-token lists (no `.shape`), so the base snippet is wrong.
        # ColPali-style checkpoints emit text queries + image documents, text-only checkpoints use symmetric encode_query / encode_document.
        display = self.usage_examples_display or self.usage_examples
        source = self.usage_examples or display
        has_non_text = bool(source) and any(not isinstance(item, (str, list)) for item in source)
        if has_non_text:
            return self._generate_multimodal_snippet(display)
        return self._generate_text_only_snippet(display)

    def _generate_text_only_snippet(self, display: list | None) -> str:
        model_class = self._snippet_model_class
        model_id = self.model_id or self._snippet_default_model_id
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
            "# Run inference: each input becomes a list of per-token vectors (variable length).",
            "sentences = [",
            *(f"    {text!r}," for text in examples),
            "]",
            "query_embeddings = model.encode_query(sentences)",
            "document_embeddings = model.encode_document(sentences)",
            "print(query_embeddings[0].shape)",
            f"# (num_query_tokens, {output_dim})",
            "",
            "# Get the MaxSim similarity scores",
            "similarities = model.similarity(query_embeddings, document_embeddings)",
        ]
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(["print(similarities.shape)", f"# [{len(examples)}, {len(examples)}]"])

        return "```python\n" + "\n".join(lines) + "\n```"

    def _generate_multimodal_snippet(self, display: list) -> str:
        """Image / multimodal-document usage snippet for ColPali-style models. Text items in
        ``display`` (if any) become example queries; non-text items become example documents. A
        generic placeholder query is used if no text items are present.
        """
        model_class = self._snippet_model_class
        model_id = self.model_id or self._snippet_default_model_id
        output_dim = self._get_snippet_output_dimensionality()

        text_items = [item for item in display if isinstance(item, (str, list))]
        non_text_items = [item for item in display if not isinstance(item, (str, list))]
        queries_repr = text_items or ["A query about the document."]

        # TODO: Check that this renders nicely in the model card with various multimodal settings
        lines = [
            f"from sentence_transformers import {model_class}",
            "",
            "# Download from the 🤗 Hub",
            f'model = {model_class}("{model_id}")',
            "# Run inference",
            "queries = [",
            *(f"    {self._format_snippet_value(item)}," for item in queries_repr),
            "]",
            "documents = [",
            *(f"    {self._format_snippet_value(item)}," for item in non_text_items),
            "]",
            "query_embeddings = model.encode_query(queries)",
            "document_embeddings = model.encode_document(documents)",
            "print(query_embeddings[0].shape)",
            f"# (num_query_tokens, {output_dim})",
            "",
            "# Get the MaxSim similarity scores",
            "similarities = model.similarity(query_embeddings, document_embeddings)",
        ]
        if self.similarities:
            lines.append("print(similarities)")
            lines.append(self.similarities)
        else:
            lines.extend(
                [
                    "print(similarities.shape)",
                    f"# [{len(queries_repr)}, {len(non_text_items)}]",
                ]
            )

        return "```python\n" + "\n".join(lines) + "\n```"

    def get_default_model_name(self) -> str:
        return self.model_type
