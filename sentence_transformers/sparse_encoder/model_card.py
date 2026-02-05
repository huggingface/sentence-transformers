from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.base.models import Module, Router
from sentence_transformers.sparse_encoder.models import SparseAutoEncoder, SparseStaticEmbedding, SpladePooling

if TYPE_CHECKING:
    from sentence_transformers.sparse_encoder.model import SparseEncoder

logger = logging.getLogger(__name__)


class SparseEncoderModelCardCallback(BaseModelCardCallback):
    pass


@dataclass
class SparseEncoderModelCardData(BaseModelCardData):
    """A dataclass storing data used in the model card.

    Args:
        language (`Optional[Union[str, List[str]]]`): The model language, either a string or a list,
            e.g. "en" or ["en", "de", "nl"]
        license (`Optional[str]`): The license of the model, e.g. "apache-2.0", "mit",
            or "cc-by-nc-sa-4.0"
        model_name (`Optional[str]`): The pretty name of the model, e.g. "SparseEncoder based on answerdotai/ModernBERT-base".
        model_id (`Optional[str]`): The model ID when pushing the model to the Hub,
            e.g. "tomaarsen/se-mpnet-base-ms-marco".
        train_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the training datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"name": "MultiNLI", "id": "nyu-mll/multi_nli"}, {"name": "STSB"}]
        eval_datasets (`List[Dict[str, str]]`): A list of the names and/or Hugging Face dataset IDs of the evaluation datasets.
            e.g. [{"name": "SNLI", "id": "stanfordnlp/snli"}, {"id": "mteb/stsbenchmark-sts"}]
        task_name (`str`): The human-readable task the model is trained on,
            e.g. "semantic search and sparse retrieval".
        tags (`Optional[List[str]]`): A list of tags for the model,
            e.g. ["sentence-transformers", "sparse-encoder"].
        local_files_only (`bool`): If True, don't attempt to find dataset or base model information on the Hub.Add commentMore actions
            Defaults to False.
        generate_widget_examples (`bool`): If True, generate widget examples from the evaluation or training dataset,
            and compute their similarities. Defaults to True.

    .. tip::

        Install `codecarbon <https://github.com/mlco2/codecarbon>`_ to automatically track carbon emission usage and
        include it in your model cards.

    Example::

        >>> model = SparseEncoder(
        ...     "microsoft/mpnet-base",
        ...     model_card_data=SparseEncoderModelCardData(
        ...         model_id="tomaarsen/se-mpnet-base-allnli",
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
            "sparse-encoder",
            "sparse",
        ]
    )

    # Automatically filled by `SparseEncoderModelCardCallback` and the Trainer directly
    predict_example: list[list[str]] | None = field(default=None, init=False)

    # Computed once, always unchanged
    pipeline_tag: str = field(default=None, init=False)
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)
    model_type: str = field(default="Sparse Encoder", init=False, repr=False)

    # Passed via `register_model` only
    model: SparseEncoder | None = field(default=None, init=False, repr=False)

    def register_model(self, model: SparseEncoder) -> None:
        super().register_model(model)

        if self.task_name is None:
            self.task_name = "semantic search and sparse retrieval"
        if self.pipeline_tag is None:
            self.pipeline_tag = "feature-extraction"

        all_modules = [module.__class__ for module in model.modules() if isinstance(module, Module)]
        model_type = []
        if Router in all_modules:
            model_type += ["Asymmetric"]

        if SparseStaticEmbedding in all_modules:
            model_type += ["Inference-free"]

        if SpladePooling in all_modules:
            model_type += ["SPLADE"]

        if SparseAutoEncoder in all_modules:
            model_type += ["CSR"]

        self.add_tags(map(str.lower, model_type))
        model_type += ["Sparse Encoder"]
        self.model_type = " ".join(model_type)

    def get_model_specific_metadata(self) -> dict[str, Any]:
        similarity_fn_name = "Dot Product"
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
            "max_active_dims": getattr(self.model, "max_active_dims", None),
        }

    def get_default_model_name(self) -> None:
        return self.model_type
