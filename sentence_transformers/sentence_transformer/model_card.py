from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from sentence_transformers.base.model_card import BaseModelCardCallback, BaseModelCardData
from sentence_transformers.sentence_transformer.modules import StaticEmbedding

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


class SentenceTransformerModelCardCallback(BaseModelCardCallback):
    pass


@dataclass
class SentenceTransformerModelCardData(BaseModelCardData):
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
    task_name: str = (
        "semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more"
    )
    tags: list[str] = field(
        default_factory=lambda: [
            "sentence-transformers",
            "sentence-similarity",
            "feature-extraction",
            "dense",
        ]
    )

    # Computed once, always unchanged
    template_path: Path = field(default=Path(__file__).parent / "model_card_template.md", init=False, repr=False)

    # Passed via `register_model` only
    model: SentenceTransformer | None = field(default=None, init=False, repr=False)

    def try_to_set_base_model(self):
        super().try_to_set_base_model()
        if isinstance(self.model[0], StaticEmbedding) and self.base_model is None:
            if self.model[0].base_model:
                self.set_base_model(self.model[0].base_model)

    def get_model_specific_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_specific_metadata()
        similarity_fn_name = "Cosine Similarity"
        if self.model.similarity_fn_name:
            similarity_fn_name = {
                "cosine": "Cosine Similarity",
                "dot": "Dot Product",
                "euclidean": "Euclidean Distance",
                "manhattan": "Manhattan Distance",
            }.get(self.model.similarity_fn_name, self.model.similarity_fn_name.replace("_", " ").title())
        metadata.update(
            {
                "output_dimensionality": self.model.get_embedding_dimension(),
                "similarity_fn_name": similarity_fn_name,
            }
        )
        return metadata


@deprecated(
    "The `ModelCardCallback` has been renamed to `SentenceTransformerModelCardCallback` and the former is now deprecated. Please use `SentenceTransformerModelCardCallback` instead."
)
class ModelCardCallback(SentenceTransformerModelCardCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
