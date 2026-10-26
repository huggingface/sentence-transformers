from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.util.distributed import get_rank

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingExportEvaluator(BaseEvaluator):
    """Save dense sentence embeddings for analysis during or after training.

    Each call writes a NumPy archive containing ``embeddings`` and ``sentences`` in
    matching row order. The filename includes the epoch and step, so different
    checkpoints can be compared. Calling with the same name, epoch, and step
    overwrites the previous archive.

    This evaluator returns no metrics. Combine it with a scoring evaluator using
    :class:`~sentence_transformers.base.evaluation.SequentialEvaluator` when using
    evaluation metrics for checkpoint selection.

    Args:
        sentences: Texts to encode and include in the archive. Must not be empty.
        name: Optional filename prefix, for example ``"queries"``. Must not contain
            path separators. Defaults to an empty string.
        batch_size: Encoding batch size. Defaults to 32.
        show_progress_bar: Whether to display encoding progress. Defaults to False.
        encode_kwargs: Additional arguments to ``model.encode``, for example
            ``{"prompt_name": "query", "normalize_embeddings": True}``. Output
            format arguments are managed by this evaluator. Defaults to None.

    Example::

        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer.evaluation import EmbeddingExportEvaluator

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        evaluator = EmbeddingExportEvaluator(["A cat sits outside.", "A dog plays in the park."], name="dev")
        evaluator(model, output_path="evaluation", epoch=1, steps=100)
        with np.load("evaluation/dev_embeddings_epoch_1_steps_100.npz", allow_pickle=False) as data:
            embeddings = data["embeddings"]
            sentences = data["sentences"]
    """

    def __init__(
        self,
        sentences: list[str],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not sentences:
            raise ValueError("sentences must not be empty.")
        if name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError("name must be a filename prefix without path separators.")
        self.sentences = list(sentences)
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.encode_kwargs = dict(encode_kwargs or {})
        reserved = {"convert_to_numpy", "convert_to_tensor", "output_value", "batch_size", "show_progress_bar"}
        if conflicts := reserved.intersection(self.encode_kwargs):
            raise ValueError(f"encode_kwargs cannot override evaluator arguments: {sorted(conflicts)}")

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if output_path is None and get_rank() == 0:
            raise ValueError("EmbeddingExportEvaluator requires output_path to save embeddings.")
        embeddings = model.encode(
            self.sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            convert_to_tensor=False,
            output_value="sentence_embedding",
            **self.encode_kwargs,
        )
        # Sharded training evaluates on every rank; only rank zero receives an output path.
        if output_path is None:
            return {}
        directory = Path(output_path)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.name}_" if self.name else ""
        path = directory / f"{prefix}embeddings_epoch_{epoch}_steps_{steps}.npz"
        np.savez(path, embeddings=embeddings, sentences=np.asarray(self.sentences, dtype=str))
        logger.info("Saved %s embeddings to %s", len(self.sentences), path)
        return {}

    def get_config_dict(self) -> dict[str, Any]:
        return {"name": self.name, "num_sentences": len(self.sentences)}
