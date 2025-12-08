from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class MSEEvaluator(SentenceEvaluator):
    """
    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.

    The MSE is computed between ||teacher.encode(source_sentences) - student.encode(target_sentences)||.

    For multilingual knowledge distillation (https://huggingface.co/papers/2004.09813), source_sentences are in English
    and target_sentences are in a different language like German, Chinese, Spanish...

    Args:
        source_sentences (List[str]): Source sentences to embed with the teacher model.
        target_sentences (List[str]): Target sentences to embed with the student model.
        teacher_model (SentenceTransformer, optional): The teacher model to compute the source sentence embeddings.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 32.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` uses the model's current truncation
            dimension. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import MSEEvaluator
            from datasets import load_dataset

            # Load a model
            student_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            teacher_model = SentenceTransformer('all-mpnet-base-v2')

            # Load any dataset with some texts
            dataset = load_dataset("sentence-transformers/stsb", split="validation")
            sentences = dataset["sentence1"] + dataset["sentence2"]

            # Given queries, a corpus and a mapping with relevant documents, the MSEEvaluator computes different MSE metrics.
            mse_evaluator = MSEEvaluator(
                source_sentences=sentences,
                target_sentences=sentences,
                teacher_model=teacher_model,
                name="stsb-dev",
            )
            results = mse_evaluator(student_model)
            '''
            MSE evaluation (lower = better) on the stsb-dev dataset:
            MSE (*100):  0.805045
            '''
            print(mse_evaluator.primary_metric)
            # => "stsb-dev_negative_mse"
            print(results[mse_evaluator.primary_metric])
            # => -0.8050452917814255
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        teacher_model=None,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super().__init__()
        self.truncate_dim = truncate_dim
        self.target_sentences = target_sentences
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.name = name

        self.csv_file = "mse_evaluation_" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MSE"]
        self.write_csv = write_csv
        self.primary_metric = "negative_mse"

        self.source_embeddings = self.embed_inputs(teacher_model, source_sentences)

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch=-1, steps=-1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        target_embeddings = self.embed_inputs(model, self.target_sentences)

        mse = ((self.source_embeddings - target_embeddings) ** 2).mean()
        mse = mse * 100

        logger.info(f"MSE evaluation (lower = better) on the {self.name} dataset{out_txt}:")
        logger.info(f"MSE (*100):\t{mse:4f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, mse])

        # Return negative score as SentenceTransformers maximizes the performance
        metrics = {"negative_mse": -mse}
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    @property
    def description(self) -> str:
        return "Knowledge Distillation"

    def get_config_dict(self):
        config_dict = {}
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
