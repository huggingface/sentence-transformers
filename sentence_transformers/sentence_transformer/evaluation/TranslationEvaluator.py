from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from sentence_transformers.base.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import pytorch_cos_sim

if TYPE_CHECKING:
    from torch import Tensor

    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class TranslationEvaluator(SentenceEvaluator):
    """
    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that fr_i is the translation of en_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accuracy in both directions

    The labels need to indicate the similarity between the sentences.

    Args:
        source_sentences (List[str]): List of sentences in the source language.
        target_sentences (List[str]): List of sentences in the target language.
        show_progress_bar (bool): Whether to show a progress bar when computing embeddings. Defaults to False.
        batch_size (int): The batch size to compute sentence embeddings. Defaults to 16.
        name (str): The name of the evaluator. Defaults to an empty string.
        print_wrong_matches (bool): Whether to print incorrect matches. Defaults to False.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. If None, the model's
            current truncation dimension will be used. Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

            # Load a parallel sentences dataset
            dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

            # Initialize the TranslationEvaluator using the same texts from two languages
            translation_evaluator = TranslationEvaluator(
                source_sentences=dataset["english"],
                target_sentences=dataset["non_english"],
                name="news-commentary-en-nl",
            )
            results = translation_evaluator(model)
            '''
            Evaluating translation matching Accuracy of the model on the news-commentary-en-nl dataset:
            Accuracy src2trg: 90.80
            Accuracy trg2src: 90.40
            '''
            print(translation_evaluator.primary_metric)
            # => "news-commentary-en-nl_mean_accuracy"
            print(results[translation_evaluator.primary_metric])
            # => 0.906
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 16,
        name: str = "",
        print_wrong_matches: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super().__init__()
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.print_wrong_matches = print_wrong_matches
        self.truncate_dim = truncate_dim

        assert len(self.source_sentences) == len(self.target_sentences)

        if name:
            name = "_" + name

        self.csv_file = "translation_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "src2trg", "trg2src"]
        self.write_csv = write_csv
        self.primary_metric = "mean_accuracy"

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
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

        logger.info(f"Evaluating translation matching Accuracy of the model on the {self.name} dataset{out_txt}:")

        embeddings1 = torch.stack(self.embed_inputs(model, self.source_sentences))
        embeddings2 = torch.stack(self.embed_inputs(model, self.target_sentences))

        cos_sims = pytorch_cos_sim(embeddings1, embeddings2).detach().cpu().numpy()

        correct_src2trg = 0
        correct_trg2src = 0

        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])

            if i == max_idx:
                correct_src2trg += 1
            elif self.print_wrong_matches:
                print("\nIncorrect  : Source", i, "is most similar to target", max_idx, "instead of target", i)
                print("Source     :", self.source_sentences[i])
                print("Pred Target:", self.target_sentences[max_idx], f"(Score: {cos_sims[i][max_idx]:.4f})")
                print("True Target:", self.target_sentences[i], f"(Score: {cos_sims[i][i]:.4f})")

                results = enumerate(cos_sims[i])
                results = sorted(results, key=lambda x: x[1], reverse=True)
                for idx, score in results[:5]:
                    print("\t", idx, f"(Score: {score:.4f})", self.target_sentences[idx])

        cos_sims = cos_sims.T
        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])
            if i == max_idx:
                correct_trg2src += 1

        acc_src2trg = correct_src2trg / len(cos_sims)
        acc_trg2src = correct_trg2src / len(cos_sims)

        logger.info(f"Accuracy src2trg: {acc_src2trg * 100:.2f}")
        logger.info(f"Accuracy trg2src: {acc_trg2src * 100:.2f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc_src2trg, acc_trg2src])

        metrics = {
            "src2trg_accuracy": acc_src2trg,
            "trg2src_accuracy": acc_trg2src,
            "mean_accuracy": (acc_src2trg + acc_trg2src) / 2,
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> list[Tensor]:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=False,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    def get_config_dict(self):
        config_dict = {}
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
