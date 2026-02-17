from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING

from sentence_transformers.base.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import paraphrase_mining

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class ParaphraseMiningEvaluator(SentenceEvaluator):
    """
    Given a large set of sentences, this evaluator performs paraphrase (duplicate) mining and
    identifies the pairs with the highest similarity. It compare the extracted paraphrase pairs
    with a set of gold labels and computes the F1 score.

    Args:
        sentences_map (Dict[str, str]): A dictionary that maps sentence-ids to sentences.
            For example, sentences_map[id] => sentence.
        duplicates_list (List[Tuple[str, str]], optional): A list with id pairs [(id1, id2), (id1, id5)]
            that identifies the duplicates / paraphrases in the sentences_map. Defaults to None.
        duplicates_dict (Dict[str, Dict[str, bool]], optional): A default dictionary mapping [id1][id2]
            to true if id1 and id2 are duplicates. Must be symmetric, i.e., if [id1][id2] => True,
            then [id2][id1] => True. Defaults to None.
        add_transitive_closure (bool, optional): If true, it adds a transitive closure,
            i.e. if dup[a][b] and dup[b][c], then dup[a][c]. Defaults to False.
        query_chunk_size (int, optional): To identify the paraphrases, the cosine-similarity between
            all sentence-pairs will be computed. As this might require a lot of memory, we perform
            a batched computation. query_chunk_size sentences will be compared against up to
            corpus_chunk_size sentences. In the default setting, 5000 sentences will be grouped
            together and compared up-to against 100k other sentences. Defaults to 5000.
        corpus_chunk_size (int, optional): The corpus will be batched, to reduce the memory requirement.
            Defaults to 100000.
        max_pairs (int, optional): We will only extract up to max_pairs potential paraphrase candidates.
            Defaults to 500000.
        top_k (int, optional): For each query, we extract the top_k most similar pairs and add it to a sorted list.
            I.e., for one sentence we cannot find more than top_k paraphrases. Defaults to 100.
        show_progress_bar (bool, optional): Output a progress bar. Defaults to False.
        batch_size (int, optional): Batch size for computing sentence embeddings. Defaults to 16.
        name (str, optional): Name of the experiment. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to.
            `None` uses the model's current truncation dimension. Defaults to None.

    Example:
        ::

            from datasets import load_dataset
            from sentence_transformers.sentence_transformer.model import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the Quora Duplicates Mining dataset
            questions_dataset = load_dataset("sentence-transformers/quora-duplicates-mining", "questions", split="dev")
            duplicates_dataset = load_dataset("sentence-transformers/quora-duplicates-mining", "duplicates", split="dev")

            # Create a mapping from qid to question & a list of duplicates (qid1, qid2)
            qid_to_questions = dict(zip(questions_dataset["qid"], questions_dataset["question"]))
            duplicates = list(zip(duplicates_dataset["qid1"], duplicates_dataset["qid2"]))

            # Initialize the paraphrase mining evaluator
            paraphrase_mining_evaluator = ParaphraseMiningEvaluator(
                sentences_map=qid_to_questions,
                duplicates_list=duplicates,
                name="quora-duplicates-dev",
            )
            results = paraphrase_mining_evaluator(model)
            '''
            Paraphrase Mining Evaluation of the model on the quora-duplicates-dev dataset:
            Number of candidate pairs: 250564
            Average Precision: 56.51
            Optimal threshold: 0.8325
            Precision: 52.76
            Recall: 59.19
            F1: 55.79
            '''
            print(paraphrase_mining_evaluator.primary_metric)
            # => "quora-duplicates-dev_average_precision"
            print(results[paraphrase_mining_evaluator.primary_metric])
            # => 0.5650940787776353
    """

    def __init__(
        self,
        sentences_map: dict[str, str],
        duplicates_list: list[tuple[str, str]] | None = None,
        duplicates_dict: dict[str, dict[str, bool]] | None = None,
        add_transitive_closure: bool = False,
        query_chunk_size: int = 5000,
        corpus_chunk_size: int = 100000,
        max_pairs: int = 500000,
        top_k: int = 100,
        show_progress_bar: bool = False,
        batch_size: int = 16,
        name: str = "",
        write_csv: bool = True,
        truncate_dim: int | None = None,
    ):
        super().__init__()
        self.sentences = []
        self.ids = []

        for id, sentence in sentences_map.items():
            self.sentences.append(sentence)
            self.ids.append(id)

        self.name = name
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.query_chunk_size = query_chunk_size
        self.corpus_chunk_size = corpus_chunk_size
        self.max_pairs = max_pairs
        self.top_k = top_k
        self.truncate_dim = truncate_dim

        self.duplicates = duplicates_dict if duplicates_dict is not None else defaultdict(lambda: defaultdict(bool))
        if duplicates_list is not None:
            for id1, id2 in duplicates_list:
                if id1 in sentences_map and id2 in sentences_map:
                    self.duplicates[id1][id2] = True
                    self.duplicates[id2][id1] = True

        # Add transitive closure
        if add_transitive_closure:
            self.duplicates = self.add_transitive_closure(self.duplicates)

        positive_key_pairs = set()
        for key1 in self.duplicates:
            for key2 in self.duplicates[key1]:
                if (
                    key1 in sentences_map
                    and key2 in sentences_map
                    and (self.duplicates[key1][key2] or self.duplicates[key2][key1])
                ):
                    positive_key_pairs.add(tuple(sorted([key1, key2])))

        self.total_num_duplicates = len(positive_key_pairs)

        if name:
            name = "_" + name

        self.csv_file: str = "paraphrase_mining_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "precision", "recall", "f1", "threshold", "average_precision"]
        self.write_csv = write_csv
        self.primary_metric = "average_precision"

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

        logger.info(f"Paraphrase Mining Evaluation of the model on the {self.name} dataset{out_txt}:")

        # Compute embedding for the sentences
        pairs_list = paraphrase_mining(
            model,
            self.sentences,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
            query_chunk_size=self.query_chunk_size,
            corpus_chunk_size=self.corpus_chunk_size,
            max_pairs=self.max_pairs,
            top_k=self.top_k,
            truncate_dim=self.truncate_dim,
        )

        logger.info("Number of candidate pairs: " + str(len(pairs_list)))

        # Compute F1 score and Average Precision
        n_extract = n_correct = 0
        threshold = 0
        best_f1 = best_recall = best_precision = 0

        average_precision = 0

        for idx in range(len(pairs_list)):
            score, i, j = pairs_list[idx]
            id1 = self.ids[i]
            id2 = self.ids[j]

            # Compute optimal threshold and F1-score
            n_extract += 1
            if self.duplicates[id1][id2] or self.duplicates[id2][id1]:
                n_correct += 1
                precision = n_correct / n_extract
                recall = n_correct / self.total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                average_precision += precision
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (pairs_list[idx][0] + pairs_list[min(idx + 1, len(pairs_list) - 1)][0]) / 2

        average_precision = average_precision / self.total_num_duplicates

        logger.info(f"Average Precision: {average_precision * 100:.2f}")
        logger.info(f"Optimal threshold: {threshold:.4f}")
        logger.info(f"Precision: {best_precision * 100:.2f}")
        logger.info(f"Recall: {best_recall * 100:.2f}")
        logger.info(f"F1: {best_f1 * 100:.2f}\n")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold, average_precision])
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, best_precision, best_recall, best_f1, threshold, average_precision])

        metrics = {
            "average_precision": average_precision,
            "f1": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "threshold": threshold,
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    @staticmethod
    def add_transitive_closure(graph):
        nodes_visited = set()
        for a in list(graph.keys()):
            if a not in nodes_visited:
                connected_subgraph_nodes = set()
                connected_subgraph_nodes.add(a)

                # Add all nodes in the connected graph
                neighbor_nodes_queue = list(graph[a])
                while len(neighbor_nodes_queue) > 0:
                    node = neighbor_nodes_queue.pop(0)
                    if node not in connected_subgraph_nodes:
                        connected_subgraph_nodes.add(node)
                        neighbor_nodes_queue.extend(graph[node])

                # Ensure transitivity between all nodes in the graph
                connected_subgraph_nodes = list(connected_subgraph_nodes)
                for i in range(len(connected_subgraph_nodes) - 1):
                    for j in range(i + 1, len(connected_subgraph_nodes)):
                        graph[connected_subgraph_nodes[i]][connected_subgraph_nodes[j]] = True
                        graph[connected_subgraph_nodes[j]][connected_subgraph_nodes[i]] = True

                        nodes_visited.add(connected_subgraph_nodes[i])
                        nodes_visited.add(connected_subgraph_nodes[j])
        return graph

    def get_config_dict(self):
        config_dict = {
            "add_transitive_closure": self.add_transitive_closure,
            "max_pairs": self.max_pairs,
            "top_k": self.top_k,
        }
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
