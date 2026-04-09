"""
Given a dataset with parallel sentences, one "english" column and one "non_english" column, this script evaluates a model on the translation task.
Given a sentence in the "english" column, the model should find the correct translation in the "non_english" column, based on just the embeddings.

It then computes an accuracy over all possible source sentences src_i. Equivalently, it computes also the accuracy for the other direction.
A high accuracy score indicates that the model is able to find the correct translation out of a large pool with sentences.

Good options for datasets are:
* sentence-transformers/parallel-sentences-wikimatrix
* sentence-transformers/parallel-sentences-tatoeba
* sentence-transformers/parallel-sentences-talks

As these have development sets.

Usage:
python examples/sentence_transformer/evaluation/evaluation_translation_matching.py [model_name_or_path] [dataset_name] [subset1] [subset2] ...

For example:
python examples/sentence_transformer/evaluation/evaluation_translation_matching.py sentence-transformers/distiluse-base-multilingual-cased sentence-transformers/parallel-sentences-tatoeba en-ar en-de en-nl
"""

import logging
import sys

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, evaluation

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1]
dataset_name = sys.argv[2]
subsets = sys.argv[3:]
inference_batch_size = 32

model = SentenceTransformer(model_name)

for subset in subsets:
    dataset = load_dataset(dataset_name, subset)
    datasets = {}
    if dataset.column_names == ["train"]:
        num_samples = min(5000, len(dataset["train"]))
        datasets[f"train[:{num_samples}]"].append(dataset["train"].select(range(num_samples)))
    else:
        for split, sub_dataset in dataset.items():
            if split != "train":
                datasets[split] = sub_dataset

    for split, sub_dataset in datasets.items():
        logging.info(f"{dataset_name}, subset={subset}, split={split}, num_samples={len(sub_dataset)}")
        translation_evaluator = evaluation.TranslationEvaluator(
            sub_dataset["english"],
            sub_dataset["non_english"],
            name=f"{dataset_name}-{subset}-{split}",
            batch_size=inference_batch_size,
        )
        translation_evaluator(model)
