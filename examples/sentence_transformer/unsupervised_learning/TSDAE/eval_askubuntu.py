"""
This scripts runs the evaluation (dev & test) for the AskUbuntu dataset

Usage:
python eval_askubuntu.py [sbert_model_name_or_path]
"""

import gzip
import logging
import os
import sys

from datasets import Dataset

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model = SentenceTransformer(sys.argv[1])


################# Download AskUbuntu and extract training corpus  #################
askubuntu_folder = "data/askubuntu"
training_corpus = os.path.join(askubuntu_folder, "train.unsupervised.txt")


## Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
for filename in ["text_tokenized.txt.gz", "dev.txt", "test.txt", "train_random.txt"]:
    filepath = os.path.join(askubuntu_folder, filename)
    if not os.path.exists(filepath):
        util.http_get("https://github.com/taolei87/askubuntu/raw/master/" + filename, filepath)

# Read the corpus
corpus = {}
dev_test_ids = set()
with gzip.open(os.path.join(askubuntu_folder, "text_tokenized.txt.gz"), "rt", encoding="utf8") as fIn:
    for line in fIn:
        id, title, *_ = line.strip().split("\t")
        corpus[id] = title


# Read dev & test dataset
def read_eval_dataset(filepath) -> Dataset:
    data = {
        "query": [],
        "positive": [],
        "negative": [],
    }
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:  # Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            data["query"].append(corpus[query_id])
            data["positive"].append([corpus[pid] for pid in relevant_id])
            data["negative"].append([corpus[pid] for pid in negative_ids])
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    dataset = Dataset.from_dict(data)
    return dataset


dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "dev.txt"))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "test.txt"))


# Create a dev evaluator
dev_evaluator = RerankingEvaluator(dev_dataset, name="AskUbuntu dev")

logging.info("Dev performance")
dev_evaluator(model)

test_evaluator = RerankingEvaluator(test_dataset, name="AskUbuntu test")
logging.info("Test performance")
test_evaluator(model)
