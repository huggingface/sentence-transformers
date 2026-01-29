"""
This scripts runs the evaluation (dev & test) for the AskUbuntu dataset

Usage:
python eval_askubuntu.py [sbert_model_name_or_path]
"""

import gzip
import logging
import os
import sys

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import RerankingEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model = SentenceTransformer(sys.argv[1])


# Download AskUbuntu and extract training corpus
askubuntu_folder = "data/askubuntu"
training_corpus = os.path.join(askubuntu_folder, "train.unsupervised.txt")


# Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
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
dataset = load_dataset("sentence-transformers/askubuntu")
dev_dataset = dataset["dev"]
test_dataset = dataset["test"]


# Create a dev evaluator
dev_evaluator = RerankingEvaluator(dev_dataset, name="AskUbuntu dev")

logging.info("Dev performance")
dev_evaluator(model)

test_evaluator = RerankingEvaluator(test_dataset, name="AskUbuntu test")
logging.info("Test performance")
test_evaluator(model)
