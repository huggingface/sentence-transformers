"""
This scripts runs the evaluation (dev & test) for the AskUbuntu dataset

Usage:
python eval_askubuntu.py [sbert_model_name_or_path]
"""

import logging
import sys

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model = SentenceTransformer(sys.argv[1])


# Read datasets
test_dataset = load_dataset("sentence-transformers/askubuntu", split="test").filter(lambda x: x["positive"])
eval_dataset = load_dataset("sentence-transformers/askubuntu", split="dev").filter(lambda x: x["positive"])

# Create a dev evaluator
dev_evaluator = RerankingEvaluator(eval_dataset, name="AskUbuntu dev")

logging.info("Dev performance")
dev_evaluator(model)

test_evaluator = RerankingEvaluator(test_dataset, name="AskUbuntu test")
logging.info("Test performance")
test_evaluator(model)
