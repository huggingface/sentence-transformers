"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
python train_bi-encoder-v3.py
"""

import gzip
import json
import logging
import os
import random
import tarfile
from datetime import datetime

import tqdm
from torch.utils.data import DataLoader, Dataset

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

# The  model we want to fine-tune
model_name = "distilbert-base-uncased"

train_batch_size = 100  # Increasing the train batch size improves the model performance, but requires more GPU memory
max_seq_length = 300  # Max length for passages. Increasing it, requires more GPU memory
ce_score_margin = 3  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = (
    5  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
)
num_epochs = 10  # Number of epochs we want to train

# We construct the SentenceTransformer bi-encoder from scratch
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = "output/train_bi-encoder-v3-{}-margin_{:.1f}-{}".format(
    model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


### Now we read the MS Marco dataset
data_folder = "msmarco-data"

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get("https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query


# Read our training file: msmarco-hard-negatives.jsonl.gz contains all queries and hard-negatives that were mined with different systems
# For each positive and mined-hard negative passage, we have a Cross-Encoder score from the cross-encoder/ms-marco-MiniLM-L-6-v2 model
# This  Cross-Encoder score allows to de-noise our hard-negatives by requiring that their CE-score is below a certain threshold
train_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get("https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz", train_filepath)


#### Create our training data
logging.info("Read train dataset")
train_queries = {}
with gzip.open(train_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        # Get the positive passage ids
        pos_pids = [item["pid"] for item in data["pos"]]
        pos_min_ce_score = min([item["ce-score"] for item in data["pos"]])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        for system_negs in data["neg"].values():
            negs_added = 0
            for item in system_negs:
                if item["ce-score"] > ce_score_threshold:
                    continue

                pid = item["pid"]
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data["qid"]] = {"query": queries[data["qid"]], "pos": pos_pids, "neg": neg_pids}

logging.info(f"Train queries: {len(train_queries)}")


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
            self.queries[qid]["neg"] = list(self.queries[qid]["neg"])
            random.shuffle(self.queries[qid]["neg"])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]

        pos_id = query["pos"].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query["pos"].append(pos_id)

        neg_id = query["neg"].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query["neg"].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=1000, use_amp=True)

# Save the model
model.save(model_save_path)
