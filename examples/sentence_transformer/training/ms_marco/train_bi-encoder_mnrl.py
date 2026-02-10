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
from datetime import datetime

import tqdm
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


train_batch_size = 64
max_seq_length = 300  # Max length for passages. Increasing it, requires more GPU memory
model_name = "microsoft/mpnet-base"
max_passages = 0
num_epochs = 1
max_steps = 1e-7
pooling_mode = "mean"
negs_to_use = None
lr = 2e-5
# We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_negs_per_system = 5
use_pretrained_model = False
use_all_queries = False
ce_score_margin = 3.0

# Load our embedding model
if use_pretrained_model:
    logging.info("Using pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Creating new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = "output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}".format(
    model_name.replace("/", "-"),
    ce_score_margin,
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
)
os.makedirs(model_save_path, exist_ok=True)
corpus = load_dataset(
    "sentence-transformers/msmarco-corpus",
    "passage",
    split="train",
)

corpus_dict = dict(zip(corpus["pid"], corpus["text"]))
queries = load_dataset(
    "omkar334/msmarcoranking-queries",
    split="train",
)

query_dict = dict(zip(queries["qid"], queries["text"]))
scores = load_dataset(
    "sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2",
    "list",
    split="train",
)

ce_scores = {
    qid: dict(zip(cids, sc))
    for qid, cids, sc in zip(
        scores["query_id"],
        scores["corpus_id"],
        scores["score"],
    )
}
logging.info("Load CrossEncoder scores dict")

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = hf_hub_download(
    repo_id="sentence-transformers/msmarco-hard-negatives",
    filename="msmarco-hard-negatives.jsonl.gz",
    repo_type="dataset",
)


logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        # Get the positive passage ids
        qid = data["qid"]
        pos_pids = data["pos"]

        if len(pos_pids) == 0:  # Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if negs_to_use is not None:  # Use specific system for negatives
                negs_to_use = negs_to_use.split(",")
            else:  # Use all systems
                negs_to_use = list(data["neg"].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data["neg"]:
                continue

            system_negs = data["neg"][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data["qid"]] = {
                "qid": data["qid"],
                "query": queries[data["qid"]],
                "pos": pos_pids,
                "neg": neg_pids,
            }

del ce_scores

logging.info(f"Train queries: {len(train_queries)}")

anchors = []
positives = []
negatives = []

for q in train_queries.values():
    query_text = q["query"]

    pos_ids = list(q["pos"])
    neg_ids = list(q["neg"])

    # shuffle once
    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    # create pairs
    for pos_id, neg_id in zip(pos_ids, neg_ids):
        anchors.append(query_text)
        positives.append(corpus_dict[pos_id])
        negatives.append(corpus_dict[neg_id])


train_dataset = Dataset.from_dict({
    "anchor": anchors,
    "positive": positives,
    "negative": negatives,
})

logging.info(f"Triplets created: {len(train_dataset)}")

train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Prepare training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_ratio=0.1,
    learning_rate=lr,
    save_strategy="steps",
    save_steps=0.001,
    logging_steps=0.01,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

trainer.train()

model.save_pretrained(model_save_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-bi-encoder-mnrl")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({model_save_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-bi-encoder-mnrl')`."
    )
