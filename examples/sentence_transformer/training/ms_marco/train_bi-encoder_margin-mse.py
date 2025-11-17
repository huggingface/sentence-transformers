import argparse
import gzip
import json
import logging
import os
import pickle
import random
import sys
import tarfile
from datetime import datetime
from shutil import copyfile

import tqdm
from torch.utils.data import DataLoader, Dataset

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used? Multiple systems separated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
args = parser.parse_args()

logging.info(str(args))


# The  model we want to fine-tune
train_batch_size = (
    args.train_batch_size
)  # Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it, requires more GPU memory

num_negs_per_system = (
    args.num_negs_per_system
)  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = f"output/train_bi-encoder-margin_mse-{model_name.replace('/', '-')}-batch_size_{train_batch_size}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, "train_script.py")
copyfile(__file__, train_script_path)
with open(train_script_path, "a") as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))


### Now we read the MS Marco dataset
data_folder = "msmarco-data"

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L6-v2 model
ce_scores_file = os.path.join(data_folder, "cross-encoder-ms-marco-MiniLM-L6-v2-scores.pkl.gz")
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get(
        "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
        ce_scores_file,
    )

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, "rb") as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get(
        "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
        hard_negatives_filepath,
    )


logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn):
        if max_passages > 0 and len(train_queries) >= max_passages:
            break
        data = json.loads(line)

        # Get the positive passage ids
        pos_pids = data["pos"]

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:  # Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:  # Use all systems
                negs_to_use = list(data["neg"].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data["neg"]:
                continue

            system_negs = data["neg"][system_name]
            negs_added = 0
            for pid in system_negs:
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data["qid"]] = {
                "qid": data["qid"],
                "query": queries[data["qid"]],
                "pos": pos_pids,
                "neg": neg_pids,
            }

logging.info(f"Train queries: {len(train_queries)}")


# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

        for qid in self.queries:
            self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
            self.queries[qid]["neg"] = list(self.queries[qid]["neg"])
            random.shuffle(self.queries[qid]["neg"])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]
        qid = query["qid"]

        if len(query["pos"]) > 0:
            pos_id = query["pos"].pop(0)  # Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query["pos"].append(pos_id)
        else:  # We only have negatives, use two negs
            pos_id = query["neg"].pop(0)  # Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query["neg"].append(pos_id)

        # Get a negative passage
        neg_id = query["neg"].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query["neg"].append(neg_id)

        pos_score = self.ce_scores[qid][pos_id]
        neg_score = self.ce_scores[qid][neg_id]

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score - neg_score)

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MarginMSELoss(model=model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=args.warmup_steps,
    use_amp=True,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=10000,
    optimizer_params={"lr": args.lr},
)

# Train latest model
model.save(model_save_path)
