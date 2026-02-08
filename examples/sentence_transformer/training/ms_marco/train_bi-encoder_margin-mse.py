import gzip
import json
import logging
import os
import random
from datetime import datetime

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset
from tqdm import tqdm

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer
from sentence_transformers.losses import MarginMSELoss
from sentence_transformers.models import Pooling, Transformer
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
warmup_steps = 1000
lr = 2e-5
# We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_negs_per_system = 5
use_pretrained_model = False
use_all_queries = False


# Load our embedding model
if use_pretrained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = f"output/train_bi-encoder-margin_mse-{model_name.replace('/', '-')}-batch_size_{train_batch_size}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# Write self to path
os.makedirs(model_save_path, exist_ok=True)


# Read the corpus files, that contain all the passages.
corpus_hf = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")

corpus = dict(zip(corpus_hf["pid"], corpus_hf["text"]))


# Read the train queries.
queries_hf = load_dataset("omkar334/msmarcoranking-queries", split="train")

queries = dict(zip(queries_hf["qid"], queries_hf["text"]))

# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid) to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L6-v2 model
ce_scores_hf = load_dataset("sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2", "list", split="train")

ce_scores = {
    qid: dict(zip(cids, scores))
    for qid, cids, scores in zip(
        ce_scores_hf["query_id"],
        ce_scores_hf["corpus_id"],
        ce_scores_hf["score"],
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
    for line in tqdm(fIn):
        if max_passages > 0 and len(train_queries) >= max_passages:
            break
        data = json.loads(line)

        # Get the positive passage ids
        pos_pids = data["pos"]

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
train_loss = MarginMSELoss(model=model)

args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    max_steps=max_steps,
    per_device_train_batch_size=train_batch_size,
    warmup_ratio=0.1,
    learning_rate=lr,
    save_strategy="steps",
    save_steps=0.001,
    logging_steps=0.01,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

# Train the model
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
    model.push_to_hub(f"{model_name}-bi-encoder-margin-mse")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({model_save_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-bi-encoder-margin-mse')`."
    )
