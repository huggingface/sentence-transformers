import gzip
import json
import logging
import os
import random
from datetime import datetime

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

from sentence_transformers import LoggingHandler, SentenceTransformer
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
os.makedirs(model_save_path, exist_ok=True)
corpus = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")

corpus_dict = dict(zip(corpus["pid"], corpus["text"]))
queries = load_dataset("omkar334/msmarcoranking-queries", split="train")

query_dict = dict(zip(queries["qid"], queries["text"]))
scores = load_dataset("sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2", "list", split="train")

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


def build_samples():
    with gzip.open(hard_negatives_filepath, "rt") as f:
        for line in f:
            data = json.loads(line)

            pos_pids = data.get("pos", [])
            neg_systems = data.get("neg", {})

            # --- skip bad rows (required) ---
            if not pos_pids or not neg_systems:
                continue

            qid = data["qid"]
            query = query_dict.get(qid)
            if query is None:
                continue

            pos = pos_pids[0]

            negs = []
            for system_negs in neg_systems.values():
                negs.extend(system_negs[:num_negs_per_system])

            if not negs:
                continue

            neg = random.choice(negs)

            yield {
                "anchor": query,
                "positive": corpus_dict[pos],
                "negative": corpus_dict[neg],
                "label": ce_scores[qid][pos] - ce_scores[qid][neg],
            }


train_dataset = Dataset.from_generator(build_samples)

logging.info(f"Training samples: {len(train_dataset)}")


# Loss function
train_loss = MarginMSELoss(model)


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
