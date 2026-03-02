import logging
import random

import tqdm
from datasets import Dataset, load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MarginMSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Quiet httpx logs


train_batch_size = 64
max_seq_length = 300  # Max length for passages. Increasing it, requires more GPU memory
model_name = "microsoft/mpnet-base"
num_epochs = 1
max_steps = -1
lr = 2e-5

# We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_negs_per_system = 5
num_negatives = 5

# Load our embedding model
logging.info("Using pretrained SBERT model")
model = SentenceTransformer(model_name)
model.max_seq_length = max_seq_length

# Map PID -> text
corpus = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
corpus_dict = dict(zip(corpus["pid"], corpus["text"]))

# Map QID -> query text
queries = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
query_dict = dict(zip(queries["qid"], queries["text"]))

# Map QID -> {PID: CE score}
scores = load_dataset("sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2", "list", split="train")
ce_scores = {
    qid: dict(zip(cids, sc)) for qid, cids, sc in zip(scores["query_id"], scores["corpus_id"], scores["score"])
}
logging.info("Load CrossEncoder scores dict")

# Datasets with 50 hard negatives mined per query using different models
SYSTEMS = {
    "bm25": "sentence-transformers/msmarco-bm25",
    "msmarco-distilbert-base-tas-b": "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-base-v3": "sentence-transformers/msmarco-msmarco-distilbert-base-v3",
    "msmarco-MiniLM-L-6-v3": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
    "distilbert-margin_mse-cls-dot-v2": "sentence-transformers/msmarco-distilbert-margin-mse-cls-dot-v2",
    "distilbert-margin_mse-cls-dot-v1": "sentence-transformers/msmarco-distilbert-margin-mse-cls-dot-v1",
    "distilbert-margin_mse-mean-dot-v1": "sentence-transformers/msmarco-distilbert-margin-mse-mean-dot-v1",
    "mpnet-margin_mse-mean-v1": "sentence-transformers/msmarco-mpnet-margin-mse-mean-v1",
    "co-condenser-margin_mse-cls-v1": "sentence-transformers/msmarco-co-condenser-margin-mse-cls-v1",
    "distilbert-margin_mse-mnrl-mean-v1": "sentence-transformers/msmarco-distilbert-margin-mse-mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v1": "sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v2": "sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v2",
    "co-condenser-margin_mse-sym_mnrl-mean-v1": "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
}

train_data = {}
for system_key, repo_id in SYSTEMS.items():
    print(f"Loading {system_key}...")
    dataset = load_dataset(repo_id, "triplet-50-ids", split="train")

    for row in tqdm.tqdm(dataset, desc=f"Processing {system_key}"):
        qid = row.pop("query")
        pos_pid = row.pop("positive")
        neg_pids = list(row.values())  # All remaining columns are negatives
        existing_neg_pids = set(train_data[qid]["neg_pids"]) if qid in train_data else set()
        pos_ce_score = ce_scores[qid][pos_pid]
        valid_neg_pids = []
        valid_neg_labels = []

        for neg_pid in neg_pids:
            if neg_pid in existing_neg_pids or neg_pid not in ce_scores[qid]:
                continue

            valid_neg_pids.append(neg_pid)
            valid_neg_labels.append(pos_ce_score - ce_scores[qid][neg_pid])
            existing_neg_pids.add(neg_pid)
            if len(valid_neg_pids) >= num_negs_per_system:
                break
        if qid not in train_data:
            train_data[qid] = {"qid": qid, "pid": pos_pid, "neg_pids": valid_neg_pids, "neg_labels": valid_neg_labels}
        else:
            train_data[qid]["neg_pids"].extend(valid_neg_pids)
            train_data[qid]["neg_labels"].extend(valid_neg_labels)

train_data = {qid: data for qid, data in train_data.items() if data["neg_pids"]}
logging.info(f"Kept {len(train_data)} queries with negatives")

train_dataset = Dataset.from_list(list(train_data.values()))


def ids_to_text_transform(batch):
    sampled = [
        random.sample(list(zip(neg_pids, neg_labels)), num_negatives)
        for neg_pids, neg_labels in zip(batch["neg_pids"], batch["neg_labels"])
    ]
    neg_pid_lists, label_lists = zip(*[zip(*s) for s in sampled])
    return {
        "anchor": [query_dict[qid] for qid in batch["qid"]],
        "positive": [corpus_dict[pid] for pid in batch["pid"]],
        **{
            f"negative_{idx}": [corpus_dict[pid] for pid in neg_ids] for idx, neg_ids in enumerate(zip(*neg_pid_lists))
        },
        "label": list(label_lists),
    }


train_dataset.set_transform(ids_to_text_transform)

# Loss function
loss = MarginMSELoss(model)

# Prepare training arguments
short_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
run_name = f"{short_model_name}-msmarco-margin-mse"
args = SentenceTransformerTrainingArguments(
    output_dir=f"output/{run_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_ratio=0.1,
    learning_rate=lr,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=0.1,
    logging_steps=0.01,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

final_model_path = f"output/{run_name}/final"
model.save_pretrained(final_model_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(f"{run_name}")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_model_path!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
