import csv
import gzip
import logging
import os
from datetime import datetime

from datasets import Dataset

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    losses,
    models,
    util,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
# print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 128
epochs = 1
max_seq_length = 75

# Save path to store our model
model_save_path = "output/training_stsb_ct-improved-{}-{}".format(
    model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Train sentences
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = "data/wiki1m_for_simcse.txt"
if not os.path.exists(wikipedia_dataset_path):
    util.http_get(
        "https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt",
        wikipedia_dataset_path,
    )

# train_sentences are simply your list of sentences
train_sentences = []
with open(wikipedia_dataset_path, encoding="utf8") as fIn:
    for line in fIn:
        s = line.strip()
        if s:
            # For CT In-Batch Negatives we use identical pairs
            train_sentences.append({"sentence1": s, "sentence2": s})

train_dataset = Dataset.from_list(train_sentences)
logging.info(f"Train samples: {len(train_dataset)}")

# Download and load STSb
data_folder = "data/stsbenchmark"
sts_dataset_path = f"{data_folder}/stsbenchmark.tsv.gz"

if not os.path.exists(sts_dataset_path):
    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
        inp_example = (row["sentence1"], row["sentence2"], score)
        if row["split"] == "dev":
            dev_samples.append(inp_example)
        elif row["split"] == "test":
            test_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator(
    [s1 for s1, _, _ in dev_samples],
    [s2 for _, s2, _ in dev_samples],
    [score for _, _, score in dev_samples],
    name="sts-dev",
)
test_evaluator = EmbeddingSimilarityEvaluator(
    [s1 for s1, _, _ in test_samples],
    [s2 for _, s2, _ in test_samples],
    [score for _, _, score in test_samples],
    name="sts-test",
)

# Initialize an SBERT model #################
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Loss
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model, scale=1, similarity_fct=util.dot_score)

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=1000,
    learning_rate=5e-5,
    save_strategy="no",
    logging_steps=100,
    fp16=True,
)

# Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=train_loss,
)

# Train the model
trainer.train()
# Load the model and evaluate on test set
model = SentenceTransformer(model_save_path)
test_evaluator(model)
