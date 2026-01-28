"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_simcse_from_file.py path/to/sentences.txt

"""

import gzip
import logging
import math
import sys
from datetime import datetime

import tqdm
from datasets import Dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
# /print debug information to stdout

# Training parameters
model_name = "distilroberta-base"
train_batch_size = 128
max_seq_length = 32
num_epochs = 1

# Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print(f"Run this script with: python {sys.argv[0]} path/to/sentences.txt")
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ""
if len(sys.argv) >= 3:
    output_name = "-" + sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = "output/train_simcse{}-{}".format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read the train corpus #################
train_samples = []
with (
    gzip.open(filepath, "rt", encoding="utf8") if filepath.endswith(".gz") else open(filepath, encoding="utf8")
) as fIn:
    for line in tqdm.tqdm(fIn, desc="Read file"):
        line = line.strip()
        if len(line) >= 10:
            train_samples.append({"sentence1": line, "sentence2": line})

logging.info(f"Train sentences: {len(train_samples)}")
train_dataset = Dataset.from_list(train_samples)

# We train our model using the MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model)

# 10% of train data for warm-up
num_train_samples = len(train_dataset)
steps_per_epoch = num_train_samples // train_batch_size
total_steps = steps_per_epoch * num_epochs
warmup_steps = math.ceil(total_steps * 0.1)
logging.info(f"Warmup-steps: {warmup_steps}")

# Prepare training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_steps=warmup_steps,
    learning_rate=5e-5,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    fp16=False,  # Set to True, if your GPU supports FP16 cores
    optim="adamw_torch",
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

trainer.train()
