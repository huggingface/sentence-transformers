import gzip
import logging
import os
from datetime import datetime

import torch
from datasets import load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models, util

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# /print debug information to stdout

# Some training parameters. We use a batch size of 16, for every positive example we include 8-1=7 negative examples
# Sentences are truncated to 75 word pieces
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be devisible by pos_neg_ratio
max_seq_length = 75
num_epochs = 1

# Download AskUbuntu and extract training corpus
askubuntu_folder = "askubuntu"
output_path = "output/train_askubuntu_ct-{}-{}-{}".format(
    model_name, batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

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
        splits = line.strip().split("\t")
        id = splits[0]
        title = splits[1]
        corpus[id] = title


# Read dev & test dataset
dataset = load_dataset("sentence-transformers/askubuntu")
dev_dataset = dataset["dev"]
test_dataset = dataset["test"]


# Now we need a list of train sentences.
# In this example we simply use all sentences that don't appear in the train/dev set
train_sentences = []
for id, sentence in corpus.items():
    if id not in dev_test_ids:
        train_sentences.append(sentence)

logging.info(f"{len(train_sentences)} train sentences")

# Initialize an SBERT model

word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Train the model

# For ContrastiveTension we need a special data loader to construct batches with the desired properties
train_dataloader = losses.ContrastiveTensionDataLoader(
    train_sentences, batch_size=batch_size, pos_neg_ratio=pos_neg_ratio
)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name="AskUbuntu dev")
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")


logging.info("Start training")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={"lr": 1e-5},
    use_amp=False,  # Set to True, if your GPU has optimized FP16 cores
)

latest_output_path = output_path + "-latest"
model.save(latest_output_path)

# Run test evaluation on the latest model. This is equivalent to not having a dev dataset
model = SentenceTransformer(latest_output_path)
test_evaluator(model)
