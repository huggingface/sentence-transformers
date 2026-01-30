import gzip
import logging
import os
from datetime import datetime

from datasets import Dataset

from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    evaluation,
    losses,
    models,
    util,
)
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

# Some training parameters. We use a batch size of 16, for every positive example we include 8-1=7 negative examples
# Sentences are truncated to 75 word pieces
# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 128
epochs = 1
max_seq_length = 75

# Download AskUbuntu and extract training corpus
askubuntu_folder = "askubuntu"
output_path = "output/train_askubuntu_ct-improved-{}-{}-{}".format(
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
def read_eval_dataset(filepath):
    dataset = []
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:  # Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            dataset.append({
                "query": corpus[query_id],
                "positive": [corpus[pid] for pid in relevant_id],
                "negative": [corpus[pid] for pid in negative_ids],
            })
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    return dataset


dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "dev.txt"))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "test.txt"))


# Now we need a list of train sentences.
# In this example we simply use all sentences that don't appear in the train/dev set
train_sentences = []
for id, sentence in corpus.items():
    if id not in dev_test_ids:
        train_sentences.append({"sentence1": sentence, "sentence2": sentence})

logging.info(f"{len(train_sentences)} train sentences")
train_dataset = Dataset.from_list(train_sentences)

# Initialize an SBERT model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Loss
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model)

# Evaluators
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name="AskUbuntu dev")
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")


logging.info("Start training")

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0.1,
    eval_steps=0.1,
    logging_steps=0.01,
    learning_rate=5e-5,
    save_strategy="no",
    fp16=True,
)


# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=train_loss,
)

logging.info("Start training")
trainer.train()

latest_output_path = output_path + "-latest"

# Run test evaluation on the latest model. This is equivalent to not having a dev dataset
test_evaluator(model)

model.save_pretrained(latest_output_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-ct-improved")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({latest_output_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-ct-improved')`."
    )
