import csv
import gzip
import logging
import os
import random
from datetime import datetime

from datasets import Dataset

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# /print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be divisible by pos_neg_ratio
num_epochs = 1
max_seq_length = 75

# Save path to store our model
output_dir = "output/train_stsb_ct-{}-{}".format(model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


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
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

logging.info(f"Train sentences: {len(train_sentences)}")


# Generate sentence pairs for ContrastiveTensionLoss (replicates ContrastiveTensionDataLoader logic)
def generate_ct_pairs(sentences, pos_neg_ratio):
    """Generate sentence pairs for ContrastiveTensionLoss with the specified pos_neg_ratio.

    This exactly replicates the logic of ContrastiveTensionDataLoader.__iter__():
    - Uses len(batch) % pos_neg_ratio to determine if pair is positive (0) or negative (>0)
    - For positive: (s1, s1) with label=1
    - For negative: (s1, s2) with label=0 where s2 is next sentence
    - Increments sentence_idx after each pair
    """
    pairs = []
    random.shuffle(sentences)
    sentence_idx = 0

    while sentence_idx + 1 < len(sentences):
        s1 = sentences[sentence_idx]
        if len(pairs) % pos_neg_ratio > 0:  # Negative (different) pair
            sentence_idx += 1
            if sentence_idx < len(sentences):
                s2 = sentences[sentence_idx]
                label = 0
            else:
                break
        else:  # Positive (identical pair)
            s2 = sentences[sentence_idx]
            label = 1

        sentence_idx += 1
        pairs.append({"sentence1": s1, "sentence2": s2, "label": label})

    return pairs


logging.info("Generating training pairs...")
train_pairs = generate_ct_pairs(train_sentences, pos_neg_ratio)

train_dataset = Dataset.from_list(train_pairs)
logging.info(f"Generated {len(train_dataset)} training pairs")

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
        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

        if row["split"] == "dev":
            dev_samples.append(inp_example)
        elif row["split"] == "test":
            test_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

# Initialize an SBERT model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# As loss, we use ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0,
    learning_rate=1e-5,
    weight_decay=0,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=False,  # Set to True, if your GPU has optimized FP16 cores
    optim="rmsprop",
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

logging.info("Evaluating on test set")
test_evaluator(model)
