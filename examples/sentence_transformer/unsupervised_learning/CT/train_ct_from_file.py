"""
This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

CT will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

Usage:
python train_ct_from_file.py path/to/sentences.txt

"""

import gzip
import logging
import random
import sys
import traceback
from datetime import datetime

import tqdm
from datasets import Dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be devisible by pos_neg_ratio
num_epochs = 1
max_seq_length = 75

# Input file path (a text file, each line a sentence)
if len(sys.argv) < 2:
    print(f"Run this script with: python {sys.argv[0]} path/to/sentences.txt")
    exit()

filepath = sys.argv[1]

# Save path to store our model
output_name = ""
if len(sys.argv) >= 3:
    output_name = "-" + sys.argv[2].replace(" ", "_").replace("/", "_").replace("\\", "_")

model_output_path = "output/train_ct{}-{}".format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read the train corpus
train_sentences = []
with (
    gzip.open(filepath, "rt", encoding="utf8") if filepath.endswith(".gz") else open(filepath, encoding="utf8") as fIn
):
    for line in tqdm.tqdm(fIn, desc="Read file"):
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

train_dataset = Dataset.from_dict({"text1": train_sentences})
logging.info(f"Train sentences: {len(train_sentences)}")


# Generate sentence pairs for ContrastiveTensionLoss
def to_ct_pairs(sample, pos_neg_ratio=8):
    pos_neg_ratio = 1 / pos_neg_ratio
    sample["text2"] = sample["text1"] if random.random() < pos_neg_ratio else random.choice(train_sentences)
    return sample


train_dataset = train_dataset.map(to_ct_pairs, fn_kwargs={"pos_neg_ratio": pos_neg_ratio})
logging.info(train_dataset)

# As loss, we use ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_ratio=0.1,
    learning_rate=5e-5,
    save_strategy="steps",
    save_steps=0.5,
    logging_steps=0.1,
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

# Show similarity between first two sentences as an example
if len(train_sentences) >= 2:
    logging.info("\nExample similarity calculation:")
    sentence1 = train_sentences[0]
    sentence2 = train_sentences[1]
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    similarity = model.similarity(embedding1, embedding2).item()
    logging.info(f"  Sentence 1: {sentence1[:60]}...")
    logging.info(f"  Sentence 2: {sentence2[:60]}...")
    logging.info(f"  Cosine similarity: {similarity:.4f}")


# Save the trained & evaluated model locally
final_output_dir = f"{model_output_path}/final"
model.save_pretrained(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-ct-from-file")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-ct-from-file')`."
    )
