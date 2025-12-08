"""
The script shows how to train Augmented SBERT (Domain-Transfer/Cross-Domain) strategy for STSb-QQP dataset.
For our example below we consider STSb (source) and QQP (target) datasets respectively.

Methodology:
Three steps are followed for AugSBERT data-augmentation strategy with Domain Transfer / Cross-Domain -
1. Cross-Encoder aka BERT is trained over STSb (source) dataset.
2. Cross-Encoder is used to label QQP training (target) dataset (Assume no labels/no annotations are provided).
3. Bi-encoder aka SBERT is trained over the labeled QQP (target) dataset.

Citation: https://huggingface.co/papers/2010.08240

Usage:
python train_sts_qqp_crossdomain.py

OR
python train_sts_qqp_crossdomain.py pretrained_transformer_model_name
"""

import csv
import gzip
import logging
import math
import os
import sys
from datetime import datetime
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
batch_size = 16
num_epochs = 1
max_seq_length = 128
use_cuda = torch.cuda.is_available()

###### Read Datasets ######
sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
qqp_dataset_path = "quora-IR-dataset"


# Check if the STSb dataset exists. If not, download and extract it
if not os.path.exists(sts_dataset_path):
    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


# Check if the QQP dataset exists. If not, download and extract
if not os.path.exists(qqp_dataset_path):
    logging.info("Dataset not found. Download")
    zip_save_path = "quora-IR-dataset.zip"
    util.http_get(url="https://sbert.net/datasets/quora-IR-dataset.zip", path=zip_save_path)
    with ZipFile(zip_save_path, "r") as zipIn:
        zipIn.extractall(qqp_dataset_path)


cross_encoder_path = (
    "output/cross-encoder/stsb_indomain_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
bi_encoder_path = (
    "output/bi-encoder/qqp_cross_domain_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

###### Cross-encoder (simpletransformers) ######

logging.info(f"Loading cross-encoder model: {model_name}")
# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
cross_encoder = CrossEncoder(model_name, num_labels=1)

###### Bi-encoder (sentence-transformers) ######

logging.info(f"Loading bi-encoder model: {model_name}")

# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

logging.info(f"Step 1: Train cross-encoder: {model_name} with STSbenchmark (source dataset)")

gold_samples = []
dev_samples = []
test_samples = []

with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1

        if row["split"] == "dev":
            dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
        elif row["split"] == "test":
            test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
        else:
            # As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
            gold_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
            gold_samples.append(InputExample(texts=[row["sentence2"], row["sentence1"]], label=score))


# We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(gold_samples, shuffle=True, batch_size=batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = CrossEncoderCorrelationEvaluator.from_input_examples(dev_samples, name="sts-dev")

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

# Train the cross-encoder model
cross_encoder.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=cross_encoder_path,
)

##################################################################
#
# Step 2: Label QQP train dataset using cross-encoder (BERT) model
#
##################################################################

logging.info(f"Step 2: Label QQP (target dataset) with cross-encoder: {model_name}")

cross_encoder = CrossEncoder(cross_encoder_path)

silver_data = []

with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if row["is_duplicate"] == "1":
            silver_data.append([row["question1"], row["question2"]])

silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

binary_silver_scores = [1 if score >= 0.5 else 0 for score in silver_scores]

###########################################################################
#
# Step 3: Train bi-encoder (SBERT) model with QQP dataset - Augmented SBERT
#
###########################################################################

logging.info(f"Step 3: Train bi-encoder: {model_name} over labeled QQP (target dataset)")

# Convert the dataset to a DataLoader ready for training
logging.info("Loading BERT labeled QQP dataset")
qqp_train_data = list(
    InputExample(texts=[data[0], data[1]], label=score) for (data, score) in zip(silver_data, binary_silver_scores)
)


train_dataloader = DataLoader(qqp_train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(bi_encoder)

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
logging.info("Read QQP dev dataset")

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

with open(os.path.join(qqp_dataset_path, "classification/dev_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row["question1"])
        dev_sentences2.append(row["question2"])
        dev_labels.append(int(row["is_duplicate"]))

evaluator = BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

# Train the bi-encoder model
bi_encoder.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=bi_encoder_path,
)

###############################################################
#
# Evaluate Augmented SBERT performance on QQP benchmark dataset
#
###############################################################

# Loading the augmented sbert model
bi_encoder = SentenceTransformer(bi_encoder_path)

logging.info("Read QQP test dataset")
test_sentences1 = []
test_sentences2 = []
test_labels = []

with open(os.path.join(qqp_dataset_path, "classification/test_pairs.tsv"), encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        test_sentences1.append(row["question1"])
        test_sentences2.append(row["question2"])
        test_labels.append(int(row["is_duplicate"]))

evaluator = BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels)
bi_encoder.evaluate(evaluator)
