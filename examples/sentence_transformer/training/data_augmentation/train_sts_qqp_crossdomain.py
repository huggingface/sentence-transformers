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

import logging
import sys
from datetime import datetime

from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# You can specify any huggingface/transformers pre-trained model here, for example, google-bert/bert-base-uncased, FacebookAI/roberta-base, FacebookAI/xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "google-bert/bert-base-uncased"
batch_size = 16
num_epochs = 1
max_seq_length = 128

# Read Datasets ######
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

# QQP (target dataset): https://huggingface.co/datasets/sentence-transformers/quora-duplicates
# The pair-class subset has (sentence1, sentence2, label), where label is 0 (different) or 1 (duplicate).
# It only has a "train" split, so we hold out small dev and test sets for evaluation.
qqp_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
qqp_dataset = qqp_dataset.train_test_split(test_size=10_000, seed=12)
qqp_train = qqp_dataset["train"]
qqp_dev = qqp_dataset["test"].select(range(5_000))
qqp_test = qqp_dataset["test"].select(range(5_000, 10_000))


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

# Cross-encoder (simpletransformers) ######

logging.info(f"Loading cross-encoder model: {model_name}")
# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
# Loading in fp32 is preferred for training if your memory can handle it
cross_encoder = CrossEncoder(model_name, num_labels=1, model_kwargs={"torch_dtype": "float32"})

# Bi-encoder (sentence-transformers) ######

logging.info(f"Loading bi-encoder model: {model_name}")

# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length, model_kwargs={"torch_dtype": "float32"})

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = Pooling(
    word_embedding_model.get_embedding_dimension(),
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

# As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
gold_dataset = concatenate_datasets(
    [train_dataset, train_dataset.rename_columns({"sentence1": "sentence2", "sentence2": "sentence1"})]
)

# We add an evaluator, which evaluates the performance during training
evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=[[row["sentence1"], row["sentence2"]] for row in eval_dataset],
    scores=[row["score"] for row in eval_dataset],
    name="sts-dev",
)

# Train the cross-encoder model
ce_loss = BinaryCrossEntropyLoss(cross_encoder)
ce_args = CrossEncoderTrainingArguments(
    output_dir=cross_encoder_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_ratio=0.1,
    eval_strategy="epoch",
)
ce_trainer = CrossEncoderTrainer(
    model=cross_encoder,
    args=ce_args,
    train_dataset=gold_dataset,
    loss=ce_loss,
    evaluator=evaluator,
)
ce_trainer.train()
cross_encoder.save_pretrained(f"{cross_encoder_path}/final")

##################################################################
#
# Step 2: Label QQP train dataset using cross-encoder (BERT) model
#
##################################################################

logging.info(f"Step 2: Label QQP (target dataset) with cross-encoder: {model_name}")

# Following AugSBERT, we treat the QQP pairs as unlabeled: we ignore the gold labels and let the
# STSb-trained cross-encoder assign each pair a soft similarity score in [0, 1]. These silver scores
# are what the bi-encoder trains on, transferring the cross-encoder's knowledge to the target domain.
silver_data = list(zip(qqp_train["sentence1"], qqp_train["sentence2"]))
silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0, 1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

###########################################################################
#
# Step 3: Train bi-encoder (SBERT) model with QQP dataset - Augmented SBERT
#
###########################################################################

logging.info(f"Step 3: Train bi-encoder: {model_name} over labeled QQP (target dataset)")

logging.info("Loading the cross-encoder labeled QQP dataset")
qqp_train_dataset = Dataset.from_dict(
    {
        "sentence1": [pair[0] for pair in silver_data],
        "sentence2": [pair[1] for pair in silver_data],
        "score": [float(score) for score in silver_scores],
    }
)

train_loss = CosineSimilarityLoss(model=bi_encoder)

# Classification ######
# Given (question1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
logging.info("Read QQP dev dataset")
evaluator = BinaryClassificationEvaluator(qqp_dev["sentence1"], qqp_dev["sentence2"], qqp_dev["label"])

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=bi_encoder_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.05,
    run_name="augmentation-qqp-crossdomain",
)

# Train the bi-encoder model
trainer = SentenceTransformerTrainer(
    model=bi_encoder,
    args=args,
    train_dataset=qqp_train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()
bi_encoder.save_pretrained(f"{bi_encoder_path}/final")

###############################################################
#
# Evaluate Augmented SBERT performance on QQP benchmark dataset
#
###############################################################

logging.info("Read QQP test dataset")
evaluator = BinaryClassificationEvaluator(qqp_test["sentence1"], qqp_test["sentence2"], qqp_test["label"])
bi_encoder.evaluate(evaluator)
