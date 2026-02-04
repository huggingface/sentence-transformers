"""
This example trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for Natural Language Inference (NLI).
It uses QuantizationAwareLoss with MultipleNegativesRankingLoss to train models that perform well with quantization.

This script trains a model on the AllNLI dataset, which is a concatenation of SNLI and MultiNLI.
The model learns to distinguish between entailment, neutral, and contradiction sentences.

Usage:
python qat_nli.py

OR
python qat_nli.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
batch_size = 128
num_train_epochs = 1
quantization_precisions = ["float32", "int8", "binary"]

# Save path of the model
output_dir = f"output/qat_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
logging.info(model)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
# We'll use a subset for faster training in this example
train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)

# 3. Define our training loss
# MultipleNegativesRankingLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)
# is a great loss for learning embeddings. It requires anchor and positive pairs.
inner_train_loss = losses.MultipleNegativesRankingLoss(model=model)
train_loss = losses.QuantizationAwareLoss(
    model, loss=inner_train_loss, quantization_precisions=quantization_precisions
)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["negative"],
    name="all-nli-dev",
)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name="qat-nli",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 8. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-nli-qat")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-qat')`."
    )
