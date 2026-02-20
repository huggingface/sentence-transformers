"""
This script trains a sentence transformer using a combination of InfoNCE (via MultipleNegativesRankingLoss)
and Global Orthogonal Regularization (GOR) loss. The combination helps learn embeddings that are both
discriminative and well-distributed in the embedding space.

The script uses the GooAQ dataset (https://huggingface.co/datasets/sentence-transformers/gooaq), which contains
question-answer pairs from Google's "People Also Ask" feature. The model learns to encode questions and answers
such that matching pairs are close in embedding space.

Usage:
python training_gooaq_infonce_gor.py
"""

import logging
import random
import traceback
from collections.abc import Iterable

import torch
from datasets import Dataset, load_dataset
from torch import Tensor

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses.GlobalOrthogonalRegularizationLoss import GlobalOrthogonalRegularizationLoss
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


# Define a custom loss that combines InfoNCE and Global Orthogonal Regularization
class InfoNCEGORLoss(torch.nn.Module):
    """
    Combines MultipleNegativesRankingLoss (InfoNCE) with Global Orthogonal Regularization Loss.

    This loss encourages the model to:
    1. Learn discriminative embeddings where positive pairs are closer than negative pairs (InfoNCE)
    2. Distribute embeddings more evenly across the embedding space to avoid mode collapse (GOR)
    """

    def __init__(self, model: SentenceTransformer, similarity_fct=cos_sim, scale=20.0) -> None:
        super().__init__()
        self.model = model
        self.info_nce_loss = MultipleNegativesRankingLoss(model, similarity_fct=similarity_fct, scale=scale)
        self.gor_loss = GlobalOrthogonalRegularizationLoss(model, similarity_fct=similarity_fct)

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        info_nce_loss: dict[str, Tensor] = {
            "info_nce": self.info_nce_loss.compute_loss_from_embeddings(embeddings, labels)
        }
        gor_loss: dict[str, Tensor] = self.gor_loss.compute_loss_from_embeddings(embeddings, labels)
        return {**info_nce_loss, **gor_loss}


# Model and training parameters
model_name = "microsoft/mpnet-base"
num_train_samples = 100_000
num_eval_samples = 10_000
train_batch_size = 64
num_epochs = 1

# 1. Load a model to finetune with optional model card data
logging.info(f"Loading model: {model_name}")
model = SentenceTransformer(
    model_name,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on GooAQ using InfoNCE + Global Orthogonal Regularization",
    ),
)

# 2. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
logging.info("Loading GooAQ dataset")
dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(num_train_samples))
dataset = dataset.add_column("id", range(len(dataset)))
dataset_dict = dataset.train_test_split(test_size=num_eval_samples, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]
logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Eval dataset size: {len(eval_dataset)}")

# 3. Define the loss function
loss = InfoNCEGORLoss(model)

# 4. (Optional) Create an evaluator for use during training
# We create a small corpus for evaluation to measure retrieval performance
logging.info("Creating evaluation corpus")
random.seed(12)
queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
# Use only the answers that correspond to the evaluation queries for a focused evaluation
corpus = {qid: dataset[qid]["answer"] for qid in queries}
relevant_docs = {qid: {qid} for qid in eval_dataset["id"]}

dev_evaluator = InformationRetrievalEvaluator(
    corpus=corpus,
    queries=queries,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    name="gooaq-dev",
)

# Evaluate the base model before training
logging.info("Performance before fine-tuning:")
dev_evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"{short_model_name}-gooaq-infonce-gor"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Use NO_DUPLICATES to ensure each batch has unique samples, which benefits MultipleNegativesRankingLoss
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. Create a trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.remove_columns("id"),
    eval_dataset=eval_dataset.remove_columns("id"),
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the trained model on the test set
logging.info("Evaluating trained model")
dev_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
