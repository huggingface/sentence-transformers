# See https://huggingface.co/collections/tomaarsen/training-with-prompts-672ce423c85b4d39aed52853 for some already trained models

import logging
import random

import numpy
import torch
from datasets import Dataset, load_dataset

from sentence_transformers import (
    BaseModelCardData,
    BaseTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)

# Feel free to adjust these variables:
use_prompts = True
include_prompts_in_pooling = True

# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "microsoft/mpnet-base",
    model_card_data=BaseModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on Natural Questions pairs",
    ),
)
model.set_pooling_include_prompt(include_prompts_in_pooling)

# 2. (Optional) Define prompts
if use_prompts:
    query_prompt = "query: "
    corpus_prompt = "document: "
    prompts = {
        "query": query_prompt,
        "answer": corpus_prompt,
    }

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)

# 5. (Optional) Specify training arguments
run_name = "mpnet-base-nq"
if use_prompts:
    run_name += "-prompts"
if not include_prompts_in_pooling:
    run_name += "-exclude-pooling-prompts"
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=5,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=12,
    prompts=prompts if use_prompts else None,
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = NanoBEIREvaluator(
    query_prompts=query_prompt if use_prompts else None,
    corpus_prompts=corpus_prompt if use_prompts else None,
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluator after training
dev_evaluator(model)

# 8. Save the trained model
model.save_pretrained(f"models/{run_name}/final")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub(run_name)
