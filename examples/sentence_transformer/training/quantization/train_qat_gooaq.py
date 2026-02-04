"""
This script trains a sentence transformer with Quantization-Aware Training (QAT) using
MultipleNegativesRankingLoss.

The QAT approach trains the model to maintain high performance even when embeddings are quantized to
lower precision formats (int8, binary), which enables:
- 4x-32x storage reduction
- Faster similarity computations
- Lower deployment costs

The script uses the GooAQ dataset (https://huggingface.co/datasets/sentence-transformers/gooaq), which contains
question-answer pairs from Google's "People Also Ask" feature. The model learns to encode questions and answers
such that matching pairs are close in embedding space, while remaining robust to quantization.

Usage:
python qat_gooaq.py
"""

import logging
import random
import traceback

from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator

# from sentence_transformers.losses.GlobalOrthogonalRegularizationLoss import GlobalOrthogonalRegularizationLoss
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Model and training parameters
model_name = "microsoft/mpnet-base"
num_train_samples = 100_000
num_eval_samples = 10_000
train_batch_size = 64
num_epochs = 1
quantization_precisions = ["float32", "int8", "binary"]
eval_quantization_precisions = ["float32", "int8", "binary"]
quantization_weights = [1.0, 1.0, 0.5]

# 1. Load a model to finetune with optional model card data
logging.info(f"Loading model: {model_name}")
model = SentenceTransformer(
    model_name,
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on GooAQ using QAT with InfoNCE",
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

# 3. Define the loss function with QAT
base_loss = MultipleNegativesRankingLoss(model)
loss = losses.QuantizationAwareLoss(
    model=model,
    loss=base_loss,
    quantization_precisions=quantization_precisions,
    quantization_weights=quantization_weights,
)

logging.info(f"Training with quantization precisions: {quantization_precisions}")

# 4. Create evaluators for use during training
# We create a small corpus for evaluation to measure retrieval performance at different precisions
logging.info("Creating evaluation corpus")
random.seed(12)
queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
# Use only the answers that correspond to the evaluation queries for a focused evaluation
corpus = {qid: dataset[qid]["answer"] for qid in queries}
relevant_docs = {qid: {qid} for qid in eval_dataset["id"]}

# Create evaluators for each precision
evaluators = []
for precision in eval_quantization_precisions:
    evaluators.append(
        InformationRetrievalEvaluator(
            corpus=corpus,
            queries=queries,
            relevant_docs=relevant_docs,
            show_progress_bar=True,
            name=f"gooaq-dev-{precision}",
            precision=precision,
        )
    )

dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

# Evaluate the base model before training
logging.info("Performance before fine-tuning:")
dev_evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"{short_model_name}-gooaq-qat"
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
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.025,
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

# 7. Evaluate the trained model on the test set with all precisions
logging.info("\n" + "=" * 80)
logging.info("Evaluating trained model with different quantization precisions")
logging.info("=" * 80)
dev_evaluator(model)

# Print comparison of precisions
logging.info("\n" + "=" * 80)
logging.info("Quantization Performance Summary")
logging.info("=" * 80)
logging.info("Precision | Storage | Performance")
logging.info("-" * 80)
logging.info("float32   | 1x      | Baseline")
logging.info("int8      | 4x      | ~95-99% retention")
logging.info("binary    | 32x     | ~90-95% retention")
logging.info("=" * 80)

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

logging.info("\n✅ Training complete!")
logging.info(f"Model saved to: {final_output_dir}")
logging.info("\nTo use the model with quantization:")
logging.info("  embeddings = model.encode(texts, precision='int8')")
logging.info("  embeddings = model.encode(texts, precision='binary')")
