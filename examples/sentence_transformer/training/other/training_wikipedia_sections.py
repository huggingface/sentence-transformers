"""
This script trains sentence transformers with a triplet loss function.

As corpus, we use the wikipedia sections dataset that was describd by Dor et al., 2018, Learning Thematic Similarity Metric Using Triplet Networks.
"""

import logging
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
from sentence_transformers.sentence_transformer.losses import TripletLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "distilbert-base-uncased"
batch_size = 16
num_train_epochs = 1

output_dir = "output/training-wikipedia-sections-" + model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the Wikipedia-Sections dataset: https://huggingface.co/datasets/sentence-transformers/wikipedia-sections
train_dataset = load_dataset("sentence-transformers/wikipedia-sections", "triplet", split="train").select(
    range(10_000)
)
eval_dataset = load_dataset("sentence-transformers/wikipedia-sections", "triplet", split="validation").select(
    range(1000)
)
test_dataset = load_dataset("sentence-transformers/wikipedia-sections", "triplet", split="test").select(range(1000))
logging.info(train_dataset)

# 3. Define our training loss
# TripletLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) needs three text columns
train_loss = TripletLoss(model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset[:1000]["anchor"],
    positives=eval_dataset[:1000]["positive"],
    negatives=eval_dataset[:1000]["negative"],
    name="wikipedia-sections-dev",
)

# 5. Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="wikipedia-sections-triplet",  # Will be used in W&B if `wandb` is installed
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


# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="wikipedia-sections-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-wikipedia-sections-triplet")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-wikipedia-sections-triplet')`."
    )
