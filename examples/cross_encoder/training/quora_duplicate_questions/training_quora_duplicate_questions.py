"""
This examples trains a CrossEncoder for the Quora Duplicate Questions Detection task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_quora_duplicate_questions.py

"""

import logging
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

train_batch_size = 64
num_epochs = 1
output_dir = "output/training_ce_quora-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Define our CrossEncoder model. We use distilroberta-base as the base model and set it up to predict 1 label
# You can also use other base models, like bert-base-uncased, microsoft/mpnet-base, or rerankers like Alibaba-NLP/gte-reranker-modernbert-base
model_name = "distilroberta-base"
model = CrossEncoder(model_name, num_labels=1)

# 2. Load the Quora duplicates dataset: https://huggingface.co/datasets/sentence-transformers/quora-duplicates
logging.info("Read quora-duplicates train dataset")
dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
eval_dataset = dataset.select(range(10_000))
test_dataset = dataset.select(range(10_000, 20_000))
train_dataset = dataset.select(range(20_000, len(dataset)))
logging.info(train_dataset)
logging.info(eval_dataset)
logging.info(test_dataset)

# 3. Define our training loss, we use one that accepts pairs with a binary label
loss = BinaryCrossEntropyLoss(model)

# 4. Before and during training, we use CrossEncoderClassificationEvaluator to measure the performance on the dev set
dev_cls_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"])),
    labels=eval_dataset["label"],
    name="quora-duplicates-dev",
)
dev_cls_evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"reranker-{short_model_name}-quora-duplicates"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_steps=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_cls_evaluator,
)
trainer.train()

# 7. Evaluate the final model on test dataset
test_cls_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"])),
    labels=eval_dataset["label"],
    name="quora-duplicates-test",
)
test_cls_evaluator(model)

# 8. Save the final model
final_output_dir = f"{output_dir}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
