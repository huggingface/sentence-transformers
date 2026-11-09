"""
Fine-tune a masked language model on a Hugging Face dataset.

Customize the model and dataset below for your domain.

Usage:
python train_mlm.py
"""

from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.utils import is_torch_bf16_gpu_available

model_name = "answerdotai/ModernBERT-base"
per_device_train_batch_size = 32

eval_save_ratio = 0.1  # Evaluate and save every 10% of total training steps
num_train_epochs = 3  # Number of epochs
max_steps = 2000  # A positive value overrides num_train_epochs
use_bf16 = is_torch_bf16_gpu_available()
mlm_prob = 0.3  # Probability of selecting a token for prediction

# Load the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


output_dir = "output/{}-{}".format(model_name.replace("/", "_"), datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print("Save checkpoints to:", output_dir)


##### Load our training datasets

dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")
dataset = dataset.filter(lambda example: len(example["text"].strip()) >= 10)


def tokenize(batch):
    return tokenizer([text.strip() for text in batch["text"]], truncation=True)


dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print("Train sentences:", len(train_dataset))
print("Eval sentences:", len(eval_dataset))


##### Training arguments

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,
    eval_strategy="steps",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=eval_save_ratio,
    save_steps=eval_save_ratio,
    logging_steps=eval_save_ratio,
    save_total_limit=1,
    prediction_loss_only=True,
    bf16=use_bf16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Save tokenizer to:", output_dir)
tokenizer.save_pretrained(output_dir)

print("Training on:", trainer.args.device)
trainer.train()

print("Save model to:", output_dir)
model.save_pretrained(output_dir)

print("Training done")
