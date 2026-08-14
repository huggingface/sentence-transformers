"""
Fine-tune a CrossEncoder for duplicate-question detection with a LoRA adapter.

Usage:
python training_quora_duplicate_questions_lora.py
"""

import logging
import traceback

from datasets import load_dataset
from peft import LoraConfig, TaskType

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

model_name = "distilbert/distilroberta-base"
model = CrossEncoder(model_name, num_labels=1, model_kwargs={"torch_dtype": "float32"})

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

logging.info("Read quora-duplicates train dataset")
dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
eval_dataset = dataset.select(range(10_000))
test_dataset = dataset.select(range(10_000, 20_000))
train_dataset = dataset.select(range(20_000, len(dataset)))

loss = BinaryCrossEntropyLoss(model)
dev_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"])),
    labels=eval_dataset["label"],
    name="quora-duplicates-dev",
)
dev_evaluator(model)

short_model_name = model_name.split("/")[-1]
run_name = f"reranker-{short_model_name}-quora-duplicates-lora"
args = CrossEncoderTrainingArguments(
    output_dir=f"models/{run_name}",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    warmup_steps=0.1,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name=run_name,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

test_evaluator = CrossEncoderClassificationEvaluator(
    sentence_pairs=list(zip(test_dataset["sentence1"], test_dataset["sentence2"])),
    labels=test_dataset["label"],
    name="quora-duplicates-test",
)
test_evaluator(model)

final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, run "
        f"`huggingface-cli login`, load the model using `model = CrossEncoder({final_output_dir!r})`, and "
        f"save it using `model.push_to_hub('{run_name}')`."
    )
