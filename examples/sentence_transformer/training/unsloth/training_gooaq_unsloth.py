import logging
import sys
import traceback

from datasets import Dataset, load_dataset
from unsloth import FastSentenceTransformer

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
model_name_only = model_name.split("/")[-1]

# 1. Load a model to finetune using FastSentenceTransformer
model = FastSentenceTransformer.from_pretrained(
    model_name,
    max_seq_length=1024,
    full_finetuning=False,
)

# 2. Create a LoRA adapter for the model using unsloth's optimized implementation
model = FastSentenceTransformer.get_peft_model(
    model,
    r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                   "gate_proj", "up_proj", "down_proj",],
    lora_alpha=128,
    lora_dropout=0.1,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    task_type="FEATURE_EXTRACTION",
)
model.bfloat16()
# From this point onwards, use the `model` as a regular SentenceTransformer model

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/gooaq", split="train")
dataset_dict = dataset.train_test_split(test_size=10_000, seed=12)
train_dataset: Dataset = dataset_dict["train"].select(range(1_000_000))
eval_dataset: Dataset = dataset_dict["test"]

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)

# 5. (Optional) Specify training arguments
run_name = f"{model_name_only}-gooaq-unsloth"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=1024,
    per_device_eval_batch_size=1024,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.01,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = NanoBEIREvaluator()
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
