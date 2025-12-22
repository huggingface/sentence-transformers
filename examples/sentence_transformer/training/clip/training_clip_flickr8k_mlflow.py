import logging
from datetime import datetime

from datasets import DatasetDict, load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.models import CLIPModel
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Any clip model (https://huggingface.co/models?other=clip)
base_model_name = "openai/clip-vit-base-patch32"
train_model_name = "clip-flickr8k"
train_batch_size = 16  # Higher batch sizes give better results for the MultipleNegativesRankingLoss
num_epochs = 1
output_dir = f"models/{train_model_name}"


def to_binary_flickr8k(batch: dict) -> dict:
    """Converts the Flickr8k dataset format to a binary classification format"""
    output = {"image": [], "positive": [], "negative": []}
    for idx in range(len(batch["image"])):
        image = batch["image"][idx]
        caption = batch["caption_0"][idx]
        # Take the next caption as negative example (cyclic)
        negative_caption = batch["caption_0"][(idx + 1) % len(batch["caption_0"])]
        output["image"].append(image)
        output["positive"].append(caption)
        output["negative"].append(negative_caption)
    return output


# 1. Load CLIP model
clip_model = CLIPModel(base_model_name)
model = SentenceTransformer(modules=[clip_model])

# 2. Prepare the dataset
dataset: DatasetDict = load_dataset("jxie/flickr8k")
dataset = dataset.map(to_binary_flickr8k, batched=True, remove_columns=dataset["train"].column_names)
train_dataset = dataset["train"].select(range(1000))
eval_dataset = dataset["validation"].select(range(400))

logging.info(f"Training samples: {len(train_dataset)}")
logging.info(f"Evaluation samples: {len(eval_dataset)}")
logging.info(train_dataset)

# 3. Define our training loss
train_loss = losses.MultipleNegativesRankingLoss(model=model)


# 4. Define an evaluator for use during training
evaluator = BinaryClassificationEvaluator(
    sentences1=list(eval_dataset["image"]) + list(eval_dataset["image"]),
    sentences2=list(eval_dataset["positive"]) + list(eval_dataset["negative"]),
    labels=[1] * len(eval_dataset) + [0] * len(eval_dataset),
    name="flickr8k-dev",
)
evaluator(model)  # Evaluate the untrained model

# 5. Define the training arguments
# If 'mlflow' is installed, it will be automatically used to log training parameters & evaluation.
# We can make that explicit by setting report_to=["mlflow"]
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    learning_rate=2e-5,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.5,
    save_total_limit=2,
    logging_steps=0.02,
    run_name=train_model_name,
    report_to=["mlflow"],
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

# 7. Begin Training and log results to MLFlow
trainer.train()

# 8. Evaluate the final model on the evaluation set
evaluator(model)

# 9. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save_pretrained(final_output_dir)
