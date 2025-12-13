import logging
import random
from datetime import datetime

import mlflow

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


model_name = "clip-ViT-B-32"
dataset_name = "flickr8k-binary"
train_batch_size = 4
num_epochs = 4
output_dir = f"data/output/models/{dataset_name}"


def convert_binary_flickr8k(dataset_split: dict) -> list[dict]:
    """
    Converts a dataset of (image, 5 captions) into a binary classification format:
    (image, matching_caption, label=1.0)
    (image, non_matching_caption, label=0.0)

    returns a dict {"image", "text", "label"}
    """
    output = []

    for idx in range(len(dataset_split["image"])):
       
        current_image = dataset_split["image"][idx]
        positive_caption = dataset_split["caption_0"][idx]
        
        # 1. Positive Example (Image matches its caption)
        output.append({
            "image": current_image,
            "text": positive_caption,
            "label": 1.0
        })
        
        # 2. Negative Example

        while True:
            random_caption = random.choice(dataset_split["caption_0"])

            if random_caption != positive_caption:
                break

        output.append({
            "image": current_image,
            "text": random_caption,
            "label": 0.0
        })
    
    return output

# 1. Load CLIP model
model = SentenceTransformer(model_name)

# 2. Prepare the dataset
ds = load_dataset("jxie/flickr8k")
train_split = ds["train"][:200]
eval_split = ds["validation"][:20]

# Create training data
train_data = convert_binary_flickr8k(train_split)
eval_data = convert_binary_flickr8k(eval_split)

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

logging.info(f"Training samples: {len(train_dataset)}")
logging.info(f"Evaluation samples: {len(eval_dataset)}")
logging.info(train_dataset)

# 3. Define our training loss
train_loss = losses.ContrastiveLoss(model=model)


# 4. Define an evaluator for use during training
evaluator = BinaryClassificationEvaluator(
    sentences1=[item["image"] for item in eval_data],
    sentences2=[item["text"] for item in eval_data],
    labels=[item["label"] for item in eval_data],
    name="clip-dev",
)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True, 
    bf16=False,
    # Optional tracking/debugging parameters:
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    logging_steps=50,
    run_name="clip-training",
    report_to=["mlflow"]
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

# 7. Begin Training and Log results to MLFlow
mlflow.set_experiment("CLIP")
with mlflow.start_run(run_name=dataset_name):

    mlflow.log_params({
        "model_name": model_name,
        "train_batch_size": train_batch_size,
        "num_epochs": num_epochs,
        "loss_function": train_loss.__class__.__name__,
    })
    
    # Start training
    trainer.train()

    # 8. Evaluate the model performance on the evaluation dataset
    metrics = evaluator(model)
    mlflow.log_metrics(metrics)

# 9. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
