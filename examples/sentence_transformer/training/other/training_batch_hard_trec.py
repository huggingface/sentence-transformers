"""
This script trains sentence transformers with a batch hard loss function.

The TREC dataset will be automatically downloaded and put in the datasets/ directory

Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
to select good triplets. If the negative sentence is selected randomly, the training objective is often
too easy and the network fails to learn good representations.

Batch hard triplet loss (https://huggingface.co/papers/1703.07737) creates triplets on the fly. It requires that the
data is labeled (e.g. labels 1, 2, 3) and we assume that samples with the same label are similar:

In a batch, it checks for sent1 with label 1 what is the other sentence with label 1 that is the furthest (hard positive)
which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
all sentences with the same label should be close and sentences for different labels should be clearly separated.
"""

import logging
import random
import traceback
from collections import defaultdict
from datetime import datetime

from datasets import Dataset, load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


def trec_dataset(
    validation_dataset_nb=500,
):
    dataset = load_dataset("omkar334/trec")

    train_set = dataset["train"].remove_columns(["coarse_label"]).rename_column("fine_label", "label")
    test_set = dataset["test"].remove_columns(["coarse_label"]).rename_column("fine_label", "label")

    # Create a dev set from train set
    if validation_dataset_nb > 0:
        dev_set = train_set.select(range(len(train_set) - validation_dataset_nb, len(train_set)))
        train_set = train_set.select(range(len(train_set) - validation_dataset_nb))

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42)  # Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets


def triplets_from_labeled_dataset(dataset):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label

    # Pre-compute label groupings
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(dataset["label"]):
        label_to_indices[label].append(idx)

    # Filter out labels with < 2 samples
    valid_labels = {k: v for k, v in label_to_indices.items() if len(v) >= 2}
    other_labels_map = {k: [label for label in valid_labels if label != k] for k in valid_labels}

    triplets = []
    for idx, (text, label) in enumerate(zip(dataset["text"], dataset["label"])):
        if label not in valid_labels:
            continue

        pos_idx = random.choice([i for i in valid_labels[label] if i != idx])
        neg_label = random.choice(other_labels_map[label])
        neg_idx = random.choice(valid_labels[neg_label])

        triplets.append({"anchor": text, "positive": dataset[pos_idx]["text"], "negative": dataset[neg_idx]["text"]})

    return Dataset.from_list(triplets)


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "all-distilroberta-v1"

# Create training dataset
batch_size = 32
output_path = "output/finetune-batch-hard-trec-" + model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 1

logging.info("Loading TREC dataset")
train_set, dev_set, test_set = trec_dataset()

# Load pretrained model
logging.info("Load model")
model = SentenceTransformer(model_name)


# Triplet losses
# There are 4 triplet loss variants:
# - BatchHardTripletLoss
# - BatchHardSoftMarginTripletLoss
# - BatchSemiHardTripletLoss
# - BatchAllTripletLoss

train_loss = losses.BatchAllTripletLoss(model=model)
# train_loss = losses.BatchHardTripletLoss(model=model)
# train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
# train_loss = losses.BatchSemiHardTripletLoss(model=model)


logging.info("Read TREC val dataset")
dev_evaluator = TripletEvaluator(
    anchors=dev_set["anchor"],
    positives=dev_set["positive"],
    negatives=dev_set["negative"],
    name="trec_dev",
)

logging.info("Performance before fine-tuning:")
model.evaluate(dev_evaluator)


# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    # GROUP_BY_LABEL ensures each batch has at least 2 distinct labels with at least 2 samples per label
    batch_sampler=BatchSamplers.GROUP_BY_LABEL,
    eval_strategy="steps",
    eval_steps=0.2,
    save_strategy="steps",
    save_steps=0.2,
    logging_steps=0.1,
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Load the stored model and evaluate its performance on TREC dataset

logging.info("Evaluating model on test set")
test_evaluator = TripletEvaluator(
    anchors=test_set["anchor"],
    positives=test_set["positive"],
    negatives=test_set["negative"],
    name="trec-test",
)
model.evaluate(test_evaluator)

# Save the trained & evaluated model locally
final_output_dir = f"{output_path}/final"
model.save_pretrained(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-trec")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-trec')`."
    )
