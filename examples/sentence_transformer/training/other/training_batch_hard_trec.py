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
import os
import random
from collections import defaultdict
from datetime import datetime

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, util
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import InputExample

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


# Inspired from torchnlp
def trec_dataset(
    directory="datasets/trec/",
    train_filename="train_5500.label",
    test_filename="TREC_10.label",
    validation_dataset_nb=500,
    urls=[
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
        "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
    ],
):
    os.makedirs(directory, exist_ok=True)

    ret = []
    for url, filename in zip(urls, [train_filename, test_filename]):
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            util.http_get(url, full_path)

        examples = []
        label_map = {}
        guid = 1
        for line in open(full_path, "rb"):
            # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
            label, _, text = line.replace(b"\xf0", b" ").strip().decode().partition(" ")

            if label not in label_map:
                label_map[label] = len(label_map)

            label_id = label_map[label]
            guid += 1
            examples.append(InputExample(guid=guid, texts=[text], label=label_id))
        ret.append(examples)

    train_set, test_set = ret
    dev_set = None

    # Create a dev set from train set
    if validation_dataset_nb > 0:
        dev_set = train_set[-validation_dataset_nb:]
        train_set = train_set[:-validation_dataset_nb]

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42)  # Fix seed, so that we always get the same triplets
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2:  # We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = "all-distilroberta-v1"

### Create a torch.DataLoader that passes training batch instances to our model
train_batch_size = 32
output_path = "output/finetune-batch-hard-trec-" + model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
num_epochs = 1

logging.info("Loading TREC dataset")
train_set, dev_set, test_set = trec_dataset()

# We create a special dataset "SentenceLabelDataset" to wrap out train_set
# It will yield batches that contain at least two samples with the same label
train_data_sampler = SentenceLabelDataset(train_set)
train_dataloader = DataLoader(train_data_sampler, batch_size=32, drop_last=True)


# Load pretrained model
logging.info("Load model")
model = SentenceTransformer(model_name)


### Triplet losses ####################
### There are 4 triplet loss variants:
### - BatchHardTripletLoss
### - BatchHardSoftMarginTripletLoss
### - BatchSemiHardTripletLoss
### - BatchAllTripletLoss
#######################################

train_loss = losses.BatchAllTripletLoss(model=model)
# train_loss = losses.BatchHardTripletLoss(model=model)
# train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
# train_loss = losses.BatchSemiHardTripletLoss(model=model)


logging.info("Read TREC val dataset")
dev_evaluator = TripletEvaluator.from_input_examples(dev_set, name="trec-dev")

logging.info("Performance before fine-tuning:")
dev_evaluator(model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=warmup_steps,
    output_path=output_path,
)

##############################################################################
#
# Load the stored model and evaluate its performance on TREC dataset
#
##############################################################################

logging.info("Evaluating model on test set")
test_evaluator = TripletEvaluator.from_input_examples(test_set, name="trec-test")
model.evaluate(test_evaluator)
