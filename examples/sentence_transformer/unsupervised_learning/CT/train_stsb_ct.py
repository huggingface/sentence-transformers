import csv
import gzip
import logging
import os
from datetime import datetime

import torch

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

## Training parameters
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be divisible by pos_neg_ratio
epochs = 1
max_seq_length = 75

# Save path to store our model
model_save_path = "output/train_stsb_ct-{}-{}".format(model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


################# Train sentences #################
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = "data/wiki1m_for_simcse.txt"
if not os.path.exists(wikipedia_dataset_path):
    util.http_get(
        "https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt",
        wikipedia_dataset_path,
    )

# train_sentences are simply your list of sentences
train_sentences = []
with open(wikipedia_dataset_path, encoding="utf8") as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

################# Download and load STSb #################
data_folder = "data/stsbenchmark"
sts_dataset_path = f"{data_folder}/stsbenchmark.tsv.gz"

if not os.path.exists(sts_dataset_path):
    util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)


dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
    reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

        if row["split"] == "dev":
            dev_samples.append(inp_example)
        elif row["split"] == "test":
            test_samples.append(inp_example)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

################# Initialize an SBERT model #################
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# For ContrastiveTension we need a special data loader to construct batches with the desired properties
train_dataloader = losses.ContrastiveTensionDataLoader(
    train_sentences, batch_size=batch_size, pos_neg_ratio=pos_neg_ratio
)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=1,
    evaluation_steps=1000,
    weight_decay=0,
    warmup_steps=0,
    optimizer_class=torch.optim.RMSprop,
    optimizer_params={"lr": 1e-5},
    output_path=model_save_path,
    use_amp=False,  # Set to True, if your GPU has optimized FP16 cores
)

########### Load the model and evaluate on test set

model = SentenceTransformer(model_save_path)
test_evaluator(model)
