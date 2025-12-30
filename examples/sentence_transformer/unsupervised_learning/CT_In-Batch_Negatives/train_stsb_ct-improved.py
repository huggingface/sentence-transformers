import csv
import gzip
import logging
import os
from datetime import datetime

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

## Training parameters
model_name = "distilbert-base-uncased"
batch_size = 128
epochs = 1
max_seq_length = 75

# Save path to store our model
model_save_path = "output/training_stsb_ct-improved-{}-{}".format(
    model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


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
        train_sentences.append(InputExample(texts=[line.strip(), line.strip()]))

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
train_dataloader = DataLoader(train_sentences, batch_size=batch_size, shuffle=True, drop_last=True)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model, scale=1, similarity_fct=util.dot_score)


# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=1,
    evaluation_steps=1000,
    warmup_steps=1000,
    output_path=model_save_path,
    optimizer_params={"lr": 5e-5},
    use_amp=True,  # Set to True, if your GPU supports FP16 cores
)

########### Load the model and evaluate on test set

model = SentenceTransformer(model_save_path)
test_evaluator(model)
