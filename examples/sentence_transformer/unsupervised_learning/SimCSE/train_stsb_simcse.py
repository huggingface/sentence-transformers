import logging
import math
from datetime import datetime

from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# /print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
train_batch_size = 128
num_epochs = 1
max_seq_length = 32

# Save path to store our model
model_save_path = "output/training_stsb_simcse-{}-{}-{}".format(
    model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")


# train_samples is a list of InputExample objects where we pass the same sentence twice to texts, i.e. texts=[sent, sent]
train_samples = []
for example in wikipedia_dataset:
    line = example["text"].strip()
    if len(line) >= 10:
        train_samples.append(InputExample(texts=[line, line]))

# Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
sts_dataset = load_dataset("sentence-transformers/stsb")

dev_samples = []
test_samples = []

for row in sts_dataset["validation"]:
    dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

for row in sts_dataset["test"]:
    test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, batch_size=train_batch_size, name="sts-dev"
)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, batch_size=train_batch_size, name="sts-test"
)

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
logging.info(f"Training sentences: {len(train_samples)}")
logging.info(f"Warmup-steps: {warmup_steps}")
logging.info("Performance before training")
dev_evaluator(model)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=evaluation_steps,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    optimizer_params={"lr": 5e-5},
    use_amp=True,  # Set to True, if your GPU supports FP16 cores
)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
