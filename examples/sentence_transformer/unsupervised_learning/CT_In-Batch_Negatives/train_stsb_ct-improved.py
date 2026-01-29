import logging
from datetime import datetime

from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

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
wikipedia_dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")


# train_sentences are simply your list of sentences
train_sentences = [
    InputExample(texts=[example["text"].strip(), example["text"].strip()]) for example in wikipedia_dataset
]

################## Download and load STSb #################
sts_dataset = load_dataset("sentence-transformers/stsb")

dev_samples = []
test_samples = []

for row in sts_dataset["validation"]:
    dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

for row in sts_dataset["test"]:
    test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

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
