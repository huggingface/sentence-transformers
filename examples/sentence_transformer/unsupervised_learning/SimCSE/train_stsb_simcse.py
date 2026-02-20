import logging
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

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
word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# We use 1 Million sentences from Wikipedia to train our model
train_dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")


# SimCSE (or InfoNCE) require positive pairs (or larger), so if we only have single sentences,
# we can simply duplicate the sentences to create a weak dataset of positive pairs. The loss
# will take negative samples from the other samples in the batch.
def simcse_map(example):
    text = example["text"].strip()
    return {
        "sentence1": text,
        "sentence2": text,
    }


train_dataset = (
    load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")
    .filter(lambda x: len(x["text"].strip()) >= 10)
    .map(simcse_map, remove_columns=["text"])
)
logging.info(train_dataset)

# Download and load STSb
eval_sts_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_sts_dataset = load_dataset("sentence-transformers/stsb", split="test")

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_sts_dataset["sentence1"],
    sentences2=eval_sts_dataset["sentence2"],
    scores=eval_sts_dataset["score"],
    name="sts-dev",
)
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_sts_dataset["sentence1"],
    sentences2=test_sts_dataset["sentence2"],
    scores=test_sts_dataset["score"],
    name="sts-test",
)

# We train our model using the MultipleNegativesRankingLoss
train_loss = MultipleNegativesRankingLoss(model)

logging.info("Performance before training")
dev_evaluator(model)
test_evaluator(model)

# Prepare training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=0.1,
    logging_steps=0.01,
    learning_rate=5e-5,
    save_strategy="no",
    fp16=True,  # If your GPU does not have FP16 cores, set fp16=False
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=train_loss,
)

logging.info("Start training")
trainer.train()

# Run evaluation
logging.info("Performance after training")
dev_evaluator(model)
test_evaluator(model)

latest_output_path = model_save_path + "-latest"
model.save_pretrained(latest_output_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-stsb-simcse")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({latest_output_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-stsb-simcse')`."
    )
