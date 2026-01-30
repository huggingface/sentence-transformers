import logging
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
# print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 128
epochs = 1
max_seq_length = 75

# Save path to store our model
model_save_path = "output/training_stsb_ct-improved-{}-{}".format(
    model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Train sentences
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")

train_dataset = wikipedia_dataset.map(lambda x: {"text": x["text"].strip()}, remove_columns=["text"])


# Download and load STSb
sts_dataset = load_dataset("sentence-transformers/stsb")
dev = sts_dataset["validation"]
test = sts_dataset["test"]


dev_evaluator = EmbeddingSimilarityEvaluator(
    dev["sentence1"],
    dev["sentence2"],
    dev["score"],
    name="sts-dev",
)

test_evaluator = EmbeddingSimilarityEvaluator(
    test["sentence1"],
    test["sentence2"],
    test["score"],
    name="sts-test",
)

# Initialize an SBERT model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Loss
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model, scale=1, similarity_fct=util.dot_score)

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0.1,
    eval_steps=0.1,
    logging_steps=0.01,
    learning_rate=5e-5,
    save_strategy="no",
    fp16=True,
)

# Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=train_loss,
)

# Train the model
trainer.train()
test_evaluator(model)

model.save_pretrained(model_save_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-simcse")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({model_save_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-simcse')`."
    )
