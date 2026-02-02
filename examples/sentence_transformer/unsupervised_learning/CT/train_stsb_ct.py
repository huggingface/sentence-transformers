import logging
import random
import traceback
from datetime import datetime

from datasets import Dataset, load_dataset

from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# /print debug information to stdout

# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be divisible by pos_neg_ratio
num_epochs = 1
max_seq_length = 75

# Save path to store our model
output_dir = "output/train_stsb_ct-{}-{}".format(model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


# Train sentences
# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset = load_dataset("sentence-transformers/wiki1m-for-simcse", split="train")

# train_sentences are simply your list of sentences
train_sentences = [example["text"].strip() for example in wikipedia_dataset if len(example["text"].strip()) >= 10]
train_dataset = Dataset.from_dict({"text1": train_sentences})
logging.info(f"Train sentences: {len(train_sentences)}")


# Generate sentence pairs for ContrastiveTensionLoss
def to_ct_pairs(sample, pos_neg_ratio=8):
    pos_neg_ratio = 1 / pos_neg_ratio
    sample["text2"] = sample["text1"] if random.random() < pos_neg_ratio else random.choice(train_sentences)
    return sample


train_dataset = train_dataset.map(to_ct_pairs, fn_kwargs={"pos_neg_ratio": pos_neg_ratio})
logging.info(f"Generated {len(train_dataset)} training pairs")

# Download and load STSb
sts_dataset = load_dataset("sentence-transformers/stsb")

dev_samples = []
test_samples = []

for row in sts_dataset["validation"]:
    dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

for row in sts_dataset["test"]:
    test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="sts-dev")
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")

# Initialize an SBERT model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# As loss, we use ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLoss(model)

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0,
    learning_rate=1e-5,
    weight_decay=0,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=False,  # Set to True, if your GPU has optimized FP16 cores
    optim="rmsprop",
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

logging.info("Evaluating on test set")
model.evaluate(test_evaluator)


# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save_pretrained(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-stsb-ct")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-multi-task')`."
    )
