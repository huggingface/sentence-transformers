import gzip
import logging
import os
from datetime import datetime

from datasets import Dataset

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models, util
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Some training parameters. For the example, we use a batch_size of 128, a max sentence length (max_seq_length)
# of 32 word pieces and as model roberta-base
model_name = "roberta-base"
batch_size = 128
max_seq_length = 32
num_epochs = 1

# Download AskUbuntu and extract training corpus
askubuntu_folder = "data/askubuntu"
output_path = "output/askubuntu-simcse-{}-{}-{}".format(
    model_name, batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
for filename in ["text_tokenized.txt.gz", "dev.txt", "test.txt", "train_random.txt"]:
    filepath = os.path.join(askubuntu_folder, filename)
    if not os.path.exists(filepath):
        util.http_get("https://github.com/taolei87/askubuntu/raw/master/" + filename, filepath)

# Read the corpus
corpus = {}
dev_test_ids = set()
with gzip.open(os.path.join(askubuntu_folder, "text_tokenized.txt.gz"), "rt", encoding="utf8") as fIn:
    for line in fIn:
        splits = line.strip().split("\t")
        id = splits[0]
        title = splits[1]
        corpus[id] = title


# Read dev & test dataset
def read_eval_dataset(filepath):
    dataset = []
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:  # Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            dataset.append({
                "query": corpus[query_id],
                "positive": [corpus[pid] for pid in relevant_id],
                "negative": [corpus[pid] for pid in negative_ids],
            })
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    return dataset


dev_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "dev.txt"))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "test.txt"))


# Now we need a list of train sentences.
# In this example we simply use all sentences that don't appear in the train/dev set
train_sentences = []
for id, sentence in corpus.items():
    if id not in dev_test_ids:
        train_sentences.append({"sentence1": sentence, "sentence2": sentence})

logging.info(f"{len(train_sentences)} train sentences")
train_dataset = Dataset.from_list(train_sentences)

# Initialize an SBERT model


word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# As Loss function, we use MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(dev_dataset, name="AskUbuntu dev")
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")

logging.info("Dev performance before training")
dev_evaluator(model)

# Warmup: 10% of steps
steps_per_epoch = len(train_dataset) // batch_size
warmup_steps = int(num_epochs * steps_per_epoch * 0.1)

# Prepare training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0.1,
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

# Run test evaluation
test_evaluator(model)

latest_output_path = output_path + "-latest"
model.save_pretrained(latest_output_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-simcse")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({latest_output_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-simcse')`."
    )
