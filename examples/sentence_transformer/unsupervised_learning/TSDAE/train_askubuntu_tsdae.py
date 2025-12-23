import gzip
import logging
import os
import random
import traceback
from datetime import datetime

from datasets import Dataset

from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import RerankingEvaluator
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Training parameters
model_name = "bert-base-uncased"
train_batch_size = 8
num_epochs = 1
max_seq_length = 75

output_dir = f"output/training_stsb_tsdae-{model_name.replace('/', '-')}-{train_batch_size}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Defining our sentence transformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# or to load a pre-trained SentenceTransformer model OR use mean pooling
# model = SentenceTransformer(model_name)
# model.max_seq_length = max_seq_length


# 2. Download the AskUbuntu dataset from https://github.com/taolei87/askubuntu
askubuntu_folder = "data/askubuntu"
result_folder = "output/askubuntu-tsdae-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 8

for filename in ["text_tokenized.txt.gz", "dev.txt", "test.txt", "train_random.txt"]:
    filepath = os.path.join(askubuntu_folder, filename)
    if not os.path.exists(filepath):
        util.http_get("https://github.com/taolei87/askubuntu/raw/master/" + filename, filepath)

# Read the corpus
corpus = {}
dev_test_ids = set()
with gzip.open(os.path.join(askubuntu_folder, "text_tokenized.txt.gz"), "rt", encoding="utf8") as fIn:
    for line in fIn:
        id, title, *_ = line.strip().split("\t")
        corpus[id] = title


# Read dev & test dataset
def read_eval_dataset(filepath) -> Dataset:
    data = {
        "query": [],
        "positive": [],
        "negative": [],
    }
    with open(filepath) as fIn:
        for line in fIn:
            query_id, relevant_id, candidate_ids, bm25_scores = line.strip().split("\t")
            if len(relevant_id) == 0:  # Skip examples without relevant entries
                continue

            relevant_id = relevant_id.split(" ")
            candidate_ids = candidate_ids.split(" ")
            negative_ids = set(candidate_ids) - set(relevant_id)
            data["query"].append(corpus[query_id])
            data["positive"].append([corpus[pid] for pid in relevant_id])
            data["negative"].append([corpus[pid] for pid in negative_ids])
            dev_test_ids.add(query_id)
            dev_test_ids.update(candidate_ids)
    dataset = Dataset.from_dict(data)
    return dataset


eval_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "dev.txt"))
test_dataset = read_eval_dataset(os.path.join(askubuntu_folder, "test.txt"))

## Now we need a list of train sentences.
## In this example we simply use all sentences that don't appear in the train/dev set
train_sentences = [sentence for id, sentence in corpus.items() if id not in dev_test_ids]
train_dataset = Dataset.from_dict({"text": train_sentences})


def noise_transform(batch, del_ratio=0.6):
    """
    Applies noise by randomly deleting words.

    WARNING: nltk's tokenization/detokenization is designed primarily for English.
    For other languages, especially those without clear word boundaries (e.g., Chinese),
    custom tokenization and detokenization are strongly recommended.

    Args:
        batch (Dict[str, List[str]]): A dictionary with the structure
            {column_name: [string1, string2, ...]}, where each list contains
            the batch data for the respective column.
        del_ratio (float): The ratio of words to delete. Defaults to 0.6.
    """
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    assert 0.0 <= del_ratio < 1.0, "del_ratio must be in the range [0, 1)"
    assert isinstance(batch, dict) and "text" in batch, "batch must be a dictionary with a 'text' key."

    texts = batch["text"]
    noisy_texts = []
    for text in texts:
        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            noisy_texts.append(text)
            continue

        kept_words = [word for word in words if random.random() < del_ratio]
        # Guarantee that at least one word remains
        if len(kept_words) == 0:
            noisy_texts.append(random.choice(words))
            continue

        noisy_texts.append(TreebankWordDetokenizer().detokenize(kept_words))
    return {"noisy": noisy_texts, "text": texts}


# TSDAE requires a dataset with 2 columns: a noisified text column and a text column
# We use a function to delete some words, but you can customize `noise_transform` to noisify your text some other way.
# We use `set_transform` instead of `map` so the noisified text differs each epoch.
train_dataset.set_transform(transform=lambda batch: noise_transform(batch), columns=["text"], output_all_columns=True)
print(train_dataset)
print(train_dataset[0])
"""
Dataset({
    features: ['text'],
    num_rows: 160436
})
{
    'noisy': 'how to get "battery is broken go?',
    'text': "how to get the `` your battery is broken '' message to go away ?",
}
"""
# As you can see, the noisy text is applied on the fly when the sample is accessed.
print(eval_dataset)
print(test_dataset)
"""
Dataset({
    features: ['query', 'positive', 'negative'],
    num_rows: 189
})
Dataset({
    features: ['query', 'positive', 'negative'],
    num_rows: 186
})
"""

# 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderLoss
# Note that this will likely result in warnings as we're loading 'model_name' as a decoder, but it likely won't
# have weights for that yet. This is fine, as we'll be training it from scratch.
train_loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
logging.info("Evaluation before training:")
dev_evaluator = RerankingEvaluator(eval_dataset, name="AskUbuntu-dev")
dev_evaluator(model)

# 5. Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    learning_rate=3e-5,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=1000,
    run_name="tsdae-askubuntu",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the test set after training
logging.info("Evaluation after training:")
test_evaluator = RerankingEvaluator(test_dataset, name="AskUbuntu-test")
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-tsdae-askubuntu")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-tsdae-askubuntu')`."
    )
