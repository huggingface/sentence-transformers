"""
This example uses a simple bag-of-words (BoW) approach. A sentence is mapped
to a sparse vector with e.g. 25,000 dimensions. Optionally, you can also use tf-idf.

To make the model trainable, we add multiple dense layers to create a Deep Averaging Network (DAN).
"""

import logging
import math
import os
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses, models, util
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.models.tokenizer.WordTokenizer import ENGLISH_STOP_WORDS
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

num_train_epochs = 1
batch_size = 32
output_dir = "output/training_tf-idf_word_embeddings-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

# 2. Define the model
# Wikipedia document frequency for words
wiki_doc_freq = "wikipedia_doc_frequencies.txt"
if not os.path.exists(wiki_doc_freq):
    util.http_get(
        "https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/wikipedia_doc_frequencies.txt", wiki_doc_freq
    )

# Create the vocab for the BoW model
stop_words = ENGLISH_STOP_WORDS
max_vocab_size = 25000  # This is also the size of the BoW sentence vector.


# Read the most common max_vocab_size words. Skip stop-words
vocab = set()
weights = {}
lines = open("wikipedia_doc_frequencies.txt", encoding="utf8").readlines()
num_docs = int(lines[0])
for line in lines[1:]:
    word, freq = line.lower().strip().split("\t")
    if word in stop_words:
        continue

    vocab.add(word)
    weights[word] = math.log(num_docs / int(freq))

    if len(vocab) >= max_vocab_size:
        break

# Create the BoW model. Because we set word_weights to the IDF values and cumulative_term_frequency=True, we
# get tf-idf vectors. Set word_weights to an empty dict and cumulative_term_frequency=False to get a 1-hot sentence encoding
bow = models.BoW(vocab=vocab, word_weights=weights, cumulative_term_frequency=True)

# Add two trainable feed-forward networks (DAN) with max_vocab_size -> 768 -> 512 dimensions.
sent_embeddings_dimension = max_vocab_size
dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=768)
dan2 = models.Dense(in_features=768, out_features=512)

model = SentenceTransformer(modules=[bow, dan1, dan2])

# 3. Define our training loss
# CosineSimilarityLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) needs two text columns and
# one similarity score column (between 0 and 1)
train_loss = losses.CosineSimilarityLoss(model=model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# 5. Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="wikipedia-tf-idf-bow",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = "wikipedia-tf-idf-bow"
try:
    model.push_to_hub(model_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}')`."
    )
