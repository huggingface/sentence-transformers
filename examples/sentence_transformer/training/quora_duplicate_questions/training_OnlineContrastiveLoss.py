"""
This scripts demonstrates how to train a sentence embedding model for question pair classification
with cosine-similarity and a simple threshold.

As dataset, we use Quora Duplicates Questions, where we have labeled pairs of questions being either duplicates (label 1) or non-duplicate (label 0).

As loss function, we use OnlineConstrativeLoss. It reduces the distance between positive pairs, i.e., it pulls the embeddings of positive pairs closer together. For negative pairs, it pushes them further apart.

An issue with constrative loss is, that it might push sentences away that are already well positioned in vector space.
"""

import logging
import random
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    InformationRetrievalEvaluator,
    ParaphraseMiningEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import OnlineContrastiveLoss
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model_name = "stsb-distilbert-base"
model = SentenceTransformer(model_name)
num_train_epochs = 1
batch_size = 64

output_dir = "output/training_ocl-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

################### Load Quora Duplicate Questions dataset ##################
# https://huggingface.co/datasets/sentence-transformers/quora-duplicates
dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
dataset = dataset.train_test_split(test_size=1000)
train_dataset = dataset["train"].select(range(100000))
eval_dataset = dataset["test"]

# Negative pairs should have a distance of at least 0.5
margin = 0.5
# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
train_loss = OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (question1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.

binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    labels=eval_dataset["label"],
    name="quora-duplicates",
)
evaluators.append(binary_acc_evaluator)


###### Duplicate Questions Mining ######
# Given a large corpus of questions, identify all duplicates in that corpus.

# Load the Quora Duplicates Mining dataset
# https://huggingface.co/datasets/sentence-transformers/quora-duplicates-mining
questions_dataset = load_dataset("sentence-transformers/quora-duplicates-mining", "questions", split="dev")
duplicates_dataset = load_dataset("sentence-transformers/quora-duplicates-mining", "duplicates", split="dev")

# Create a mapping from qid to question & a list of duplicates (qid1, qid2)
qid_to_questions = dict(zip(questions_dataset["qid"], questions_dataset["question"]))
duplicates = list(zip(duplicates_dataset["qid1"], duplicates_dataset["qid2"]))

# The ParaphraseMiningEvaluator computes the cosine similarity between all sentences and
# extracts a list with the pairs that have the highest similarity. Given the duplicate
# information in dev_duplicates, it then computes and F1 score how well our duplicate mining worked
paraphrase_mining_evaluator = ParaphraseMiningEvaluator(qid_to_questions, duplicates, name="quora-duplicates-dev")

evaluators.append(paraphrase_mining_evaluator)


###### Duplicate Questions Information Retrieval ######
# Given a question and a large corpus of thousands questions, find the most relevant (i.e. duplicate) question
# in that corpus.

# https://huggingface.co/datasets/BeIR/quora
# https://huggingface.co/datasets/BeIR/quora-qrels
new_ir_corpus = load_dataset("BeIR/quora", "corpus", split="corpus")
new_ir_queries = load_dataset("BeIR/quora", "queries", split="queries")
new_ir_relevant_docs_data = load_dataset("BeIR/quora-qrels", split="validation")

# Shrink the corpus size heavily to only the relevant documents + 10,000 random documents
required_corpus_ids = list(map(str, new_ir_relevant_docs_data["corpus-id"]))
required_corpus_ids += random.sample(new_ir_corpus["_id"], k=10_000)
new_ir_corpus = new_ir_corpus.filter(lambda x: x["_id"] in required_corpus_ids)

# Convert the datasets to dictionaries
new_ir_corpus = dict(zip(new_ir_corpus["_id"], new_ir_corpus["text"]))  # Our corpus (qid => question)
new_ir_queries = dict(zip(new_ir_queries["_id"], new_ir_queries["text"]))  # Our queries (qid => question)
new_ir_relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_question_ids])
for qid, corpus_ids in zip(new_ir_relevant_docs_data["query-id"], new_ir_relevant_docs_data["corpus-id"]):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in new_ir_relevant_docs:
        new_ir_relevant_docs[qid] = set()
    new_ir_relevant_docs[qid].add(corpus_ids)

# Given queries, a corpus and a mapping with relevant documents, the InformationRetrievalEvaluator computes different IR
# metrics. For our use case MRR@k and Accuracy@k are relevant.
ir_evaluator = InformationRetrievalEvaluator(new_ir_queries, new_ir_corpus, new_ir_relevant_docs)
evaluators.append(ir_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
seq_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logging.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # OCL benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    logging_steps=100,
    run_name="online-contrastive-loss",  # Will be used in W&B if `wandb` is installed
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=seq_evaluator,
)
trainer.train()

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-ocl")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-ocl')`."
    )
