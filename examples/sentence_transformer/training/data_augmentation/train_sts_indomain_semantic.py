"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with Semantic Search Sampling.


Methodology:
Three steps are followed for AugSBERT data-augmentation strategy with Semantic Search -
    1. Fine-tune cross-encoder (BERT) on gold STSb dataset
    2. Fine-tuned Cross-encoder is used to label on Sem. Search sampled unlabeled pairs (silver STSb dataset)
    3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver STSb dataset

Citation: https://huggingface.co/papers/2010.08240

Usage:
python train_sts_indomain_semantic.py

OR
python train_sts_indomain_semantic.py pretrained_transformer_model_name top_k

python train_sts_indomain_semantic.py google-bert/bert-base-uncased 3
"""

import logging
import sys
from datetime import datetime

import torch
import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import cos_sim

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


# You can specify any huggingface/transformers pre-trained model here, for example, google-bert/bert-base-uncased, FacebookAI/roberta-base, FacebookAI/xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "google-bert/bert-base-uncased"
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

batch_size = 16
num_epochs = 1
max_seq_length = 128

# Read Datasets ######

train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")


cross_encoder_path = (
    "output/cross-encoder/stsb_indomain_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
bi_encoder_path = (
    "output/bi-encoder/stsb_augsbert_SS_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Cross-encoder (simpletransformers) ######
logging.info(f"Loading cross-encoder model: {model_name}")
# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for cross-encoder model
# Loading in fp32 is preferred for training if your memory can handle it
cross_encoder = CrossEncoder(model_name, num_labels=1, model_kwargs={"torch_dtype": "float32"})


# Bi-encoder (sentence-transformers) ######
logging.info(f"Loading bi-encoder model: {model_name}")
# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length, model_kwargs={"torch_dtype": "float32"})

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = Pooling(
    word_embedding_model.get_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

logging.info(f"Step 1: Train cross-encoder: {model_name} with STSbenchmark (gold dataset)")

# As we want to get symmetric scores, i.e. CrossEncoder(A,B) = CrossEncoder(B,A), we pass both combinations to the train set
gold_dataset = concatenate_datasets(
    [train_dataset, train_dataset.rename_columns({"sentence1": "sentence2", "sentence2": "sentence1"})]
)

# We add an evaluator, which evaluates the performance during training
evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=[[row["sentence1"], row["sentence2"]] for row in eval_dataset],
    scores=[row["score"] for row in eval_dataset],
    name="sts-dev",
)

# Train the cross-encoder model
ce_loss = BinaryCrossEntropyLoss(cross_encoder)
ce_args = CrossEncoderTrainingArguments(
    output_dir=cross_encoder_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=1000,
)
CrossEncoderTrainer(
    model=cross_encoder,
    args=ce_args,
    train_dataset=gold_dataset,
    loss=ce_loss,
    evaluator=evaluator,
).train()

############################################################################
#
# Step 2: Find silver pairs to label
#
############################################################################

# Top k similar sentences to be retrieved ####
# Larger the k, bigger the silver dataset ####

logging.info(
    f"Step 2.1: Generate STSbenchmark (silver dataset) using pretrained SBERT \
    model and top-{top_k} semantic search combinations"
)

silver_data = []
sentences = list(set(train_dataset["sentence1"]) | set(train_dataset["sentence2"]))
sent2idx = {sentence: idx for idx, sentence in enumerate(sentences)}  # storing id and sentence in dictionary
duplicates = set()  # not to include gold pairs of sentences again
for sentence1, sentence2 in zip(train_dataset["sentence1"], train_dataset["sentence2"]):
    duplicates.add((sent2idx[sentence1], sent2idx[sentence2]))
    duplicates.add((sent2idx[sentence2], sent2idx[sentence1]))


# For simplicity we use a pretrained model
semantic_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
semantic_search_model = SentenceTransformer(semantic_model_name)
logging.info(f"Encoding unique sentences with semantic search model: {semantic_model_name}")

# encoding all unique sentences present in the training dataset
embeddings = semantic_search_model.encode(sentences, batch_size=batch_size, convert_to_tensor=True)

logging.info(f"Retrieve top-{top_k} with semantic search model: {semantic_model_name}")

# retrieving top-k sentences given a sentence from the dataset
progress = tqdm.tqdm(unit="docs", total=len(sent2idx))
for idx in range(len(sentences)):
    sentence_embedding = embeddings[idx]
    cos_scores = cos_sim(sentence_embedding, embeddings)[0]
    cos_scores = cos_scores.cpu()
    progress.update(1)

    # We use torch.topk to find the highest 5 scores
    top_results = torch.topk(cos_scores, k=top_k + 1)

    for score, iid in zip(top_results[0], top_results[1]):
        if iid != idx and (iid, idx) not in duplicates:
            silver_data.append((sentences[idx], sentences[iid]))
            duplicates.add((idx, iid))

progress.reset()
progress.close()

logging.info(f"Length of silver_dataset generated: {len(silver_data)}")
logging.info(f"Step 2.2: Label STSbenchmark (silver dataset) with cross-encoder: {model_name}")
cross_encoder = CrossEncoder(cross_encoder_path)
silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

############################################################################################
#
# Step 3: Train bi-encoder model with both STSbenchmark and labeled AllNlI - Augmented SBERT
#
############################################################################################

logging.info(f"Step 3: Train bi-encoder: {model_name} with STSbenchmark (gold + silver dataset)")

# Combine the gold and silver pairs for bi-encoder training
logging.info("Read STSbenchmark gold and silver train dataset")
silver_samples = Dataset.from_dict(
    {
        "sentence1": [data[0] for data in silver_data],
        "sentence2": [data[1] for data in silver_data],
        "score": silver_scores,
    }
)
bi_encoder_train_dataset = concatenate_datasets([gold_dataset, silver_samples])

train_loss = CosineSimilarityLoss(model=bi_encoder)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    name="sts-dev",
)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=bi_encoder_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name="augmentation-indomain-semantic-sts",
)

# Train the bi-encoder model
SentenceTransformerTrainer(
    model=bi_encoder,
    args=args,
    train_dataset=bi_encoder_train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
).train()

#################################################################################
#
# Evaluate cross-encoder and Augmented SBERT performance on STS benchmark dataset
#
#################################################################################

# load the stored augmented-sbert model
bi_encoder = SentenceTransformer(bi_encoder_path)
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    name="sts-test",
)
test_evaluator(bi_encoder, output_path=bi_encoder_path)
