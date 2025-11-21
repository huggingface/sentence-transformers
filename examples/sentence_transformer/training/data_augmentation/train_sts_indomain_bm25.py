"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with BM25 sampling.
We utilise easy and practical elasticsearch (https://www.elastic.co/) for BM25 sampling.

Installations:
For this example, elasticsearch to be installed (pip install elasticsearch)
[NOTE] You need to also install Elasticsearch locally on your PC or desktop.
link for download - https://www.elastic.co/downloads/elasticsearch
Or to run it with Docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

Methodology:
Three steps are followed for AugSBERT data-augmentation with BM25 Sampling -
    1. Fine-tune cross-encoder (BERT) on gold STSb dataset
    2. Fine-tuned Cross-encoder is used to label on BM25 sampled unlabeled pairs (silver STSb dataset)
    3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver STSb dataset

Citation: https://huggingface.co/papers/2010.08240

Usage:
python train_sts_indomain_bm25.py

OR
python train_sts_indomain_bm25.py pretrained_transformer_model_name top_k

python train_sts_indomain_bm25.py bert-base-uncased 3

"""

import logging
import math
import sys
import traceback
from datetime import datetime

import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset
from elasticsearch import Elasticsearch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# suppressing INFO messages for elastic-search logger
tracer = logging.getLogger("elasticsearch")
tracer.setLevel(logging.CRITICAL)
es = Elasticsearch()

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

batch_size = 16
num_epochs = 1
max_seq_length = 128

cross_encoder_path = (
    "output/cross-encoder/stsb_indomain_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
sentence_transformer_path = (
    "output/bi-encoder/stsb_augsbert_BM25_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Use a Hugging Face model (like BERT, RoBERTa, XLNet, XLM-R) for loading the CrossEncoder and SentenceTransformer
cross_encoder = CrossEncoder(model_name, num_labels=1)
sentence_transformer = SentenceTransformer(model_name)
sentence_transformer.max_seq_length = max_seq_length


#####################################################################
#
# Step 1: Train cross-encoder model with (gold) STS benchmark dataset
#
#####################################################################

logging.info(f"Step 1: Train cross-encoder: ({model_name}) with STSbenchmark")

# Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

gold_samples = [
    InputExample(texts=[sentence1, sentence2], label=data["score"])
    for data in train_dataset
    for sentence1, sentence2 in [(data["sentence1"], data["sentence2"]), (data["sentence2"], data["sentence1"])]
]

# We wrap gold_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(gold_samples, shuffle=True, batch_size=batch_size)

# We add an evaluator, which evaluates the performance during training
evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=[[data["sentence1"], data["sentence2"]] for data in eval_dataset],
    scores=[data["score"] for data in eval_dataset],
    name="sts-dev",
)

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info(f"Warmup-steps: {warmup_steps}")

# Train the cross-encoder model
cross_encoder.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=cross_encoder_path,
)

############################################################################
#
# Step 2: Label BM25 sampled STSb (silver dataset) using cross-encoder model
#
############################################################################

#### Top k similar sentences to be retrieved ####
#### Larger the k, bigger the silver dataset ####

index_name = "stsb"  # index-name should be in lowercase
logging.info(f"Step 2.1: Generate STSbenchmark (silver dataset) using top-{top_k} bm25 combinations")

unique_sentences = set()

for sample in gold_samples:
    unique_sentences.update(sample.texts)

unique_sentences = list(unique_sentences)  # unique sentences
sent2idx = {sentence: idx for idx, sentence in enumerate(unique_sentences)}  # storing id and sentence in dictionary
duplicates = set(
    (sent2idx[data.texts[0]], sent2idx[data.texts[1]]) for data in gold_samples
)  # not to include gold pairs of sentences again

# Ignore 400 cause by IndexAlreadyExistsException when creating an index
logging.info(f"Creating elastic-search index - {index_name}")
es.indices.create(index=index_name, ignore=[400])

# indexing all sentences
logging.info("Starting to index....")
for sent in unique_sentences:
    response = es.index(index=index_name, id=sent2idx[sent], body={"sent": sent})

logging.info(f"Indexing complete for {len(unique_sentences)} unique sentences")

silver_data = []
progress = tqdm.tqdm(unit="docs", total=len(sent2idx))

# retrieval of top-k sentences which forms the silver training data
for sent, idx in sent2idx.items():
    res = es.search(index=index_name, body={"query": {"match": {"sent": sent}}}, size=top_k)
    progress.update(1)
    for hit in res["hits"]["hits"]:
        if idx != int(hit["_id"]) and (idx, int(hit["_id"])) not in set(duplicates):
            silver_data.append((sent, hit["_source"]["sent"]))
            duplicates.add((idx, int(hit["_id"])))

progress.reset()
progress.close()

logging.info(f"Number of silver pairs generated for STSbenchmark: {len(silver_data)}")
logging.info(f"Step 2.2: Label STSbenchmark (silver dataset) with cross-encoder: {model_name}")

cross_encoder = CrossEncoder(cross_encoder_path)
silver_scores = cross_encoder.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

#################################################################################################
#
# Step 3: Train bi-encoder model with both (gold + silver) STSbenchmark dataset - Augmented SBERT
#
#################################################################################################

logging.info(f"Step 3: Train bi-encoder: {model_name} with STSbenchmark (gold + silver dataset)")

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark gold and silver train dataset")
silver_samples = Dataset.from_dict(
    {
        "sentence1": [data[0] for data in silver_data],
        "sentence2": [data[1] for data in silver_data],
        "score": silver_scores,
    }
)
train_dataset = concatenate_datasets([train_dataset, silver_samples])

train_loss = losses.CosineSimilarityLoss(model=sentence_transformer)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=sentence_transformer_path,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="augmentation-indomain-bm25-sts",  # Will be used in W&B if `wandb` is installed
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=sentence_transformer,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()


# Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(sentence_transformer)

# Save the trained & evaluated model locally
final_output_dir = f"{sentence_transformer_path}/final"
sentence_transformer.save(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    sentence_transformer.push_to_hub(f"{model_name}-augmentation-indomain-bm25-sts")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-augmentation-indomain-bm25-sts')`."
    )
