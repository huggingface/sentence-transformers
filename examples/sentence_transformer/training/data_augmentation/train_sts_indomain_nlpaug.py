"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset with nlp textual augmentation.
We utilise nlpaug (https://github.com/makcedward/nlpaug) for data augmentation strategies over a single sentence.

We chose synonym replacement for our example with (can be extended to other techniques) -
    1. Word-embeddings (word2vec)
    2. WordNet
    3. Contextual word-embeddings (BERT)

Methodology:
Take a gold STSb pair, like (A, B, 0.6) Then replace synonyms in A and B, which gives you (A', B', 0.6)
These are the silver data and SBERT is finally trained on (gold + silver) STSb data.

Additional requirements:
pip install nlpaug

Information:
We went over the nlpaug package and found from our experience, the commonly used and effective technique
is synonym replacement with words. However feel free to use any textual data augmentation mentioned
in the example - (https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb)

You could also extend the easy data augmentation methods for other languages too, a good example can be
found here - (https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb)


Citation: https://huggingface.co/papers/2010.08240

Usage:
python train_sts_indomain_nlpaug.py
"""

import logging
import sys
import traceback
from datetime import datetime

import nlpaug.augmenter.word as naw
import torch
import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 1

output_dir = (
    "output/bi-encoder/stsb_indomain_eda_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
model = SentenceTransformer(model_name)

# Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

##################################################################################
#
# Data Augmentation: Synonym Replacement with word2vec, BERT, WordNet using nlpaug
#
##################################################################################

logging.info("Starting with synonym replacement...")

#### Synonym replacement using Word2Vec ####
# Download the word2vec pre-trained Google News corpus (GoogleNews-vectors-negative300.bin)
# link: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

# aug = naw.WordEmbsAug(
#     model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
#     action="substitute")

#### Synonym replacement using WordNet ####
# aug = naw.SynonymAug(aug_src='wordnet')

#### Synonym replacement using BERT ####
aug = naw.ContextualWordEmbsAug(model_path=model_name, action="insert")

silver_samples = {
    "sentence1": [],
    "sentence2": [],
    "score": [],
}
progress = tqdm.tqdm(unit="docs", total=len(test_dataset))

for sample in train_dataset:
    augmented_texts = aug.augment([sample["sentence1"], sample["sentence2"]])
    silver_samples["sentence1"].append(augmented_texts[0])
    silver_samples["sentence2"].append(augmented_texts[1])
    silver_samples["score"].append(sample["score"])
    progress.update(1)

silver_dataset = Dataset.from_dict(silver_samples)

progress.reset()
progress.close()
logging.info("Textual augmentation completed....")
logging.info(f"Number of silver pairs generated: {len(silver_samples)}")

###################################################################
#
# Train SBERT model with both (gold + silver) STS benchmark dataset
#
###################################################################

train_dataset = concatenate_datasets([train_dataset, silver_dataset])
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
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
    run_name="augmentation-indomain-nlpaug-sts",  # Will be used in W&B if `wandb` is installed
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
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
test_evaluator(model)

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-augmentation-indomain-nlpaug-sts")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-augmentation-indomain-nlpaug-sts')`."
    )
