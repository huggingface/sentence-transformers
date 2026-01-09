"""
This file contains an example how to make a SentenceTransformer model faster and lighter.

This is achieved by using Knowledge Distillation: We use a well working teacher model to train
a fast and light student model. The student model learns to imitate the produced
sentence embeddings from the teacher. We train this on a diverse set of sentences we got
from SNLI + Multi+NLI + Wikipedia.

After the distillation is finished, the student model produce nearly the same embeddings as the
teacher, however, it will be much faster.

The script implements to options two options to initialize the student:
Option 1: Train a light transformer model like TinyBERT to imitate the teacher
Option 2: We take the teacher model and keep only certain layers, for example, only 4 layers.

Option 2) works usually better, as we keep most of the weights from the teacher. In Option 1, we have to tune all
weights in the student from scratch.

There is a performance - speed trade-off. However, we found that a student with 4 instead of 12 layers keeps about 99.4%
of the teacher performance, while being 2.3 times faster.
"""

import logging
import traceback
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.decomposition import PCA

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Teacher Model: Model we want to distill to a smaller model
teacher_model_name = "stsb-roberta-base-v2"
teacher_model = SentenceTransformer(teacher_model_name)

output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We will train a small model like TinyBERT to imitate the teacher.
# You can find some small BERT models here: https://huggingface.co/nreimers
student_model_name = "nreimers/TinyBERT_L-4_H-312_v2"
student_model = SentenceTransformer(student_model_name)

inference_batch_size = 64
train_batch_size = 64

logging.info("Load the AllNLI dataset")
# Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
nli_train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
nli_eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="dev")
# Concatenate all sentences into a new column "sentence"


def combine_sentences(batch):
    return {"sentence": batch["sentence1"] + batch["sentence2"]}


nli_train_dataset = nli_train_dataset.map(
    combine_sentences, batched=True, remove_columns=nli_train_dataset.column_names
)
nli_eval_dataset = nli_eval_dataset.map(combine_sentences, batched=True, remove_columns=nli_eval_dataset.column_names)


def deduplicate(dataset):
    df = pd.DataFrame(dataset)
    df = df.drop_duplicates()
    return Dataset.from_pandas(df, preserve_index=False)


nli_train_dataset = deduplicate(nli_train_dataset)
nli_eval_dataset = deduplicate(nli_eval_dataset)
logging.info(nli_train_dataset)


logging.info("Load the STSB dataset")
# Load the STSB eval/test datasets: https://huggingface.co/datasets/sentence-transformers/stsb
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(stsb_eval_dataset)


logging.info("Load the Wikipedia dataset")
# Load the Wikipedia dataset: https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences
wikipedia_train_dataset = load_dataset("sentence-transformers/wikipedia-en-sentences", split="train")
# Take 5000 random sentences from the Wikipedia dataset for evaluation
wikipedia_train_dataset_dict = wikipedia_train_dataset.train_test_split(test_size=5000)
wikipedia_train_dataset = wikipedia_train_dataset_dict["train"]
wikipedia_eval_dataset = wikipedia_train_dataset_dict["test"]
logging.info(wikipedia_train_dataset)


# Concatenate the NLI and Wikipedia datasets for training
train_dataset: Dataset = concatenate_datasets([nli_train_dataset, wikipedia_train_dataset])
# Create a relatively small dataset for evaluation
eval_dataset: Dataset = concatenate_datasets(
    [nli_eval_dataset.select(range(5000)), wikipedia_eval_dataset.select(range(5000))]
)

# Create an STSB evaluator
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Teacher Performance")
dev_evaluator_stsb(teacher_model)

# Student model has fewer dimensions. Compute PCA for the teacher to reduce the dimensions
if student_model.get_sentence_embedding_dimension() < teacher_model.get_sentence_embedding_dimension():
    logging.info("Student model has fewer dimensions than the teacher. Compute PCA for down projection")
    pca_sentences = nli_train_dataset[:20000]["sentence"] + wikipedia_train_dataset[:20000]["sentence"]
    pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
    pca = PCA(n_components=student_model.get_sentence_embedding_dimension())
    pca.fit(pca_embeddings)

    # Add Dense layer to teacher that projects the embeddings down to the student embedding size
    dense = models.Dense(
        in_features=teacher_model.get_sentence_embedding_dimension(),
        out_features=student_model.get_sentence_embedding_dimension(),
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    teacher_model.add_module("dense", dense)

    logging.info(f"Teacher Performance with {teacher_model.get_sentence_embedding_dimension()} dimensions:")
    dev_evaluator_stsb(teacher_model)


# Use the teacher model to get the gold embeddings
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }


train_dataset = train_dataset.select(range(200000))
train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
# Optionally, save the dataset to disk to speed up future runs
train_dataset.save_to_disk("datasets/distillation_train_dataset")
# from datasets import DatasetDict, load_from_disk

# train_dataset = load_from_disk("datasets/distillation_train_dataset")
# if isinstance(train_dataset, DatasetDict):
#     train_dataset = train_dataset["train"]
eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)

train_loss = losses.MSELoss(model=student_model)

# We create an evaluator, that measure the Mean Squared Error (MSE) between the teacher and the student embeddings
eval_sentences = eval_dataset["sentence"]
dev_evaluator_mse = evaluation.MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)
dev_evaluator = evaluation.SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse])

# Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    metric_for_best_model="eval_sts-dev_spearman_cosine",
    load_best_model_at_end=True,
    learning_rate=1e-4,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name="distillation-layer-reduction",  # Will be used in W&B if `wandb` is installed
)

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_test_dataset["sentence1"],
    sentences2=stsb_test_dataset["sentence2"],
    scores=stsb_test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(student_model)

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
if "/" in student_model_name:
    student_model_name = student_model_name.split("/")[-1]
if "/" in teacher_model_name:
    teacher_model_name = teacher_model_name.split("/")[-1]
repo_id = f"{student_model_name}-distilled-from-{teacher_model_name}"
try:
    student_model.push_to_hub(repo_id)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub({repo_id!r})`."
    )
