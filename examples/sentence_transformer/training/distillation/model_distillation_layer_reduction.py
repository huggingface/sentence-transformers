"""
Distill a slow but high-quality teacher SentenceTransformer into a faster student
by surgically dropping layers from the teacher and fine-tuning the remaining stack
to recover the teacher's embeddings.

This script starts from ``mxbai-embed-large-v1`` and keeps every third transformer
layer (8 of 24), then trains the resulting student to imitate the original teacher's
sentence embeddings on a diverse corpus (SNLI + Multi-NLI + Wikipedia). For an
alternative that trains a separate small model from scratch, see
``model_distillation.py``.

Pipeline:

1. Clone the teacher into a student, then drop layers from the student's encoder.
2. Pre-compute teacher embeddings for every training sentence and store them as the
   dataset's ``label`` column. Cached to disk so subsequent runs skip the teacher
   inference step.
3. Train the student with :class:`MSELoss`. No projection is needed: the layer-reduced
   student inherits the teacher's hidden size, so it stays a drop-in replacement at
   the same dimensionality.

This recipe usually outperforms training a small model from scratch (Option 1 in the
README) because it keeps most of the teacher's weights, which already encode useful
linguistic structure.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.sentence_transformer.losses import MSELoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


# Teacher Model: Model we want to distill into a smaller model
teacher_model_name = "mixedbread-ai/mxbai-embed-large-v1"
teacher_model = SentenceTransformer(teacher_model_name)

output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a smaller student model by using only some of the teacher layers
student_model = SentenceTransformer(teacher_model_name)

# Get the underlying transformers model so we can surgically remove layers
auto_model = student_model.transformers_model

# Which layers to keep from the teacher model. We equally spread the layers to keep over the original teacher
# layers_to_keep = [5]
# layers_to_keep = [3, 7]
# layers_to_keep = [3, 7, 11]
# layers_to_keep = [0, 2, 4, 6, 8, 10]
# layers_to_keep = [0, 1, 3, 4, 6, 7, 9, 10]
# Keep every third layer:
layers_to_keep = [0, 3, 6, 9, 12, 15, 18, 21]

logging.info(f"Remove layers from student. Only keep these layers: {layers_to_keep}")
new_layers = torch.nn.ModuleList(
    [layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep]
)
auto_model.encoder.layer = new_layers
auto_model.config.num_hidden_layers = len(layers_to_keep)
print(
    f"Number of parameters in the Teacher model: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)}"
)
print(
    f"Number of parameters in the Student model: {sum(p.numel() for p in student_model.parameters() if p.requires_grad)}"
)

inference_batch_size = 128
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
# Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
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


# Use the teacher model to get the gold embeddings for each sentence
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }


# Pre-compute teacher embeddings for the training set, caching to disk so reruns skip
# the teacher inference step. Eval is small enough to recompute each run.
# Note: distinct cache path from model_distillation.py since the teacher (and embedding
# dim) is different.
train_cache_dir = Path("datasets/distillation_layer_reduction_train_dataset")
if train_cache_dir.exists():
    logging.info("Loading pre-computed teacher embeddings from disk...")
    train_dataset = load_from_disk(str(train_cache_dir))
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
else:
    train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
    train_dataset.save_to_disk(str(train_cache_dir))

eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)

# Prepare the training loss. The student inherits the teacher's hidden size after layer
# reduction, so no projection_dim is needed.
train_loss = MSELoss(model=student_model)

# Create an STSB evaluator
dev_evaluator_stsb = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Running STSB evaluation on the teacher model")
dev_evaluator_stsb(teacher_model)

# We create an evaluator that measures the Mean Squared Error (MSE) between the teacher and the student embeddings
eval_sentences = eval_dataset["sentence"]
dev_evaluator_mse = MSEEvaluator(eval_sentences, eval_sentences, teacher_model=teacher_model)
dev_evaluator = SequentialEvaluator([dev_evaluator_stsb, dev_evaluator_mse])

# Run the evaluator before training to get a baseline performance of the student model
dev_evaluator(student_model)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
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
    eval_steps=5000,
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,
    logging_steps=1000,
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
model_name = teacher_model_name if "/" not in teacher_model_name else teacher_model_name.split("/")[-1]
try:
    student_model.push_to_hub(f"{model_name}-{len(layers_to_keep)}-layers")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-{len(layers_to_keep)}-layers')`."
    )
