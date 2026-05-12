"""
Distill a slow but high-quality teacher SentenceTransformer into a smaller, faster student
by training the student to imitate the teacher's sentence embeddings.

This script trains a light transformer (TinyBERT) to match the embeddings of a
``stsb-roberta-base-v2`` teacher on a diverse corpus (SNLI + Multi-NLI + Wikipedia).
For the alternative that keeps a subset of the teacher's own layers (and produces a
drop-in replacement at the teacher's dimensionality), see
``model_distillation_layer_reduction.py``.

Pipeline:

1. Pre-compute teacher embeddings for every training sentence and store them as the
   dataset's ``label`` column. This is cached to disk so subsequent runs skip the
   teacher inference step.
2. Train the student with :class:`EmbedDistillLoss` using cosine distance. When the
   student and teacher have different embedding dimensions (the default here: 312-dim
   student vs 768-dim teacher), the loss carries a learnable projection that maps
   student embeddings into the teacher's space for the comparison. The projection
   lives on the loss, not the model, so the final student keeps its native output
   dimension.

The student we get is a faster, narrower model, not a drop-in replacement for the
teacher. For a drop-in replacement at the teacher's dimensionality, use the
layer-reduction recipe instead. Even with this trade-off, the student typically
retains 97-99% of the teacher's benchmark performance while being 2-3x faster.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import EmbedDistillLoss
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util.similarity import SimilarityFunction

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Teacher Model: Model we want to distill into a smaller model
teacher_model_name = "sentence-transformers/stsb-roberta-base-v2"
teacher_model = SentenceTransformer(teacher_model_name)

output_dir = "output/model-distillation-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We will train a small TinyBERT model to imitate the teacher.
# You can find some small BERT models here: https://huggingface.co/nreimers
student_model_name = "nreimers/TinyBERT_L-4_H-312_v2"
# Loading in fp32 is preferred for training if your memory can handle it
student_model = SentenceTransformer(student_model_name, model_kwargs={"torch_dtype": "float32"})

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
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Teacher Performance")
dev_evaluator(teacher_model)


# Use the teacher model to get the gold embeddings for each sentence
def map_embeddings(batch):
    return {
        "label": teacher_model.encode(
            batch["sentence"], batch_size=inference_batch_size, show_progress_bar=False
        ).tolist()
    }


# Pre-compute teacher embeddings for the training set, caching to disk so reruns skip
# the teacher inference step. Eval is small enough to recompute each run.
train_cache_dir = Path("datasets/distillation_train_dataset")
if train_cache_dir.exists():
    logging.info("Loading pre-computed teacher embeddings from disk...")
    train_dataset = load_from_disk(str(train_cache_dir))
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]
else:
    train_dataset = train_dataset.select(range(200000))
    train_dataset = train_dataset.map(map_embeddings, batched=True, batch_size=50000)
    train_dataset.save_to_disk(str(train_cache_dir))

eval_dataset = eval_dataset.map(map_embeddings, batched=True, batch_size=50000)

# If the student and teacher have different embedding dimensions (the default here:
# 312-dim student vs 768-dim teacher), we ask the loss for a learnable projection that
# maps student embeddings into the teacher's space during training. The projection only
# exists on the loss; the saved student keeps its native output dimension.
student_dim = student_model.get_embedding_dimension()
teacher_dim = teacher_model.get_embedding_dimension()
projection_dim = teacher_dim if student_dim != teacher_dim else None
train_loss = EmbedDistillLoss(
    model=student_model,
    distance_metric="cosine",
    projection_dim=projection_dim,
)

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
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    run_name="distillation-tinybert",  # Will be used in W&B if `wandb` is installed
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

# (Optional) persist the projection layer for multi-stage training.
# Reuse via `loss.load_projection(...)` on the next-stage loss.
# train_loss.save_projection(f"{output_dir}/projection.safetensors")

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
