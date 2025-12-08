"""
This script contains an example how to extend an existent sentence embedding model to new languages.

Given a (monolingual) teacher model you would like to extend to new languages, which is specified in the teacher_model_name
variable. We train a multilingual student model to imitate the teacher model (variable student_model_name)
on multiple languages.

For training, you need parallel sentence data (machine translation training data). You need tab-seperated files (.tsv)
with the first column a sentence in a language understood by the teacher model, e.g. English,
and the further columns contain the according translations for languages you want to extend to.

This scripts downloads automatically the parallel sentences corpus. This corpus contains transcripts from
talks translated to 100+ languages. For other parallel data, see get_parallel_data_[].py scripts

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://huggingface.co/papers/2004.09813
"""

import logging
import traceback
from datetime import datetime

import numpy as np
from datasets import DatasetDict, load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.losses import MSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)


# The teacher model is monolingual, we use it for English embeddings
teacher_model_name = "paraphrase-distilroberta-base-v2"
# The student model is multilingual, we train it such that embeddings of non-English texts mimic the teacher model's English embeddings
student_model_name = "xlm-roberta-base"

student_max_seq_length = 128  # Student model max. lengths for inputs (number of word pieces)
train_batch_size = 64  # Batch size for training
inference_batch_size = 64  # Batch size at inference
max_sentences_per_language = 500000  # Maximum number of  parallel sentences for training

num_train_epochs = 5  # Train for x epochs
num_evaluation_steps = 5000  # Evaluate performance after every xxxx steps


# Define the language codes you would like to extend the model to
source_languages = set(["en"])  # Our teacher model accepts English (en) sentences
# We want to extend the model to these new languages. For language codes, see the header of the train file
target_languages = set(["de", "es", "it", "fr", "ar", "tr"])


output_dir = (
    "output/make-multilingual-"
    + "-".join(sorted(list(source_languages)) + sorted(list(target_languages)))
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# 1a. Here we define our SentenceTransformer teacher model.
teacher_model = SentenceTransformer(teacher_model_name)
# If we want, we can limit the maximum sequence length for the model
# teacher_model.max_seq_length = 128
logging.info(f"Teacher model: {teacher_model}")

# 1b. Here we define our SentenceTransformer student model. If not already a Sentence Transformer model,
# it will automatically create one with "mean" pooling.
student_model = SentenceTransformer(student_model_name)
# If we want, we can limit the maximum sequence length for the model
student_model.max_seq_length = student_max_seq_length
logging.info(f"Student model: {student_model}")

# 2. Load the parallel sentences training dataset: https://huggingface.co/datasets?other=sentence-transformers&sort=trending&search=parallel-sentences
# NOTE: We can also use multiple datasets if we want
dataset_to_use = "sentence-transformers/parallel-sentences-talks"
# dataset_to_use = "sentence-transformers/parallel-sentences-europarl"
# dataset_to_use = "sentence-transformers/parallel-sentences-global-voices"
# dataset_to_use = "sentence-transformers/parallel-sentences-muse"
# dataset_to_use = "sentence-transformers/parallel-sentences-jw300"
# dataset_to_use = "sentence-transformers/parallel-sentences-news-commentary"
# dataset_to_use = "sentence-transformers/parallel-sentences-opensubtitles"
# dataset_to_use = "sentence-transformers/parallel-sentences-tatoeba"
# dataset_to_use = "sentence-transformers/parallel-sentences-wikimatrix"
# dataset_to_use = "sentence-transformers/parallel-sentences-wikititles"
train_dataset_dict = DatasetDict()
eval_dataset_dict = DatasetDict()
for source_lang in source_languages:
    for target_lang in target_languages:
        subset = f"{source_lang}-{target_lang}"
        try:
            train_dataset = load_dataset(dataset_to_use, subset, split="train")
            if len(train_dataset) > max_sentences_per_language:
                train_dataset = train_dataset.select(range(max_sentences_per_language))
        except Exception as exc:
            logging.error(f"Could not load dataset {dataset_to_use}/{source_lang}-{target_lang}: {exc}")
            continue

        try:
            eval_dataset = load_dataset(dataset_to_use, subset, split="dev")
            if len(eval_dataset) > 1000:
                eval_dataset = eval_dataset.select(range(1000))
        except Exception:
            logging.info(
                f"Could not load dataset {dataset_to_use}/{source_lang}-{target_lang} dev split, splitting 1k samples from train"
            )
            dataset = train_dataset.train_test_split(test_size=1000, shuffle=True)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]

        train_dataset_dict[subset] = train_dataset
        eval_dataset_dict[subset] = eval_dataset
logging.info(train_dataset_dict)


# We want the student EN embeddings to be similar to the teacher EN embeddings and
# the student non-EN embeddings to be similar to the teacher EN embeddings
def prepare_dataset(batch):
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": teacher_model.encode(batch["english"], batch_size=inference_batch_size, show_progress_bar=False),
    }


column_names = list(train_dataset_dict.values())[0].column_names
train_dataset_dict = train_dataset_dict.map(
    prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names
)
logging.info("Prepared datasets for training:", train_dataset_dict)

# 3. Define our training loss
# MSELoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss) needs one text columns and one
# column with embeddings from the teacher model
train_loss = MSELoss(model=student_model)

# 4. Define evaluators for use during training. This is useful to keep track of alongside the evaluation loss.
evaluators = []

for subset, eval_dataset in eval_dataset_dict.items():
    logger.info(f"Creating evaluators for {subset}")

    # Mean Squared Error (MSE) measures the (euclidean) distance between teacher and student embeddings
    dev_mse = MSEEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name=subset,
        teacher_model=teacher_model,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_mse)

    # TranslationEvaluator computes the embeddings for all parallel sentences. It then check if the embedding of
    # source[i] is the closest to target[i] out of all available target sentences
    dev_trans_acc = TranslationEvaluator(
        source_sentences=eval_dataset["english"],
        target_sentences=eval_dataset["non_english"],
        name=subset,
        batch_size=inference_batch_size,
    )
    evaluators.append(dev_trans_acc)

    # Try to load this subset from STS17
    test_dataset = None
    try:
        test_dataset = load_dataset("mteb/sts17-crosslingual-sts", subset, split="test")
    except Exception:
        try:
            test_dataset = load_dataset("mteb/sts17-crosslingual-sts", f"{subset[3:]}-{subset[:2]}", split="test")
            subset = f"{subset[3:]}-{subset[:2]}"
        except Exception:
            pass
    if test_dataset:
        test_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=[score / 5.0 for score in test_dataset["score"]],  # Convert 0-5 scores to 0-1 scores
            batch_size=inference_batch_size,
            name=f"sts17-{subset}-test",
            show_progress_bar=False,
        )
        evaluators.append(test_evaluator)

evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: np.mean(scores))
# Now also prepare the evaluation datasets for training
eval_dataset_dict = eval_dataset_dict.map(prepare_dataset, batched=True, batch_size=30000, remove_columns=column_names)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    learning_rate=2e-5,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=num_evaluation_steps,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=100,
    run_name=f"multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset_dict,
    eval_dataset=eval_dataset_dict,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()

# 7. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)

# 8. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = student_model_name if "/" not in student_model_name else student_model_name.split("/")[-1]
try:
    student_model.push_to_hub(f"{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-multilingual-{'-'.join(source_languages)}-{'-'.join(target_languages)}')`."
    )
