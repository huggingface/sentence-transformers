"""
This is an example how to train SentenceTransformers in a multi-task setup.

The system trains BERT on the AllNLI and on the STSbenchmark dataset.
"""

import logging
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments, MultiDatasetBatchSamplers
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss, SoftmaxLoss
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Read the dataset
model_name = "bert-base-uncased"
num_train_epochs = 1
batch_size = 16
output_dir = "output/training_multi-task_" + model_name + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2a. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
nli_train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
nli_eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev").select(range(1000))
logging.info(nli_train_dataset)

# 2b. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
stsb_train_dataset = load_dataset("sentence-transformers/stsb", split="train")
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(stsb_train_dataset)

# 3. Define our training losses
# 3a. SoftmaxLoss for the NLI data (sentence_A, sentence_B, class), see also https://sbert.net/docs/training/loss_overview.html
train_loss_nli = SoftmaxLoss(
    model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=3
)
# 3b. CosineSimilarityLoss for the STSB data (sentence_A, sentence_B, similarity score between 0 and 1)
train_loss_sts = CosineSimilarityLoss(model=model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
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
    # With ROUND_ROBIN you'll sample the same amount from each dataset, until one of the multi-datasets is exhausted
    # The alternative is PROPORTIONAL, which samples from each dataset in proportion to the dataset size,
    # but that will lead to a lot of samples from the larger dataset (AllNLI in this case)
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="multi-task",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset={
        "all-nli": nli_train_dataset,
        "sts": stsb_train_dataset,
    },
    eval_dataset={
        "all-nli": nli_eval_dataset,
        "sts": stsb_eval_dataset,
    },
    loss={
        "all-nli": train_loss_nli,
        "sts": train_loss_sts,
    },
    evaluator=dev_evaluator,
)
trainer.train()


# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_test_dataset["sentence1"],
    sentences2=stsb_test_dataset["sentence2"],
    scores=stsb_test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-multi-task")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-multi-task')`."
    )
