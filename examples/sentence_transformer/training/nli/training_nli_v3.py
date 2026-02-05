"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with GISTEmbedLoss, using all-MiniLM-L6-v2 as an efficient guiding model. Entailments are positive pairs and the contradiction
on AllNLI dataset is added as a hard negative. At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v3.py

OR
python training_nli_v3.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments, BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
train_batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 1

# Save path of the model
output_dir = (
    "output/training_nli_v3_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
# We'll start with 10k training samples, but you can increase this to get a stronger model
logging.info("Read AllNLI train dataset")
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train").select(range(10000))
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev").select(range(1000))
logging.info(train_dataset)

# 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss
# The guiding model
guide_model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.GISTEmbedLoss(model, guide_model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)
logging.info("Evaluation before training:")
dev_evaluator(model)

# 5. Define the training arguments
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
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    logging_steps=100,
    run_name="nli-v3",  # Will be used in W&B if `wandb` is installed
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
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
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
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-nli-v3")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-v3')`."
    )
