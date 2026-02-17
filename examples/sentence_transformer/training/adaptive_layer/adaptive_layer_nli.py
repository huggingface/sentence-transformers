"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with AdaptiveLayerLoss using MultipleNegativesRankingLoss. Entailing texts are used as positive pairs and contradictory
texts are seen as negative pairs. At every 100 training steps, the model is evaluated on the STS benchmark dataset.

Usage:
python adaptive_layer_nli.py

OR
python adaptive_layer_nli.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import (
    BaseTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 1

# Save path of the model
output_dir = f"output/adaptive_layer_nli_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
logging.info(train_dataset)

# If you wish, you can limit the number of training samples
# train_dataset = train_dataset.select(range(5000))

# 3. Define our training loss
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.AdaptiveLayerLoss(model, inner_train_loss)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
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
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="adaptive-layer-nli",  # Will be used in W&B if `wandb` is installed
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
    model.push_to_hub(f"{model_name}-nli-adaptive-layer")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-adaptive-layer')`."
    )
