"""
This example runs a CNN after the word embedding lookup. The output of the CNN is than pooled,
for example with mean-pooling.


"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import BaseTrainingArguments
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
num_train_epochs = 1
batch_size = 32
output_dir = "output/training_stsbenchmark_cnn-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

# 2. Define the model
# Map tokens to vectors using BERT
word_embedding_model = models.Transformer(model_name)

cnn = models.CNN(
    in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
    out_channels=256,
    kernel_sizes=[1, 3, 5],
)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(
    cnn.get_word_embedding_dimension(),
    pooling_mode="mean",
)
model = SentenceTransformer(modules=[word_embedding_model, cnn, pooling_model])

# 3. Define our training loss
# CosineSimilarityLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) needs two text columns and
# one similarity score column (between 0 and 1)
train_loss = losses.CosineSimilarityLoss(model=model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
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
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="cnn",  # Will be used in W&B if `wandb` is installed
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
    model.push_to_hub(f"{model_name}-cnn")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-cnn')`."
    )
