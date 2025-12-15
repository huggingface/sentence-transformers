"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MatryoshkaLoss using MultipleNegativesRankingLoss. This trains a model at output dimensions [768, 512, 256, 128, 64].
Entailments are positive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset at the different output dimensions.

The difference between this script and matryoshka_nli.py is that this script uses a reduced dimensionality of the base
model by adding a Dense layer with `reduced_dim=256` output dimensions. This might be useful when your desired output
dimensionality is lower than the base model's default output dimensionality.

Usage:
python matryoshka_nli_reduced_dim.py

OR
python matryoshka_nli_reduced_dim.py pretrained_transformer_model_name
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
    models,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
)

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
batch_size = 128  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 1
reduced_dim = 256
matryoshka_dims = [256, 128, 64, 32, 16]

# Save path of the model
output_dir = (
    f"output/matryoshka_nli_reduced_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# dense = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=reduced_dim)
model.add_module(
    "reduced_dim", models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=reduced_dim)
)
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
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )
dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])

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
    run_name="matryoshka-nli-reduced",  # Will be used in W&B if `wandb` is installed
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
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-nli-matryoshka-reduced")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-matryoshka-reduced')`."
    )
