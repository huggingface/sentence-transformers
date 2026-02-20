import logging
import random
import traceback

from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.models import LandmarkTransformer, Normalize, Pooling
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# 1. Load a model to finetune with 2. (Optional) model card data
base_model_name = "jhu-clsp/ettin-encoder-150m"
short_model_name = base_model_name.split("/")[-1]
# Use the LandmarkTransformer to insert landmark tokens at regular intervals, which can help the model capture long-range
# dependencies and improve performance on longer texts. The pooling layer is set to "lmk" to derive sentence embeddings
# from these landmark tokens.
landmark_transformer = LandmarkTransformer(
    base_model_name, landmark_interval=32, train_landmark_interval=[32, 64, 128, 256]
)
pooling = Pooling(
    landmark_transformer.get_word_embedding_dimension(), "lmk", lmk_token_id=landmark_transformer.lmk_token_id
)
print(f"Using landmark token ID {landmark_transformer.lmk_token_id} for pooling.")
"""
# TODO: Remove this; it's for testing against MEAN Pooling
transformer = Transformer(base_model_name)
pooling = Pooling(transformer.get_word_embedding_dimension(), "mean")
"""
normalize = Normalize()
model = SentenceTransformer(
    modules=[landmark_transformer, pooling, normalize],
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name=f"{base_model_name} with Landmark Pooling trained on MIRIAD pairs using CachedMultipleNegativesRankingLoss",
    ),
)

# 3. Load a dataset to finetune on
dataset = load_dataset("tomaarsen/miriad-4.4M-split", split="train").select(range(100_000))
dataset = dataset.add_column("id", range(len(dataset)))
dataset_dict = dataset.train_test_split(test_size=1_000, seed=12)
train_dataset: Dataset = dataset_dict["train"]
eval_dataset: Dataset = dataset_dict["test"]

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)

# 5. (Optional) Specify training arguments
run_name = f"{short_model_name}-miriad-cmnrl-lmk"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    learning_rate=8e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # CachedMultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.05,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
# The full corpus, but only the evaluation queries
corpus = dict(zip(dataset["id"], dataset["passage_text"]))
random.seed(12)
queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
corpus = {qid: dataset[qid]["passage_text"] for qid in queries}
relevant_docs = {qid: {qid} for qid in eval_dataset["id"]}
dev_evaluator = InformationRetrievalEvaluator(
    corpus=corpus,
    queries=queries,
    relevant_docs=relevant_docs,
    show_progress_bar=True,
    name="miriad-dev",
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset.remove_columns("id"),
    eval_dataset=eval_dataset.remove_columns("id"),
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluator after training
dev_evaluator(model)

# 8. Save the trained model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) Push it to the Hugging Face Hub
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
