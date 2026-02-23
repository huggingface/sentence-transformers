"""
This scripts demonstrates how to train a Sparse Encoder model for Information Retrieval
using CachedSpladeLoss, which enables much larger batch sizes without additional GPU memory.

As dataset, we use MSMARCO version with hard negatives from the bert-ensemble-margin-mse dataset.

As loss function, we use MarginMSELoss in the CachedSpladeLoss.

Usage:
python train_splade_msmarco_margin_mse_cached.py
"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder import evaluation, losses

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "distilbert/distilbert-base-uncased"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    global_batch_size = 512
    mini_batch_size = 32

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"splade-{short_model_name} trained on MS MARCO hard negatives with distillation",
        ),
    )
    model.max_seq_length = 256  # Set the max sequence length to 256 for the training
    logging.info("Model max length: %s", model.max_seq_length)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    dataset_size = 100_000  # We only use the first 100k samples for training
    logging.info("The dataset has not been fully stored as texts on disk yet. We will do this now.")
    corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
    queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    queries = dict(zip(queries["query_id"], queries["query"]))
    dataset = load_dataset("sentence-transformers/msmarco", "bert-ensemble-margin-mse", split="train")
    dataset = dataset.select(range(dataset_size))

    def id_to_text_map(batch):
        return {
            "query": [queries[qid] for qid in batch["query_id"]],
            "positive": [corpus[pid] for pid in batch["positive_id"]],
            "negative": [corpus[pid] for pid in batch["negative_id"]],
            "score": batch["score"],
        }

    dataset = dataset.map(id_to_text_map, batched=True, remove_columns=["query_id", "positive_id", "negative_id"])
    dataset = dataset.train_test_split(test_size=10_000)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 3. Define our training loss.
    query_regularizer_weight = 5e-5
    document_regularizer_weight = 3e-5

    loss = losses.CachedSpladeLoss(
        model=model,
        loss=losses.SparseMarginMSELoss(model=model),
        mini_batch_size=mini_batch_size,
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )

    # 4. Define evaluator. We use the SparseNanoBEIREvaluator, which is a light-weight evaluator
    evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"], show_progress_bar=True, batch_size=mini_batch_size
    )
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"splade-{short_model_name}-msmarco-hard-negatives-{global_batch_size}bs"
    training_args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=global_batch_size,
        per_device_eval_batch_size=global_batch_size,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        save_total_limit=2,
        logging_steps=0.05,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
        # Uncomment the following lines to enable loading the best model at the end of training based on evaluation performance
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model again
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
