"""
This example trains a SparseEncoder for the GooAQ dataset.
The training script fine-tunes a SparseEncoder using the Splade loss function for retrieval.
It loads a subset of the GooAQ dataset, splits it into training and evaluation subsets,
and trains the model as a retriever. After training, the model is evaluated and saved locally,
with an optional step to push the trained model to the Hugging Face Hub.

Usage:
python train_splade_gooaq.py
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
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sparse_encoder import evaluation, losses

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "distilbert/distilbert-base-uncased"

    train_batch_size = 32
    num_epochs = 1

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="splade-distilbert-base-uncased trained on GooAQ",
        ),
    )
    model.max_seq_length = 256  # Set the max sequence length to 256 for the training
    logging.info("Model max length: %s", model.max_seq_length)

    # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    logging.info("Read the gooaq training dataset")
    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss.
    query_regularizer_weight = 5e-5
    document_regularizer_weight = 3e-5

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=query_regularizer_weight,  # Weight for query loss
        document_regularizer_weight=document_regularizer_weight,  # Weight for document loss
    )

    # 4. Define evaluator. We use the SparseNanoBEIREvaluator, which is a light-weight evaluator
    evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"], show_progress_bar=True, batch_size=train_batch_size
    )

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"splade-{short_model_name}-gooaq"
    training_args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=610,
        save_strategy="steps",
        save_steps=610,
        save_total_limit=2,
        logging_steps=100,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
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

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    test_evaluator = evaluation.SparseNanoBEIREvaluator(show_progress_bar=True, batch_size=train_batch_size)
    test_evaluator(model)

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
