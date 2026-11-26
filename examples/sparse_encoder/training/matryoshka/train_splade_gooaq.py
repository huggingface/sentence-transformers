# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]",
# ]
# ///
"""Train SPLADE on GooAQ with hard negatives at several pruning budgets, with an optional baseline."""

import argparse
import logging
import traceback

from datasets import load_dataset
from transformers import set_seed

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    SparseMatryoshkaLoss,
    SparseMultipleNegativesRankingLoss,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Train without SparseMatryoshkaLoss.")
    cli = parser.parse_args()

    model_name = "distilbert/distilbert-base-uncased"
    train_batch_size = 32
    max_active_dims = [None, 256, 128, 64]
    seed = 42
    run_name = f"splade-{model_name.split('/')[-1]}-gooaq-matryoshka{'-baseline' if cli.baseline else ''}"

    # 1. Load a model to finetune
    # Seed before loading so repeated runs initialize any new MLM weights identically.
    set_seed(seed)
    model = SparseEncoder(
        model_name,
        model_kwargs={"torch_dtype": "float32"},
        model_card_data=SparseEncoderModelCardData(language="en", license="apache-2.0", model_name=run_name),
    )
    model.max_seq_length = 256

    # 2. Load the GooAQ dataset with hard negatives
    dataset = load_dataset("tomaarsen/gooaq-hard-negatives", "triplet-5", split="train").select(range(100_000))
    dataset = dataset.train_test_split(test_size=1_000, seed=12)
    logging.info(dataset)

    # 3. Define the training loss
    loss = SparseMultipleNegativesRankingLoss(model)
    if not cli.baseline:
        loss = SparseMatryoshkaLoss(model, loss, max_active_dims=max_active_dims)
    loss = CachedSpladeLoss(
        model,
        loss=loss,
        query_regularizer_weight=5e-5,
        document_regularizer_weight=1e-4,
        mini_batch_size=32,  # You can lower this if you have memory issues
    )

    # 4. Define evaluators for use during training
    # Monitor the full embeddings and the smallest budget to limit evaluation time.
    evaluators = [
        SparseNanoBEIREvaluator(
            dataset_names=["msmarco", "nfcorpus", "nq"],
            max_active_dims=budget,
            batch_size=train_batch_size,
            show_progress_bar=True,
        )
        for budget in [None, max_active_dims[-1]]
    ]

    # 5. Define the training arguments
    training_args = SparseEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.05,
        report_to="none",
        run_name=run_name,
        seed=seed,
        data_seed=seed,
    )

    # 6. Create the trainer and start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=loss,
        evaluator=evaluators,
    )
    trainer.train()

    # 7. Evaluate the final model at each budget on all NanoBEIR datasets
    for budget in max_active_dims:
        evaluator = SparseNanoBEIREvaluator(
            max_active_dims=budget,
            batch_size=train_batch_size,
            show_progress_bar=True,
        )
        evaluator(model)

    # 8. Save the final model
    final_output_dir = f"{training_args.output_dir}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub
    # Run `hf auth login` to log into your Hugging Face account first.
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`hf auth login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
