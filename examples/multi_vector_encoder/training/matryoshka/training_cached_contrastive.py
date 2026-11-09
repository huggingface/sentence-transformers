"""Train retrieval at smaller token embedding dimensions with a matched baseline."""

import argparse
import logging
import traceback

from datasets import load_dataset
from transformers import set_seed

from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer, MultiVectorEncoderTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import (
    CachedMultiVectorMultipleNegativesRankingLoss,
    MultiVectorMatryoshkaLoss,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Train with the ordinary cached contrastive loss.")
    cli = parser.parse_args()

    seed = 42
    num_train_samples = 30_000
    run_name = f"mve-ModernBERT-base-matryoshka{'-baseline' if cli.baseline else ''}"
    set_seed(seed)

    # 1. Initialize both losses from the same encoder and token projection
    model = MultiVectorEncoder("answerdotai/ModernBERT-base", model_kwargs={"torch_dtype": "float32"})
    # Expand shorter queries to at least 32 token positions
    model[0].query_expansion = {"strategy": "min", "length": 32}

    # 2. Keep the training rows and held-out split identical between runs
    dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
    dataset = dataset.select(range(num_train_samples + 1_000)).train_test_split(test_size=1_000, seed=42)

    # 3. Change only the loss for the baseline comparison
    loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=8)
    if not cli.baseline:
        loss = MultiVectorMatryoshkaLoss(model, loss, matryoshka_dims=[128, 64, 32], matryoshka_weights=[1 / 3] * 3)

    # 4. Evaluate full and truncated embeddings on MS MARCO during training
    evaluators = []
    for truncate_dim in [None, 32]:
        evaluators.append(
            MultiVectorNanoBEIREvaluator(
                dataset_names=["msmarco"],
                batch_size=8,
                corpus_chunk_size=512,
                chunk_elements=20_000_000,
                truncate_dim=truncate_dim,
            )
        )

    # 5. Train for a fixed budget
    args = MultiVectorEncoderTrainingArguments(
        output_dir=f"models/{run_name}",
        num_train_epochs=1,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=3e-5,
        warmup_steps=0.05,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        max_length=300,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="no",
        logging_steps=0.05,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = MultiVectorEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=loss,
        evaluator=evaluators,
    )
    trainer.train()

    # 6. Save the final model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 7. (Optional) save the model to the Hugging Face Hub!
    # Run `hf auth login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, "
            f"you can run `hf auth login`, followed by loading the model using "
            f"`model = MultiVectorEncoder({final_output_dir!r})`, then saving it using "
            f"`model.push_to_hub('{run_name}')`."
        )

    # 8. Retrieve with smaller token embeddings
    sample = dataset["test"][0]
    queries = model.encode_query(sample["query"], truncate_dim=32)
    documents = model.encode_document([sample["positive"], sample["negative"]], truncate_dim=32)
    logging.info("Retrieval scores: %s", model.similarity(queries, documents))


if __name__ == "__main__":
    main()
