"""Train for a document token budget and compare against ordinary contrastive training."""

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
    MultiVectorTokenPoolingLoss,
)
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Use the ordinary contrastive loss.")
    cli = parser.parse_args()

    seed = 42
    num_train_samples = 120_000
    run_name = f"mve-ModernBERT-base-token-pooling{'-baseline' if cli.baseline else ''}"
    set_seed(seed)

    # 1. Initialize the encoder and token projection identically for both losses
    model = MultiVectorEncoder("answerdotai/ModernBERT-base", model_kwargs={"torch_dtype": "float32"})
    # Set query expansion to "expand to at least 32 tokens"
    model[0].query_expansion = {"strategy": "min", "length": 32}

    # 2. Use the same training data and held-out split
    dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
    dataset = dataset.select(range(num_train_samples + 1_000)).train_test_split(test_size=1_000, seed=42)

    # 3. Change only the loss for the baseline
    pooling = HierarchicalTokenPooling(pool_factor=4)
    loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=64)
    if not cli.baseline:
        loss = MultiVectorTokenPoolingLoss(
            model,
            loss,
            token_poolings=[None, pooling],
            weights=[0.5, 0.5],
        )

    # 4. Evaluate both deployment settings on MS MARCO during training
    evaluators = []
    for token_pooling in [None, pooling]:
        evaluators.append(
            MultiVectorNanoBEIREvaluator(
                dataset_names=["msmarco"],
                batch_size=16,
                corpus_chunk_size=1024,
                chunk_elements=100_000_000,
                token_pooling=token_pooling,
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

    # 8. Retrieve with compressed document embeddings
    sample = dataset["test"][0]
    queries = model.encode_query(sample["query"])
    documents = model.encode_document([sample["positive"], sample["negative"]], token_pooling=pooling)
    logging.info("Document token counts: %s", [len(embedding) for embedding in documents])
    logging.info("Retrieval scores: %s", model.similarity(queries, documents))


if __name__ == "__main__":
    main()
