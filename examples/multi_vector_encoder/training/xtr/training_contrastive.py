"""
This script demonstrates how to train an XTR-style multi-vector retriever on MS MARCO triplets.

This is identical to the ColBERT-style contrastive recipe (../msmarco/training_contrastive.py) except for the score
metric: we pass `score_metric=XTRScores()` to switch from ColBERT-style MaxSim scoring to XTR-style global top-k token
scoring. The same losses, trainer, evaluator, and data pipeline work for both.

As dataset, we use sentence-transformers/msmarco-bm25, which has (query, positive, negative) triplets with BM25-mined
hard negatives.

As loss function, we use MultiVectorMultipleNegativesRankingLoss with an XTRScores score metric.
"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderModelCardData,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss
from sentence_transformers.multi_vector_encoder.scoring import XTRScores

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    model_name = "answerdotai/ModernBERT-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    train_batch_size = 32
    num_epochs = 1
    learning_rate = 3e-5

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    # Loading in fp32 is preferred for training if your memory can handle it
    model = MultiVectorEncoder(
        model_name,
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"XTR {short_model_name} trained on MS MARCO triplets",
        ),
        model_kwargs={"torch_dtype": "float32"},
    )

    # 2. Load the MS MARCO triplets dataset: https://huggingface.co/datasets/sentence-transformers/msmarco-bm25
    full_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train").select(range(51_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss. Same loss class as the ColBERT recipe; we only swap the scoring callable for
    # XTRScores, which keeps the top-k token matches per query token across all in-batch documents.
    loss = MultiVectorMultipleNegativesRankingLoss(model=model, score_metric=XTRScores(k=256))

    # 4. Define the evaluator. We use the MultiVectorNanoBEIREvaluator, which is a light-weight evaluator for English
    # (retrieval is scored with MaxSim, the standard late-interaction inference, regardless of the training score metric)
    evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=train_batch_size)
    # Run the base model through the evaluator first to get a baseline before training.
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"xtr-{short_model_name}-msmarco"
    args = MultiVectorEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=0.05,  # Warm up over the first 5% of training steps
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
    )

    # 6. Create the trainer & start training
    trainer = MultiVectorEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    test_evaluator = MultiVectorNanoBEIREvaluator(show_progress_bar=True, batch_size=train_batch_size)
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
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, "
            f"you can run `huggingface-cli login`, followed by loading the model using "
            f"`model = MultiVectorEncoder({final_output_dir!r})`, then saving it using "
            f"`model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
