"""
This script demonstrates how to train a ColBERT-style multi-vector model with knowledge distillation on MS MARCO.

As dataset, we use lightonai/ms-marco-en-bge, which provides per-query lists of N candidate document IDs together with
teacher scores from a BGE bi-encoder. KDProcessing resolves those IDs against the queries / documents datasets on the
fly during training.

As loss function, we use MultiVectorDistillKLDivLoss, which minimizes the KL divergence between the (softmaxed) teacher
scores and the student's MaxSim scores over each query's candidate documents. This recipe is adapted from PyLate's
`examples/train/knowledge_distillation.py`.
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
from sentence_transformers.multi_vector_encoder import KDProcessing
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    model_name = "answerdotai/ModernBERT-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    n_ways = 32  # Number of candidate documents (1 positive + negatives) scored per query
    train_batch_size = 4  # Each batch holds train_batch_size * n_ways documents, so keep it small
    num_epochs = 1
    learning_rate = 3e-5

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    # Loading in fp32 is preferred for training if your memory can handle it
    model = MultiVectorEncoder(
        model_name,
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ColBERT {short_model_name} distilled from BGE on MS MARCO",
        ),
        model_kwargs={"torch_dtype": "float32"},
    )
    print(model)

    # 2. Load the lightonai/ms-marco-en-bge dataset: https://huggingface.co/datasets/lightonai/ms-marco-en-bge
    # The `train` split holds query_id + document_ids + teacher scores; `queries` and `documents` hold the texts.
    train_dataset = load_dataset("lightonai/ms-marco-en-bge", "train", split="train").select(range(20_000))
    queries = load_dataset("lightonai/ms-marco-en-bge", "queries", split="train")
    documents = load_dataset("lightonai/ms-marco-en-bge", "documents", split="train")

    # KDProcessing resolves query_id -> query text and document_ids -> document texts on the fly.
    train_dataset.set_transform(KDProcessing(queries=queries, documents=documents, n_ways=n_ways).transform)
    logging.info(train_dataset)

    # 3. Define our training loss: KL divergence between the teacher and student score distributions
    loss = MultiVectorDistillKLDivLoss(model=model)

    # 4. Define the evaluator. We use the MultiVectorNanoBEIREvaluator, which is a light-weight evaluator for English
    evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=train_batch_size)
    # Run the base model through the evaluator first to get a baseline before training.
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"multivector-{short_model_name}-msmarco-kd"
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_MaxSim_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",  # The NanoBEIR evaluator runs on its own datasets, so no eval_dataset is needed
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
