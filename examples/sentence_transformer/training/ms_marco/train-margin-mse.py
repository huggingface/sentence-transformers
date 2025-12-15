import logging
import random

import numpy
import torch
from datasets import load_dataset

from sentence_transformers import (
    BaseModelCardData,
    BaseTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
)
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    SequentialEvaluator,
)
from sentence_transformers.sentence_transformer.losses import MarginMSELoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)


def main():
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "microsoft/mpnet-base",
        model_card_data=BaseModelCardData(
            language="en",
            license="apache-2.0",
            model_name="mpnet-base finetuned on MSMARCO via distillation",
        ),
    )

    # 3. Load the MS MARCO dataset: https://huggingface.co/datasets/tomaarsen/msmarco-Qwen3-Reranker-0.6B
    logging.info("Read datasets")
    train_dataset = load_dataset("tomaarsen/msmarco-Qwen3-Reranker-0.6B", split="train").select(range(10_000))
    eval_dataset = load_dataset("tomaarsen/msmarco-Qwen3-Reranker-0.6B", split="eval").select(range(1_000))
    logging.info(train_dataset)
    logging.info(train_dataset[0])

    # 4. Define a loss function
    loss = MarginMSELoss(model=model)

    # 5. (Optional) Specify training arguments
    run_name = "mpnet-base-msmarco-margin-mse"
    args = BaseTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=4e-5,
        warmup_steps=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        logging_steps=20,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    queries = eval_dataset["query"][:1000]
    eval_queries = {query_id: query for query_id, query in enumerate(queries)}
    corpus = eval_dataset["positive"]
    eval_corpus = {doc_id: doc for doc_id, doc in enumerate(corpus)}
    eval_relevant_docs = {index: [index] for index in range(len(queries))}
    dev_evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_relevant_docs,
        batch_size=16,
        name="msmarco-eval-1kq-1kd",
    )
    nano_beir_evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)
    dev_evaluator = SequentialEvaluator([dev_evaluator, nano_beir_evaluator])

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 8. Evaluate the model performance again after training
    dev_evaluator(model)

    # 9. Save the trained model
    model.save_pretrained(f"models/{run_name}/final")
    print(f"Model saved to models/{run_name}/final")

    # 10. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub(run_name)


if __name__ == "__main__":
    main()
