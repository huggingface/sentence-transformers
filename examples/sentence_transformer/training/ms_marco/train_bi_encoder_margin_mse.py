"""
This example shows how to train a bi-encoder on the MS MARCO Passage Ranking dataset
(https://github.com/microsoft/MSMARCO-Passage-Ranking) using MarginMSELoss with the
canonical Hofstätter et al. 2020 distillation recipe (https://huggingface.co/papers/2010.02666).

For each ``(query, positive, negative)`` triplet we provide a label
``ce(query, positive) - ce(query, negative)`` computed by the BERT-CAT ensemble teacher
that Hofstätter et al. released. The bi-encoder learns to match those score differences
with its own dot-product similarity.

The data lives in `sentence-transformers/msmarco <https://huggingface.co/datasets/sentence-transformers/msmarco>`_:

- ``corpus``: maps ``passage_id -> passage``
- ``queries``: maps ``query_id -> query``
- ``bert-ensemble-margin-mse``: ``(query_id, positive_id, negative_id, score)`` triplets

For the combined MarginMSE + MultipleNegativesRankingLoss recipe (the canonical
``msmarco-distilbert-margin-mse-mnrl-mean-v1`` family of models), see
``train_bi_encoder_margin_mse_mnrl.py``.

Running this script:
python train_bi_encoder_margin_mse.py
"""

import logging
import random

import numpy
import torch
from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import MarginMSELoss

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)


def main():
    model_name = "distilbert/distilbert-base-uncased"
    train_batch_size = 32
    max_seq_length = 300
    dataset_size = 1_000_000  # bert-ensemble-margin-mse has ~40M triplets. Cap for a single-epoch example.

    # 1. Load a model to finetune with 2. (Optional) model card data. Weights stay in fp32
    # so the optimizer accumulates updates at full precision. `bf16=True` in TrainingArguments
    # below adds bf16 autocast on the forward/backward.
    short_model_name = model_name.split("/")[-1]
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{short_model_name} finetuned on MSMARCO with MarginMSELoss (BERT-CAT ensemble teacher)",
        ),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"model_max_length": max_seq_length},
        similarity_fn_name="dot",  # MarginMSELoss matches with raw dot product
    )

    # 3. Load corpus, queries, and the (query_id, positive_id, negative_id, score) triplets.
    logging.info("Loading MS MARCO corpus, queries, and bert-ensemble-margin-mse triplets")
    corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
    queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    queries = dict(zip(queries["query_id"], queries["query"]))
    dataset = load_dataset("sentence-transformers/msmarco", "bert-ensemble-margin-mse", split="train")
    dataset = dataset.select(range(dataset_size))

    # 4. Resolve IDs to text once with .map(). The ``score`` column stays as-is and is picked
    # up as the loss label by the data collator's default label-column heuristic.
    def id_to_text_map(batch):
        return {
            "query": [queries[qid] for qid in batch["query_id"]],
            "positive": [corpus[pid] for pid in batch["positive_id"]],
            "negative": [corpus[pid] for pid in batch["negative_id"]],
            "score": batch["score"],
        }

    dataset = dataset.map(
        id_to_text_map,
        batched=True,
        remove_columns=["query_id", "positive_id", "negative_id"],
        desc="Resolving IDs to text",
    )

    # 5. Define the loss. MarginMSELoss takes one negative per row and learns to match the
    # teacher's (pos_score - neg_score) margin.
    loss = MarginMSELoss(model)

    # 6. (Optional) Specify training arguments
    run_name = f"{short_model_name}-msmarco-margin-mse"
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    # 7. (Optional) Create an evaluator & evaluate the base model.
    dev_evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        dev_evaluator(model)

    # 8. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 9. Evaluate the model performance again after training
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        dev_evaluator(model)

    # 10. Save the trained model
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 11. (Optional) Push it to the Hugging Face Hub
    # Run `huggingface-cli login` first if you haven't already.
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub. To upload it manually, run "
            f"`huggingface-cli login`, then load with `model = SentenceTransformer({final_output_dir!r})` "
            f"and call `model.push_to_hub({run_name!r})`."
        )


if __name__ == "__main__":
    main()
