import logging
import traceback
from datetime import datetime

import torch
from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import LambdaLoss, NDCGLoss2PPScheme
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments


def main():
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # train_batch_size and eval_batch_size inform the size of the batches, while mini_batch_size is used by the loss
    # to subdivide the batch into smaller parts. This mini_batch_size largely informs the training speed and memory usage.
    # Keep in mind that the loss does not process `train_batch_size` pairs, but `train_batch_size * num_docs` pairs.
    train_batch_size = 16
    eval_batch_size = 16
    mini_batch_size = 16
    num_epochs = 1
    max_docs = 20

    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Define our CrossEncoder model
    # Set the seed so the new classifier weights are identical in subsequent runs
    torch.manual_seed(12)
    model = CrossEncoder(model_name, num_labels=1)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    corpus = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    corpus = dict(zip(corpus["passage_id"], corpus["passage"]))
    queries = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    queries = dict(zip(queries["query_id"], queries["query"]))
    dataset = load_dataset("sentence-transformers/msmarco", "labeled-list", split="train")

    def it_to_text_transform(batch):
        return {
            "query_id": [queries[qid] for qid in batch["query_id"]],
            "doc_ids": [[corpus[pid] for pid in doc_ids[:max_docs]] for doc_ids in batch["doc_ids"]],
            "labels": [labels[:max_docs] for labels in batch["labels"]],
        }

    dataset.set_transform(it_to_text_transform)
    dataset = dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)
    logging.info(train_dataset[0])

    # 3. Define our training loss
    loss = LambdaLoss(
        model=model,
        weighting_scheme=NDCGLoss2PPScheme(),
        mini_batch_size=mini_batch_size,
    )

    # 4. Define the evaluator. We use the CENanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=eval_batch_size)
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-{short_model_name}-lambdaloss"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}_{dt}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        warmup_steps=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    final_output_dir = f"models/{run_name}_{dt}/final"
    model.save_pretrained(final_output_dir)

    # 9. (Optional) save the model to the Hugging Face Hub!
    # It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
    try:
        model.push_to_hub(run_name)
    except Exception:
        logging.error(
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
