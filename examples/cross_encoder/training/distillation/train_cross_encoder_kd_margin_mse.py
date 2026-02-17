import logging
import traceback

from datasets import load_dataset, load_from_disk

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses.MarginMSELoss import MarginMSELoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments


def main():
    model_name = "microsoft/MiniLM-L12-H384-uncased"

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    train_batch_size = 16
    num_epochs = 1
    dataset_size = 2_000_000

    # 1. Define our CrossEncoder model
    model = CrossEncoder(model_name)
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Load the MS MARCO dataset: https://huggingface.co/datasets/sentence-transformers/msmarco
    logging.info("Read train dataset")
    try:
        train_dataset = load_from_disk("ms-marco-margin-mse-train")
        eval_dataset = load_from_disk("ms-marco-margin-mse-eval")
    except FileNotFoundError:
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

        train_dataset.save_to_disk("ms-marco-margin-mse-train")
        eval_dataset.save_to_disk("ms-marco-margin-mse-eval")
        logging.info(
            "The dataset has now been stored as texts on disk. The script will now stop to ensure that memory is freed. "
            "Please restart the script to start training."
        )
        quit()
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = MarginMSELoss(model)

    # 4. Define the evaluator. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-msmarco-margin-mse"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=8e-6,  # Lower than usual
        warmup_steps=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=20000,
        save_strategy="steps",
        save_steps=20000,
        save_total_limit=2,
        logging_steps=4000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
        dataloader_num_workers=4,
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
    final_output_dir = f"models/{run_name}/final"
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
