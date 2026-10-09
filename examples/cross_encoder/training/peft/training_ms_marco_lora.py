import logging
import sys
import traceback

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType

from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    train_batch_size = 32
    num_epochs = 1

    # 1. Define our CrossEncoder model
    # Set the seed so the newly added adapter weights are identical in subsequent runs
    torch.manual_seed(12)
    # Loading in fp32 is preferred for training if your memory can handle it
    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="ms-marco-MiniLM-L6-v2 adapter finetuned on MS MARCO",
        ),
        model_kwargs={"torch_dtype": "float32"},
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    # 2. Create a LoRA adapter for the model & add it. Only the adapter weights
    # are trained, the base model weights stay frozen.
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model.add_adapter(peft_config)

    # 3. Load the MS MARCO dataset: https://huggingface.co/datasets/microsoft/ms_marco
    logging.info("Read train dataset")
    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    def bce_mapper(batch):
        queries = []
        passages = []
        labels = []
        for query, passages_info in zip(batch["query"], batch["passages"]):
            for idx, is_selected in enumerate(passages_info["is_selected"]):
                queries.append(query)
                passages.append(passages_info["passage_text"][idx])
                labels.append(is_selected)
        return {"query": queries, "passage": passages, "label": labels}

    dataset = dataset.map(bce_mapper, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=10_000)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    logging.info(train_dataset)

    # 4. Define our training loss
    loss = BinaryCrossEntropyLoss(model)

    # 5. Define the evaluator. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    evaluator = CrossEncoderNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    evaluator(model)

    # 6. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-msmarco-v1.1-{short_model_name}-lora"
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_steps=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_R100_mean_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=4_000,
        save_strategy="steps",
        save_steps=4_000,
        save_total_limit=2,
        logging_steps=1_000,
        logging_first_step=True,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=12,
    )

    # 7. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 8. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 9. Save the final model. Only the adapter weights are saved, the base model
    # weights are referenced instead of duplicated.
    final_output_dir = f"models/{run_name}/final"
    model.save_pretrained(final_output_dir)

    # 10. (Optional) save the model to the Hugging Face Hub!
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
