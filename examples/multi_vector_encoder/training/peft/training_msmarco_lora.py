"""
This script demonstrates how to train a ColBERT-style multi-vector (late-interaction) model for Information Retrieval
with a LoRA adapter from PEFT, i.e. without finetuning all of the model parameters.

As dataset, we use sentence-transformers/msmarco-bm25, which has (query, positive, negative) triplets with BM25-mined
hard negatives.

As loss function, we use MultiVectorMultipleNegativesRankingLoss, which uses the in-batch negatives with MaxSim scoring.
This recipe is identical to ../msmarco/training_contrastive.py, except that a LoRA adapter is added to the Transformer
backbone. Only the LoRA matrices and the fresh projection layer are trained, which considerably reduces the gradient
and optimizer memory.

Reference results (full 13-dataset NanoBEIR mean nDCG@10, single RTX 3090):
- this adapter recipe: 0.4609, uploaded as https://huggingface.co/tomaarsen/multivector-ModernBERT-base-msmarco-peft
- the full finetune from ../msmarco/training_contrastive.py on the same data: 0.4831
- without the query expansion line the two tie (adapter 0.4601, full finetune 0.4581, adapter model at
  https://huggingface.co/tomaarsen/multivector-ModernBERT-base-msmarco-peft-no-query-expansion), so the
  expansion benefit largely does not transfer through the adapter
"""

import logging
import traceback

from datasets import load_dataset
from peft import LoraConfig, TaskType

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderModelCardData,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.base.modules import Dense, Normalize, Transformer
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
for logger_name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def main():
    model_name = "answerdotai/ModernBERT-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    train_batch_size = 16
    num_epochs = 1
    learning_rate = 3e-5

    # 1a. Build the model to finetune with 1b. (Optional) model card data. Queries are padded to at
    # least 32 tokens with [MASK] expansion tokens acting as learned soft query terms (see the
    # reference results above for the gain). Loading in fp32 is preferred for training if your
    # memory can handle it, bf16=True below handles the autocast.
    transformer = Transformer(
        model_name,
        model_kwargs={"torch_dtype": "float32"},
        query_expansion={"strategy": "min", "length": 32},
    )
    linear = Dense(
        transformer.get_embedding_dimension(),
        128,
        bias=False,
        activation_function=None,
        module_input_name="token_embeddings",
    )
    mask = MultiVectorMask()
    normalize = Normalize(module_input_name="token_embeddings")
    model = MultiVectorEncoder(
        modules=[transformer, linear, mask, normalize],
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ColBERT {short_model_name} adapter trained on MS MARCO triplets",
        ),
    )

    # 1c. Create a LoRA adapter for the Transformer backbone. PEFT has no default target modules for
    # ModernBERT, so we list its attention and MLP linear layers explicitly. The randomly initialized
    # projection sits outside the backbone and remains fully trainable.
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        target_modules=["Wqkv", "Wo", "Wi"],
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
    )
    model.add_adapter(peft_config)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    logging.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})"
    )

    # 2. Load the MS MARCO triplets dataset: https://huggingface.co/datasets/sentence-transformers/msmarco-bm25
    full_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train").select(range(51_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss: in-batch negatives scored with MaxSim
    loss = MultiVectorMultipleNegativesRankingLoss(model=model)

    # 4. Define the evaluator. We use the MultiVectorNanoBEIREvaluator, which is a light-weight evaluator for English
    evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=train_batch_size)
    # Run the base model through the evaluator first to get a baseline before training.
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"multivector-{short_model_name}-msmarco-peft"
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
