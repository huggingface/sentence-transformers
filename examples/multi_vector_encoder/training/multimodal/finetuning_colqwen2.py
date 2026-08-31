"""
This script demonstrates how to finetune a pretrained ColPali-style multi-vector (late-interaction) model for
visual document retrieval: text queries are matched directly against document page images, skipping OCR entirely.
Finetuning adapts an already-strong Col* checkpoint to your own document collection or domain.

As base model, we use vidore/colqwen2-v1.0-hf, a transformers-native ColQwen2 retriever that MultiVectorEncoder
auto-detects. Its 2.2B parameter backbone is too large for full finetuning on a 24GB consumer GPU, so we freeze
everything except the top layers of the language model and the multi-vector projection. Keeping the vision tower
frozen is standard ColVLM practice: the ViDoRe Col* checkpoints themselves were trained with a frozen vision
tower and LoRA adapters on the language model, and layer freezing is the adapter-free variant of that recipe.
Together with bf16 weights, training peaks at roughly 12GB of VRAM.

As dataset, we use vidore/syntheticDocQA_energy_train, one of the domain components of the original ColPali
training mix: scraped PDF page images from the energy domain paired with synthetic retrieval queries. After
keeping one query per unique page, each row is a (query, page image) pair. The trainer reads columns
positionally: the first column is encoded as the query and the remaining columns as documents, so image columns
need no special handling.

Because this domain is part of ColQwen2's own training mix, the measurable gain here is small (0.9571 ->
0.9592 nDCG@10 on the held-out pages in a reference run, uploaded as
https://huggingface.co/tomaarsen/multivector-colqwen2-v1.0-hf-docqa-energy): this script demonstrates the
finetuning mechanics. Swap in your own document collection, which the base model has never seen, for the
real adaptation gains.

As loss function, we use MultiVectorMultipleNegativesRankingLoss, which uses the in-batch negatives with MaxSim
scoring: image documents produce token embeddings from image patches, just like text documents do from text tokens.
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
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
for logger_name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)


def main():
    model_name = "vidore/colqwen2-v1.0-hf"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    train_batch_size = 8
    num_epochs = 1
    learning_rate = 2e-5
    num_trainable_layers = 8

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    # Loading in fp32 is preferred for training if your memory can handle it, but 2.2B parameters in fp32
    # plus training activations exceed 24GB, so we keep the checkpoint's native bf16
    model = MultiVectorEncoder(
        model_name,
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{short_model_name} finetuned on energy document pages",
        ),
        model_kwargs={"torch_dtype": "bfloat16"},
    )

    # 1c. Freeze all but the top language model layers and the multi-vector projection. Frozen bottom layers
    # need no activation storage or optimizer state, which is what makes this fit on a 24GB GPU.
    retrieval_model = model.transformers_model
    retrieval_model.requires_grad_(False)
    retrieval_model.vlm.language_model.layers[-num_trainable_layers:].requires_grad_(True)
    retrieval_model.embedding_proj_layer.requires_grad_(True)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    logging.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})"
    )

    # 2. Load the energy domain dataset: https://huggingface.co/datasets/vidore/syntheticDocQA_energy_train
    # It has up to 3 queries per page. Keep the first query of every unique page, so a batch never contains
    # another sample's positive page as an in-batch negative.
    dataset = load_dataset("vidore/syntheticDocQA_energy_train", split="train")
    first_indices = {}
    for idx, filename in enumerate(dataset["image_filename"]):
        first_indices.setdefault(filename, idx)
    dataset = dataset.select(list(first_indices.values())).select_columns(["query", "image"])
    dataset_dict = dataset.train_test_split(test_size=400, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss: in-batch negatives scored with MaxSim
    loss = MultiVectorMultipleNegativesRankingLoss(model=model)

    # 4. Define the evaluator: retrieve the matching page for every held-out query among all held-out pages
    queries = {f"query_{idx}": query for idx, query in enumerate(eval_dataset["query"])}
    corpus = {f"page_{idx}": image for idx, image in enumerate(eval_dataset["image"])}
    relevant_docs = {f"query_{idx}": {f"page_{idx}"} for idx in range(len(eval_dataset))}
    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="energy-dev",
        batch_size=train_batch_size,
    )
    # Run the base model through the evaluator first to get a baseline before training.
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"multivector-{short_model_name}-docqa-energy"
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
        metric_for_best_model="eval_energy-dev_maxsim_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        # Checkpoints of a 2.2B model carry optimizer states roughly the size of the model again.
        # This short run never needs resuming, so save weights only: load_best_model_at_end still works.
        save_only_model=True,
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

    # 7. Evaluate the final model again, now that the best checkpoint has been loaded
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
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, "
            f"you can run `huggingface-cli login`, followed by loading the model using "
            f"`model = MultiVectorEncoder({final_output_dir!r})`, then saving it using "
            f"`model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
