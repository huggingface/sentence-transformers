"""
This example trains a SparseEncoder for the Natural Questions (NQ) dataset.
The training script fine-tunes a SparseEncoder using the CSR loss function for retrieval.
It loads a subset of the Natural Questions dataset, splits it into training and evaluation subsets,
and trains the model as a retriever. After training, the model is evaluated and saved locally,
with an optional step to push the trained model to the Hugging Face Hub.

Usage:
python train_csr_nq.py
"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    util,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder import evaluation, losses

# Set up logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "mixedbread-ai/mxbai-embed-large-v1"

    train_batch_size = 64
    num_epochs = 1

    # 1a. Load a model to finetune with 1b. (Optional) model card data and the cosine similarity function,
    # as most pretrained dense embedding models are trained with cosine similarity.
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Sparse CSR model trained on Natural Questions",
        ),
        similarity_fn_name="cosine",
    )

    # Freeze the first module of the model, i.e. the encoder, & print the number of (trainable) parameters
    model[0].requires_grad_(False)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {num_params:,}, Trainable parameters: {num_trainable_params:,}, Trainable ratio: {num_trainable_params / num_params:.2%}"
    )

    # 2a. Load the NQ dataset: https://huggingface.co/datasets/sentence-transformers/natural-questions
    logging.info("Read the Natural Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss. We use the Cosine Similarity similarity as the pretrained model was also trained with it.
    # The scale of 20 is common for MultipleNegativesRankingLoss with cosine similarity.
    # The lower gamma gives higher weight to the high-sparsity performance, whereas a higher gamma would give more weight
    # to the low-sparsity performance.
    loss = losses.CSRLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model, scale=20.0, similarity_fct=util.cos_sim),
        gamma=0.1,
    )

    # 4. Define evaluator. This is useful to keep track of alongside the evaluation loss.
    evaluators = []
    for k_dim in [4, 8, 16, 32, 64, 128, 256]:
        queries = dict(enumerate(eval_dataset["query"]))
        corpus = dict(enumerate(eval_dataset["answer"]))
        relevant_docs = {index: [index] for index in range(len(eval_dataset["query"]))}
        evaluator = evaluation.SparseInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            max_active_dims=k_dim,
            batch_size=train_batch_size,
            name=f"nq_eval_{k_dim}",
        )
        evaluators.append(evaluator)
    dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    dev_evaluator(model)

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"csr-{short_model_name}-nq"
    training_args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=4e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
        logging_steps=100,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model again
    dev_evaluator(model)

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
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
