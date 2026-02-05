"""
This scripts demonstrates how to train a Sparse Encoder model for Information Retrieval.

As dataset, we use Quora Duplicates Questions, where we have pairs of duplicate questions.

As loss function, we use MultipleNegativesRankingLoss in the SpladeLoss. Here, we only need positive pairs, i.e., pairs of sentences/texts that are considered to be relevant. Our dataset looks like this (a_1, b_1), (a_2, b_2), ... with a_i / b_i a text and (a_i, b_i) are relevant (e.g. are duplicates).

MultipleNegativesRankingLoss takes a random subset of these, for example (a_1, b_1), ..., (a_n, b_n). a_i and b_i are considered to be relevant and should be close in vector space. All other b_j (for i != j) are negative examples and the distance between a_i and b_j should be maximized. Note: MultipleNegativesRankingLoss only works if a random b_j is likely not to be relevant for a_i. This is the case for our duplicate questions dataset: If a sample randomly b_j, it is unlikely to be a duplicate of a_i.

"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder import evaluation, losses

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def main():
    model_name = "distilbert/distilbert-base-uncased"

    train_batch_size = 12
    num_epochs = 1

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    model = SparseEncoder(
        model_name,
        model_card_data=SparseEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="splade-distilbert-base-uncased trained on Quora Duplicates Questions",
        ),
    )
    model.max_seq_length = 256  # Set the max sequence length to 256 for the training
    logging.info("Model max length: %s", model.max_seq_length)

    # 2a. Load the Quora Duplicate Questions dataset:  https://huggingface.co/datasets/sentence-transformers/quora-duplicates
    logging.info("Read the Quora Duplicate Questions training dataset")
    full_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train").select(
        range(100000)
    )
    dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 3. Define our training loss.
    query_regularizer_weight = 5e-5
    document_regularizer_weight = 3e-5

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=query_regularizer_weight,  # Weight for query loss
        document_regularizer_weight=document_regularizer_weight,  # Weight for document loss
    )

    # 4. Define an evaluators. We add 2 evaluators, that evaluate the model on Duplicate Questions pair classification,
    # and NanoBeir with Duplicate Questions Information Retrieval
    evaluators = []

    duplicate_classes_dataset = load_dataset(
        "sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]"
    )
    binary_acc_evaluator = evaluation.SparseBinaryClassificationEvaluator(
        sentences1=duplicate_classes_dataset["sentence1"],
        sentences2=duplicate_classes_dataset["sentence2"],
        labels=duplicate_classes_dataset["label"],
        name="quora_duplicates_dev",
        show_progress_bar=True,
        similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
    )
    evaluators.append(binary_acc_evaluator)

    nano_beir_evaluator = evaluation.SparseNanoBEIREvaluator(
        dataset_names=["msmarco", "nq", "nfcorpus", "quoraretrieval"],
        show_progress_bar=True,
        batch_size=train_batch_size,
    )
    evaluators.append(nano_beir_evaluator)

    # Create a SequentialEvaluator. This SequentialEvaluator runs the 2 evaluators in a sequential order.
    # We optimize the model with respect to the score from the last evaluator (scores[-1])
    seq_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    # 5. Define the training arguments
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"splade-{short_model_name}-quora-duplicates"
    training_args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_dot_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=1650,
        save_strategy="steps",
        save_steps=1650,
        save_total_limit=2,
        logging_steps=200,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=42,
    )

    # 6. Create the trainer & start training
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=seq_evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, using the complete NanoBEIR dataset
    test_evaluator = evaluation.SparseNanoBEIREvaluator(show_progress_bar=True, batch_size=train_batch_size)
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
            f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
            f"`huggingface-cli login`, followed by loading the model using `model = SparseEncoder({final_output_dir!r})` "
            f"and saving it using `model.push_to_hub('{run_name}')`."
        )


if __name__ == "__main__":
    main()
