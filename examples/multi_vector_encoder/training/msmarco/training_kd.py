"""
This script demonstrates how to train a strong ColBERT-style multi-vector model with knowledge distillation
on MS MARCO.

As dataset, we use lightonai/ms-marco-en-bge-gemma, which provides per-query lists of 32 candidate documents
(mined with ColBERT) together with teacher scores from the bge-reranker-v2-gemma cross-encoder. resolve_ids
resolves those IDs against the queries / documents datasets on the fly during training.

As loss function, we use MultiVectorDistillKLDivLoss, which minimizes the KL divergence between the
(softmaxed) teacher scores and the student's MaxSim scores over each query's candidate documents. The
dataset's teacher scores are min-max normalized per query, which makes the default softmax nearly flat over
32 candidates, so we sharpen with temperature=0.25. It applies to both sides, and measured better here than
the default of 1.0.

The model is built explicitly in the classic ColBERT configuration: a retrieval-pretrained backbone
(Alibaba-NLP/gte-modernbert-base), queries mask-expanded to at least 32 tokens (the "min" strategy: short
queries gain learned soft query terms, long queries pass through untruncated), a 128-dim projection, and a
punctuation skiplist. Tokenization is capped at 180 tokens during training only, via the max_length
training argument: the saved model keeps the backbone's full 8192 context, and long documents help at
retrieval time (+0.03 nDCG@10 here) while the tight cap only helps training.
A deeper residual bottleneck head (https://huggingface.co/papers/2510.12327) measured ~0.02 below this
single projection at these data scales, as did "query: " / "document: " prompts.

Reference results (NanoBEIR msmarco/nq/fiqa2018 mean nDCG@10, with the full 13-dataset mean in
parentheses, single RTX 3090):
- pre-training baseline, warm-started backbone with a random projection: 0.5943
- as-is, 20k training rows (about 1 hour): 0.6770 (full suite: 0.6537),
  uploaded as https://huggingface.co/tomaarsen/multivector-gte-modernbert-base-msmarco-kd
- num_rows = 200_000 (about 7.5 hours): 0.6879 (full suite: 0.6634)
For comparison on the same three datasets: colbert-ir/colbertv2.0 scores 0.6000,
answerdotai/answerai-colbert-small-v1 scores 0.6443, and lightonai/LateOn scores 0.6958.
"""

import logging
import string
import traceback

from datasets import load_dataset

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderModelCardData,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.base.modules import Dense, Normalize, Transformer
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
from sentence_transformers.util import resolve_ids

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def main():
    model_name = "Alibaba-NLP/gte-modernbert-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    num_rows = 20_000  # Increase to 200_000 for the strongest configuration (about 7.5 hours on an RTX 3090)
    # Each batch holds train_batch_size * 32 documents. This fits a 24 GB GPU (peak ~17 GB). On smaller
    # GPUs, lower train_batch_size and raise gradient_accumulation_steps to keep their product at 4:
    # 2 x 2 peaks at ~10 GB and 1 x 4 at ~6.5 GB, with identical results (this loss has no in-batch
    # negatives, so accumulation is exact).
    train_batch_size = 4
    gradient_accumulation_steps = 1
    num_epochs = 1
    learning_rate = 3e-5

    # 1a. Build a model to finetune with 1b. (Optional) model card data. Queries are padded to at
    # least 32 tokens with [MASK] expansion tokens acting as learned soft query terms, and
    # punctuation tokens are excluded from MaxSim scoring. Loading in fp32 is preferred for
    # training if your memory can handle it, bf16=True below handles the autocast.
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
    mask = MultiVectorMask(skiplist_words=list(string.punctuation))
    normalize = Normalize(module_input_name="token_embeddings")
    model = MultiVectorEncoder(
        modules=[transformer, linear, mask, normalize],
        model_card_data=MultiVectorEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ColBERT {short_model_name} distilled from bge-reranker-v2-gemma on MS MARCO",
        ),
    )
    # MultiVectorEncoder(model_name, ...) alone builds this same stack, minus the query expansion
    # and the punctuation skiplist (a domain choice: helps English prose, hurts e.g. code retrieval).
    print(model)

    # 2. Load the lightonai/ms-marco-en-bge-gemma dataset: https://huggingface.co/datasets/lightonai/ms-marco-en-bge-gemma
    # The `train` split holds query_id + document_ids + teacher scores. `queries` and `documents` hold the texts.
    train_dataset = load_dataset("lightonai/ms-marco-en-bge-gemma", "train", split="train").select(range(num_rows))
    queries = load_dataset("lightonai/ms-marco-en-bge-gemma", "queries", split="train")
    documents = load_dataset("lightonai/ms-marco-en-bge-gemma", "documents", split="train")

    # resolve_ids resolves query_id -> query text and document_ids -> document texts on the fly.
    # This dataset stores document_ids and scores as string-encoded lists, which resolve_ids parses.
    train_dataset.set_transform(resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=32))
    logging.info(train_dataset)

    # 3. Define our training loss: KL divergence between the teacher and student score distributions.
    # temperature=0.25 sharpens the per-query normalized teacher scores, at the default 1.0 the target
    # distribution over 32 candidates is nearly flat.
    loss = MultiVectorDistillKLDivLoss(model=model, temperature=0.25)

    # 4. Define the evaluator. We use the MultiVectorNanoBEIREvaluator, which is a light-weight evaluator for English
    evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=16)
    # Run the base model through the evaluator first to get a baseline before training
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"multivector-{short_model_name}-msmarco-kd"
    args = MultiVectorEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=learning_rate,
        max_length=180,  # Cap training tokenization at 180 tokens, the query width floor stays 32 via the expansion
        warmup_steps=0.05,  # Warm up over the first 5% of training steps
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        load_best_model_at_end=True,
        metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10",
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
    test_evaluator = MultiVectorNanoBEIREvaluator(show_progress_bar=True, batch_size=16)
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
