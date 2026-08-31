"""
This script demonstrates how to train a domain-specific ColBERT-style multi-vector model with contrastive
learning, using medical question -> passage retrieval on MIRIAD as the example domain.

As dataset, we use tomaarsen/miriad-4.4M-split, a train/eval/test split of MIRIAD: 4.4M medical questions
paired with the peer-reviewed paper passage they were generated from. Passages average ~940 tokens, so this
is also an example of long-document late interaction training.

As loss function, we use CachedMultiVectorMultipleNegativesRankingLoss, which treats the other in-batch
passages as negatives and uses GradCache to decouple the effective batch size (large, for strong in-batch
negatives) from the memory footprint (bounded by mini_batch_size). Two settings matter here:
- scale=1.0 (the default): MaxSim scores are unbounded sums, a bi-encoder-style scale would saturate them.
- BatchSamplers.NO_DUPLICATES: MIRIAD reuses passages across questions, and a duplicated in-batch passage
  would act as a false negative.

The model is built explicitly in the classic ColBERT configuration on a retrieval-pretrained backbone.
Tokenization is capped at 1024 tokens during training only, via the max_length training argument: the
saved model keeps the backbone's full 8192 context.

Reference results (nDCG@10 against the deduplicated passage corpus of each split, single RTX 3090,
640k training rows in about 9.5 hours):
- pre-training baseline (warm-started backbone, random projection): 0.9176
- full eval split (10k queries): 0.9827, full test split (10k queries): 0.9826,
  uploaded as https://huggingface.co/tomaarsen/multivector-gte-modernbert-base-miriad
The previous best on this protocol, splade-modernbert-base-miriad, scores 0.8609 / 0.8626.
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
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


def build_ir_evaluator(split, name: str, max_queries: int | None = None):
    """Corpus = deduplicated passages of the split, queries = questions, 1:1 qrels."""
    passage_to_id = {}
    corpus = {}
    queries = {}
    relevant_docs = {}
    for idx, (question, passage) in enumerate(zip(split["question"], split["passage_text"])):
        if passage not in passage_to_id:
            passage_to_id[passage] = f"p{len(passage_to_id)}"
            corpus[passage_to_id[passage]] = passage
        queries[f"q{idx}"] = question
        relevant_docs[f"q{idx}"] = {passage_to_id[passage]}
    if max_queries is not None and len(queries) > max_queries:
        keep = list(queries)[:max_queries]
        queries = {qid: queries[qid] for qid in keep}
        relevant_docs = {qid: relevant_docs[qid] for qid in keep}
    return MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=8,
        name=name,
    )


def main():
    model_name = "Alibaba-NLP/gte-modernbert-base"
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]

    num_rows = 640_000  # About 9.5 hours on an RTX 3090, lower for a faster (but weaker) run
    # The effective batch size drives in-batch negative quality, while mini_batch_size bounds the
    # memory: GradCache makes the results identical regardless of mini_batch_size, so lower it for
    # smaller GPUs at only a wall-clock cost.
    train_batch_size = 128
    mini_batch_size = 8
    num_epochs = 1
    learning_rate = 3e-5

    # 1a. Build a model to finetune with 1b. (Optional) model card data
    # The module stack is built explicitly in the classic ColBERT configuration: queries are
    # mask-expanded to at least 32 tokens, punctuation tokens are excluded from MaxSim scoring, and
    # a single linear layer projects the backbone's token embeddings to 128 dimensions.
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
            model_name=f"ColBERT {short_model_name} trained on MIRIAD medical question-passage pairs",
        ),
    )
    print(model)

    # 2. Load the MIRIAD splits: https://huggingface.co/datasets/tomaarsen/miriad-4.4M-split
    dataset = load_dataset("tomaarsen/miriad-4.4M-split")
    train_dataset = dataset["train"].select(range(num_rows))
    logging.info(train_dataset)

    # 3. Define our training loss
    loss = CachedMultiVectorMultipleNegativesRankingLoss(model=model, mini_batch_size=mini_batch_size)

    # 4. Define the evaluator: a 1000-query subsample of the eval split against its full passage
    # corpus keeps the in-training evaluations affordable. The full splits are evaluated at the end.
    evaluator = build_ir_evaluator(dataset["eval"], "miriad_eval", max_queries=1000)
    # Run the base model through the evaluator first to get a baseline before training
    evaluator(model)

    # 5. Define the training arguments
    run_name = f"multivector-{short_model_name}-miriad"
    args = MultiVectorEncoderTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        learning_rate=learning_rate,
        max_length=1024,  # Cap training tokenization: passages average ~940 tokens, the model serves 8192
        warmup_steps=0.05,  # Warm up over the first 5% of training steps
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        load_best_model_at_end=True,
        metric_for_best_model="eval_miriad_eval_maxsim_ndcg@10",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",  # The evaluator runs on its own datasets, so no eval_dataset is needed
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.005,
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

    # 7. Evaluate the final model on the complete eval and test splits.
    full_eval = build_ir_evaluator(dataset["eval"], "miriad_eval_full")
    test_eval = build_ir_evaluator(dataset["test"], "miriad_test")
    full_eval(model)
    test_eval(model)

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
