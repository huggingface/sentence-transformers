"""
This example shows how to train a bi-encoder on the MS MARCO Passage Ranking dataset
(https://github.com/microsoft/MSMARCO-Passage-Ranking) using MultipleNegativesRankingLoss.

Queries and passages are independently encoded into fixed-size embeddings, then compared
by cosine or dot similarity. We train with triplets ``(query, positive, negative_1, ...)``
where the negatives are hard negatives mined by different retrieval systems and filtered
by a CrossEncoder score margin to remove false negatives.

The hard negatives come from the 13 mining datasets in the
`MS MARCO Mined Triplets collection <https://huggingface.co/collections/sentence-transformers/ms-marco-mined-triplets>`_.
For each query, we union the top N negatives across all mining systems (deduplicated and
filtered against a teacher CrossEncoder score), then sample ``num_negatives`` of them
fresh per batch via ``dataset.set_transform`` so the text lookup stays out of memory
until a batch is actually consumed.

Running this script:
python train_bi_encoder_mnrl.py
"""

import logging
import random

import numpy
import torch
import tqdm
from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.modules import Normalize

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
random.seed(12)
torch.manual_seed(12)
numpy.random.seed(12)

# Hard-negative mining datasets on the Hub. Each has a ``triplet-50-ids`` subset with
# columns (query, positive, negative_1, ..., negative_50), all integer IDs that look up
# into sentence-transformers/msmarco-corpus.
SYSTEMS = {
    "bm25": "sentence-transformers/msmarco-bm25",
    "msmarco-distilbert-base-tas-b": "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-base-v3": "sentence-transformers/msmarco-msmarco-distilbert-base-v3",
    "msmarco-MiniLM-L-6-v3": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
    "distilbert-margin_mse-cls-dot-v2": "sentence-transformers/msmarco-distilbert-margin-mse-cls-dot-v2",
    "distilbert-margin_mse-cls-dot-v1": "sentence-transformers/msmarco-distilbert-margin-mse-cls-dot-v1",
    "distilbert-margin_mse-mean-dot-v1": "sentence-transformers/msmarco-distilbert-margin-mse-mean-dot-v1",
    "mpnet-margin_mse-mean-v1": "sentence-transformers/msmarco-mpnet-margin-mse-mean-v1",
    "co-condenser-margin_mse-cls-v1": "sentence-transformers/msmarco-co-condenser-margin-mse-cls-v1",
    "distilbert-margin_mse-mnrl-mean-v1": "sentence-transformers/msmarco-distilbert-margin-mse-mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v1": "sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v1",
    "distilbert-margin_mse-sym_mnrl-mean-v2": "sentence-transformers/msmarco-distilbert-margin-mse-sym-mnrl-mean-v2",
    "co-condenser-margin_mse-sym_mnrl-mean-v1": "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
}


def main():
    model_name = "distilbert/distilbert-base-uncased"
    train_batch_size = 256  # In-batch negatives are the dominant signal in MNRL; larger batches help quality.
    train_mini_batch_size = 32  # This controls the memory usage
    max_seq_length = 300

    num_negs_per_system = 5  # How many negatives to take from each mining system per query
    num_negatives = 4  # Hard negatives sampled per query per batch (in addition to ~batch_size in-batch negatives)
    ce_score_margin = 3.0  # CE-score margin between positive and negative to consider a negative valid

    # 1. Load a model to finetune with 2. (Optional) model card data. Weights stay in fp32
    # so the optimizer accumulates updates at full precision. `bf16=True` in TrainingArguments
    # below adds bf16 autocast on the forward/backward.
    short_model_name = model_name.split("/")[-1]
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{short_model_name} trained on MSMARCO with CachedMultipleNegativesRankingLoss",
        ),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"model_max_length": max_seq_length},
    )
    # Append a Normalize module so the encoder emits unit-length vectors. It's what most
    # users want at inference time for retrieval, and avoids surprises if someone later
    # swaps this script's loss for one that uses raw dot product.
    model.add_module("normalize", Normalize())

    # 3. Load MS MARCO corpus and queries. These give PID -> text and QID -> text lookups.
    logging.info("Loading MS MARCO corpus and queries")
    corpus = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
    corpus_dict = dict(zip(corpus["pid"], corpus["text"]))
    queries = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
    query_dict = dict(zip(queries["qid"], queries["text"]))

    # Load CrossEncoder scores for (query, passage) pairs so we can filter out false negatives.
    scores = load_dataset("sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2", "list", split="train")
    ce_scores = {
        qid: dict(zip(cids, sc)) for qid, cids, sc in zip(scores["query_id"], scores["corpus_id"], scores["score"])
    }

    # 4. Aggregate hard negatives across mining systems. For each query we collect the
    # union of negatives from all systems, deduplicated, with any negative whose CE score
    # is above (positive_score - ce_score_margin) discarded as a likely false negative.
    train_data = {}
    for system_key, repo_id in SYSTEMS.items():
        logging.info(f"Loading hard negatives from {system_key}")
        dataset = load_dataset(repo_id, "triplet-50-ids", split="train")

        for row in tqdm.tqdm(dataset, desc=f"Processing {system_key}"):
            qid = row["query"]
            pos_pid = row["positive"]
            if qid not in ce_scores or pos_pid not in ce_scores[qid]:
                continue
            pos_ce_score = ce_scores[qid][pos_pid]

            entry = train_data.setdefault(qid, {"qid": qid, "pid": pos_pid, "neg_pids": set()})
            added = 0
            for i in range(1, 51):
                neg_pid = row[f"negative_{i}"]
                if neg_pid in entry["neg_pids"] or neg_pid not in ce_scores[qid]:
                    continue
                if ce_scores[qid][neg_pid] >= pos_ce_score - ce_score_margin:
                    continue
                entry["neg_pids"].add(neg_pid)
                added += 1
                if added >= num_negs_per_system:
                    break

    # Keep only queries that have enough hard negatives to sample ``num_negatives`` without replacement.
    train_data = {qid: data for qid, data in train_data.items() if len(data["neg_pids"]) >= num_negatives}
    logging.info(f"Kept {len(train_data)} queries with >= {num_negatives} negatives")

    train_dataset = Dataset.from_list(
        [{"qid": d["qid"], "pid": d["pid"], "neg_pids": list(d["neg_pids"])} for d in train_data.values()]
    )

    # The transform resolves IDs to text on-the-fly and samples fresh negatives per batch.
    def ids_to_text_transform(batch):
        sampled_neg_ids = [random.sample(neg_pids, num_negatives) for neg_pids in batch["neg_pids"]]
        return {
            "anchor": [query_dict[qid] for qid in batch["qid"]],
            "positive": [corpus_dict[pid] for pid in batch["pid"]],
            **{
                f"negative_{i + 1}": [corpus_dict[pid] for pid in per_sample]
                for i, per_sample in enumerate(zip(*sampled_neg_ids))
            },
        }

    train_dataset.set_transform(ids_to_text_transform)

    # 5. Define a loss function. CachedMultipleNegativesRankingLoss lets us use large batches
    # by re-encoding mini-batches at backprop time, keeping peak memory bounded.
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=train_mini_batch_size)

    # 6. (Optional) Specify training arguments
    run_name = f"{short_model_name}-msmarco-mnrl"
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=3.2e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MNRL benefits from no duplicates in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        run_name=run_name,  # Will be used in experiment tracking
    )

    # 7. (Optional) Create an evaluator & evaluate the base model.
    dev_evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=train_batch_size)
    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        dev_evaluator(model)

    # 8. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
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
