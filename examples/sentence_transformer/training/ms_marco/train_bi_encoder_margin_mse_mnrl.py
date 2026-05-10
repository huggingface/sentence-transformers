"""
This example shows how to train a bi-encoder on the MS MARCO Passage Ranking dataset
(https://github.com/microsoft/MSMARCO-Passage-Ranking) with the combined
``MarginMSELoss + MultipleNegativesRankingLoss`` recipe used by the canonical
``sentence-transformers/msmarco-distilbert-margin-mse-mnrl-mean-v1`` family of models.

MarginMSELoss distills CrossEncoder score margins (teacher supervision) but has no
in-batch contrastive term. MultipleNegativesRankingLoss adds the in-batch contrast
that gives MNRL its strength. Combining both gets distillation + contrastive signal
in one training pass; we share the forward pass across both loss components by
computing embeddings once and calling each loss's ``compute_loss_from_embeddings``,
so memory stays the same as a single loss.

Both losses score with raw dot product to match the unnormalized embedding setup:
MNRL gets ``similarity_fct=dot_score`` and ``scale=1.0`` (the default ``cos_sim``
with ``scale=20`` would re-introduce magnitude normalization through the back door).

Hard negatives are mined from the 13 datasets in the
`MS MARCO Mined Triplets collection <https://huggingface.co/collections/sentence-transformers/ms-marco-mined-triplets-6644d6f1ff58c5103fe65f23>`_
and unioned per query, then ``dataset.set_transform`` samples ``num_negatives`` of them
fresh every batch, deferring the ID -> text lookup until a batch is actually consumed.

Running this script:
python train_bi_encoder_margin_mse_mnrl.py
"""

import logging
import random

import numpy
import torch
import tqdm
from datasets import Dataset, load_dataset
from torch import nn

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import MarginMSELoss, MultipleNegativesRankingLoss
from sentence_transformers.util import dot_score

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
    train_batch_size = 32  # The MNRL component benefits from larger batches, but also increases memory usage
    max_seq_length = 300

    num_negs_per_system = 5  # How many negatives to take from each mining system per query
    num_negatives = 10  # Negatives sampled per query per batch; each contributes one MSE term to MarginMSE.

    # 1. Load a model to finetune with 2. (Optional) model card data. Weights stay in fp32
    # so the optimizer accumulates updates at full precision; `bf16=True` in TrainingArguments
    # below adds bf16 autocast on the forward/backward.
    short_model_name = model_name.split("/")[-1]
    model = SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"{short_model_name} finetuned on MSMARCO with MarginMSELoss + MultipleNegativesRankingLoss",
        ),
        model_kwargs={"torch_dtype": torch.float32},
        processor_kwargs={"model_max_length": max_seq_length},
        similarity_fn_name="dot",  # MarginMSELoss + MNRL with dot product to match the unnormalized teacher scores
    )

    # 3. Load MS MARCO corpus and queries. These give PID -> text and QID -> text lookups.
    logging.info("Loading MS MARCO corpus and queries")
    corpus = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train")
    corpus_dict = dict(zip(corpus["pid"], corpus["text"]))
    queries = load_dataset("sentence-transformers/msmarco-corpus", "query", split="train")
    query_dict = dict(zip(queries["qid"], queries["text"]))

    # Load CrossEncoder scores; these serve as the teacher labels for MarginMSELoss.
    scores = load_dataset("sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2", "list", split="train")
    ce_scores = {
        qid: dict(zip(cids, sc)) for qid, cids, sc in zip(scores["query_id"], scores["corpus_id"], scores["score"])
    }

    # 4. Aggregate hard negatives across mining systems, keeping the per-(neg) margin label
    # (pos_ce - neg_ce) for MarginMSELoss. We dedupe across systems via a per-query dict
    # ``pid -> margin label``.
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

            entry = train_data.setdefault(qid, {"qid": qid, "pid": pos_pid, "neg_labels_by_pid": {}})
            added = 0
            for i in range(1, 51):
                neg_pid = row[f"negative_{i}"]
                if neg_pid in entry["neg_labels_by_pid"] or neg_pid not in ce_scores[qid]:
                    continue
                entry["neg_labels_by_pid"][neg_pid] = pos_ce_score - ce_scores[qid][neg_pid]
                added += 1
                if added >= num_negs_per_system:
                    break

    train_data = {qid: data for qid, data in train_data.items() if len(data["neg_labels_by_pid"]) >= num_negatives}
    logging.info(f"Kept {len(train_data)} queries with >= {num_negatives} negatives")

    train_dataset = Dataset.from_list(
        [
            {
                "qid": d["qid"],
                "pid": d["pid"],
                "neg_pids": list(d["neg_labels_by_pid"].keys()),
                "neg_labels": list(d["neg_labels_by_pid"].values()),
            }
            for d in train_data.values()
        ]
    )

    # The transform resolves IDs to text on-the-fly and samples fresh (negative, label) pairs
    # per batch, keeping only the ID/score mapping in memory until a batch is consumed.
    def ids_to_text_transform(batch):
        sampled = [
            random.sample(list(zip(neg_pids, neg_labels)), num_negatives)
            for neg_pids, neg_labels in zip(batch["neg_pids"], batch["neg_labels"])
        ]
        neg_pid_lists, label_lists = zip(*[zip(*s) for s in sampled])
        return {
            "anchor": [query_dict[qid] for qid in batch["qid"]],
            "positive": [corpus_dict[pid] for pid in batch["pid"]],
            **{
                f"negative_{i + 1}": [corpus_dict[pid] for pid in per_sample]
                for i, per_sample in enumerate(zip(*neg_pid_lists))
            },
            "label": [list(labels) for labels in label_lists],
        }

    train_dataset.set_transform(ids_to_text_transform)

    # 5. Define the loss. MNRL uses dot_score with scale=1.0 to match the unnormalized
    # embedding regime that MarginMSE trains in (the default cos_sim + scale=20 would
    # re-introduce magnitude normalization through the back door). We share the forward
    # pass across both losses so peak memory stays the same as a single loss.
    class CombinedMarginMSEMNRLLoss(nn.Module):
        def __init__(self, model, mnrl_weight: float = 1.0):
            super().__init__()
            self.model = model
            self.margin_mse = MarginMSELoss(model)
            self.mnrl = MultipleNegativesRankingLoss(model, scale=1.0, similarity_fct=dot_score)
            self.mnrl_weight = mnrl_weight

        def forward(self, sentence_features, labels):
            embeddings = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
            # Returning a dict lets the trainer log each component separately while training with the sum
            return {
                "margin_mse": self.margin_mse.compute_loss_from_embeddings(embeddings, labels),
                "mnrl": self.mnrl_weight * self.mnrl.compute_loss_from_embeddings(embeddings, labels),
            }

    loss = CombinedMarginMSEMNRLLoss(model)

    # 6. (Optional) Specify training arguments
    run_name = f"{short_model_name}-msmarco-margin-mse-mnrl"
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"models/{run_name}",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        save_total_limit=2,
        logging_steps=0.01,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
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
