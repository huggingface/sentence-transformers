import argparse
import gzip
import json
import logging
import os
import pickle
import tarfile
from datetime import datetime
from typing import List, Dict, Union, Optional
from dataclasses import dataclass
import tqdm

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    LoggingHandler,
    models,
    evaluation,
    losses,
    training_args,
    util)

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import (AutoTokenizer,
                          EarlyStoppingCallback)
from datasets import Dataset


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used? Multiple systems separated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--hard_neg_count_per_query", default=10, type=int)
parser.add_argument("--use_curated_negatives", default=True, action="store_true")

args = parser.parse_args()

print(args)


model_name = args.model_name
use_curated_negatives = args.use_curated_negatives

train_batch_size = (
    args.train_batch_size
)  # Increasing the train batch size improves the model performance, but requires more GPU memory
ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = (
    args.num_negs_per_system
)  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    #setting max sequence length can slow down training, used dynamic padding in the collator,
    #see https://huggingface.co/docs/transformers/v4.34.0/en/pad_truncation
    #visit FlatMNRLCollator data collator to see how I implemented the dynamic padding
    #model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = "output/train_bi-encoder-mnrl-{}-margin_{:.1f}-{}".format(
    model_name.replace("/", "-"), ce_score_margin, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


### Now we read the MS Marco dataset
data_folder = "msmarco-data"

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, "collection.tsv")
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, "collection.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, encoding="utf8") as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L6-v2 model
ce_scores_file = os.path.join(data_folder, "cross-encoder-ms-marco-MiniLM-L6-v2-scores.pkl.gz")
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get(
        "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz",
        ce_scores_file,
    )

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, "rb") as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get(
        "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
        hard_negatives_filepath,
    )


logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        # Get the positive passage ids
        qid = data["qid"]
        pos_pids = data["pos"]

        if len(pos_pids) == 0:  # Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data["pos"]])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        # Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:  # Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:  # Use all systems
                negs_to_use = list(data["neg"].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data["neg"]:
                continue

            system_negs = data["neg"][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data["qid"]] = {
                "qid": data["qid"],
                "query": queries[data["qid"]],
                "pos": pos_pids,
                "neg": neg_pids,
            }

del ce_scores

logging.info(f"Train queries: {len(train_queries)}")


## train eval split before preparing hf datasets
# bc we will provide extra negatives for MNRL,
# and it is unrequired to add predefined negatives to anchors during eval.
from sklearn.model_selection import train_test_split

def train_test_split_dict(data_dict, test_size=0.3, seed=42):
    """
    data_dict: {qid: {"query": str, "pos": [...], "neg": [...]}, ...}
    """
    items = list(data_dict.items())
    train_items, eval_items = train_test_split(
        items, test_size=test_size, random_state=seed
    )
    train_dict_ = dict(train_items)
    eval_dict_ = dict(eval_items)
    return train_dict_, eval_dict_

train_dict, eval_dict = train_test_split_dict(train_queries)

logging.info(f"Total query length: {len(train_queries)}")
logging.info(f"Train split       : {len(train_dict)}")
logging.info(f"Eval split        : {len(eval_dict)}")

## Prepare hf datasets: train and eval

def build_mnrl_dataset_with_curated_negatives(data_dict, corpus, negative_count=5):
    """
    Prepares rectangular dataset with p,q,n columns for MNRL loss
    This function will be used to prepare train dataset only if the use_curated_negatives was set to True
    it would be still okay if we provide, q and p for MNRL using build_mnrl_pair_dataset though
    but since we have predefined negatives we will create training hf dataset using this one
    Don't forget to use a batch sampler though, because rectangular dataset format
    may cause batch to be exhausted with same anchor positive pairs all the time
    :return Hf.Dataset
    """
    rows = []
    for _, rec in data_dict.items():
        qid = rec["qid"]
        q = rec["query"]
        neg_ids = list(rec.get("neg", []))
        pos_ids = list(rec.get("pos", []))
        for pid in pos_ids:
            if pid in corpus:
                for nid in neg_ids[:negative_count]:
                  if nid in corpus:
                    rows.append({"q": q,
                                "p": corpus[pid],
                                "n": corpus[nid],
                                "qid": qid,
                                "pid_pos": pid,
                                "nid_pos": nid})
    return Dataset.from_list(rows)



def build_mnrl_pair_dataset(data_dict, corpus):
    """
    Prepares rectangular dataset with p,q columns for MNRL loss
    This function will be used to prepare train dataset if use_curated_negatives is set to False
    This function will be used for MNRL loss EVAL
    Also you can use this function if you don't want to explicitly add curated negatives

    :return Hf.Dataset

    """
    rows = []
    for _, rec in data_dict.items():
        qid = rec["qid"]
        q = rec["query"]
        pos_ids = list(rec.get("pos", []))
        for pid in pos_ids:
            if pid in corpus:
                  rows.append({"q": q,
                               "p": corpus[pid],
                               "qid": qid,
                               "pid_pos": pid})
    return Dataset.from_list(rows)

if use_curated_negatives:
    # if you use predefined or on the fly curated negatives in training
    negative_count = args.hard_neg_count_per_query

    hf_dataset_train = build_mnrl_dataset_with_curated_negatives(train_dict,
                                            corpus=corpus,
                                            negative_count=negative_count)
else:
    hf_dataset_train = build_mnrl_pair_dataset(train_dict,
                                               corpus=corpus)


hf_dataset_eval = build_mnrl_pair_dataset(eval_dict,
                                          corpus=corpus)

# Prep for eval data
eval_corpus={}
eval_dev_queries={}
eval_rel_docs={}

eval_dataset = hf_dataset_eval.shuffle(seed=42).select(range(10000))

for x in eval_dataset:
    pid = str(x["pid_pos"])
    qid = str(x["qid"])
    if pid not in eval_corpus:
        eval_corpus[pid] = x["p"]

    if qid not in eval_dev_queries:
        eval_dev_queries[qid] = x["q"]

    if qid not in eval_rel_docs:
        eval_rel_docs[qid] = set()
    eval_rel_docs[qid].add(pid)


ir_evaluator = evaluation.InformationRetrievalEvaluator(
    queries=eval_dev_queries,
    corpus=eval_corpus,
    relevant_docs =eval_rel_docs,
    show_progress_bar=False,
    corpus_chunk_size=1000, # avoid oom, move that amount of chunk each time, to cpu
    precision_recall_at_k=[1,5,10],
    name="ms_marco",
)


@dataclass
class FlatMNRLCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    valid_label_columns: set = None

    def __post_init__(self):
        if self.valid_label_columns is None:
            self.valid_label_columns = set()

    def __call__(self, features: List[Dict]) -> Dict[str, any]:
        # Prepares a rectangular dataset with columns
        # sample["q"] : str
        # sample["p"]   : str
        # sample["qid"]   : int, index of the query
        # sample["pid"] :int, index of the positive
        # and these if negatives in dataset
        # sample["nid"] :int, index of the negative

        anchors, positives, negatives = [], [], []
        # train and eval use this collator together,
        # but in eval dataset there won't be negatives
        # that is where add_curated_negatives_in_batch comes into play

        add_curated_negatives_in_batch = True if "n" in features[0] else False

        for sample in features:
            query = sample["q"]
            pos_text = sample["p"]

            anchors.append(query)
            positives.append(pos_text)

            if add_curated_negatives_in_batch:
              neg_text = sample["n"]
              negatives.append(neg_text)

        # dynamic padding applied here,
        # details:  https://huggingface.co/docs/transformers/v4.34.0/en/pad_truncation

        fq = self.tokenizer(anchors, padding=self.padding, truncation=True,
                            max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of,
                            return_tensors=self.return_tensors)
        fp = self.tokenizer(positives, padding=self.padding, truncation=True,
                            max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of,
                            return_tensors=self.return_tensors)


        # sentence_0 -> query
        # sentence_1 -> positive
        # sentence_2 -> negative

        # No labels for MNRL labels  ST Trainer collect_features will make labels=None
        out = {"sentence_0_input_ids": fq["input_ids"],
               "sentence_0_attention_mask": fq["attention_mask"],
               "sentence_1_input_ids": fp["input_ids"],
               "sentence_1_attention_mask": fp["attention_mask"],
               "labels":None}
        if add_curated_negatives_in_batch:
          fn = self.tokenizer(negatives, padding=self.padding, truncation=True,
                      max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of,
                      return_tensors=self.return_tensors)
          out.update({"sentence_2_input_ids": fn["input_ids"],
                      "sentence_2_attention_mask": fn["attention_mask"]})

        return out


# Callbacks

early_stop_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=5e-5,

)

callbacks = [early_stop_callback]

# 5. TrainingArguments and Trainer
logging_steps=100
straining_args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    num_train_epochs=args.epochs,
    logging_steps=logging_steps,
    save_steps=logging_steps,
    logging_dir=f"{model_save_path}/logs",
    report_to="none",
    save_strategy="steps",
    remove_unused_columns=False,
    eval_strategy="steps",
    eval_steps=logging_steps,
    metric_for_best_model="eval_ms_marco_cosine_accuracy@5",  # or "mrr_at_10"
    greater_is_better=True,
    batch_sampler=training_args.BatchSamplers.NO_DUPLICATES,
    gradient_accumulation_steps=4,
    fp16=True,
    load_best_model_at_end=True,
    eval_accumulation_steps=10,
    max_grad_norm=1.0 )


loss = losses.MultipleNegativesRankingLoss(model=model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = SentenceTransformerTrainer(
            model=model,
            args=straining_args,
            train_dataset=hf_dataset_train,
            eval_dataset=hf_dataset_eval,
            loss=loss,
            data_collator=FlatMNRLCollator(tokenizer),
            evaluator=ir_evaluator,
            callbacks=callbacks


)

trainer.train()