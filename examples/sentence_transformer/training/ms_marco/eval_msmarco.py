"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performance metrics for cosine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

import logging
import sys

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Name of the SBERT model
model_name = sys.argv[1]

# You can limit the approx. max size of the corpus. Pass 100 as second parameter and the corpus has a size of approx 100k docs
corpus_max_size = int(sys.argv[2]) * 1000 if len(sys.argv) >= 3 else 0


#  Load model

model = SentenceTransformer(model_name)

# Read the MS MARCO dataset

corpus = {}  # Our corpus pid => passage
dev_queries = {}  # Our dev queries. qid => query
dev_rel_docs = {}  # Mapping qid => set with relevant pids
needed_pids = set()  # Passage IDs we need
needed_qids = set()  # Query IDs we need

# Load the 6980 dev queries
for row in load_dataset("mteb/msmarco", "default", split="dev"):
    qid, pid = str(row["query-id"]), str(row["corpus-id"])
    dev_rel_docs.setdefault(qid, set()).add(pid)
    needed_qids.add(qid)
    needed_pids.add(pid)

# Load the dev query texts
for row in load_dataset("mteb/msmarco", "queries", split="queries"):
    qid = str(row["_id"])
    if qid in needed_qids:
        dev_queries[qid] = row["text"].strip()

# Read passages
for row in load_dataset("sentence-transformers/msmarco", "corpus", split="train"):
    pid = str(row["passage_id"])
    if pid in needed_pids or corpus_max_size <= 0 or len(corpus) <= corpus_max_size:
        corpus[pid] = row["passage"].strip()


# Run evaluator
logging.info(f"Queries: {len(dev_queries)}")
logging.info(f"Corpus: {len(corpus)}")

ir_evaluator = InformationRetrievalEvaluator(
    dev_queries,
    corpus,
    dev_rel_docs,
    show_progress_bar=True,
    corpus_chunk_size=100000,
    precision_recall_at_k=[10, 100],
    name="msmarco dev",
)

ir_evaluator(model)
