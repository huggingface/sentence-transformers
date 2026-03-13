"""
This script translates the queries in the MS MARCO dataset to the defined target languages.

For machine translation, we use EasyNMT: https://github.com/UKPLab/EasyNMT
You can install it via: pip install easynmt

Usage:
python translate_queries [target_language]
"""

import logging
import os
import sys
import tarfile

from easynmt import EasyNMT

from sentence_transformers import LoggingHandler
from sentence_transformers.util import http_get

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

target_lang = sys.argv[1]
output_folder = "multilingual-data"
data_folder = "../msmarco-data"

output_filename = os.path.join(output_folder, f"train_queries.en-{target_lang}.tsv")
os.makedirs(output_folder, exist_ok=True)


## Does the output file exists? If yes, read it so we can continue the translation
translated_qids = set()
if os.path.exists(output_filename):
    with open(output_filename, encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            translated_qids.add(splits[0])

### Now we read the MS Marco dataset
os.makedirs(data_folder, exist_ok=True)

# Read qrels file for relevant positives per query
train_queries = {}
qrels_train = os.path.join(data_folder, "qrels.train.tsv")
if not os.path.exists(qrels_train):
    http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv", qrels_train)

with open(qrels_train) as fIn:
    for line in fIn:
        qid, _, pid, _ = line.strip().split()
        if qid not in translated_qids:
            train_queries[qid] = None

# Read all queries
queries_filepath = os.path.join(data_folder, "queries.train.tsv")
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, "queries.tar.gz")
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        http_get("https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)


with open(queries_filepath, encoding="utf8") as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        if qid in train_queries:
            train_queries[qid] = query.strip()


qids = [qid for qid in train_queries if train_queries[qid] is not None]
queries = [train_queries[qid] for qid in qids]

# Define our translation model
translation_model = EasyNMT("opus-mt")

print(f"Start translation of {len(queries)} queries.")
print("This can take a while. But you can stop this script at any point")


with open(output_filename, "a" if os.path.exists(output_filename) else "w", encoding="utf8") as fOut:
    for qid, query, translated_query in zip(
        qids,
        queries,
        translation_model.translate_stream(
            queries,
            source_lang="en",
            target_lang=target_lang,
            beam_size=2,
            perform_sentence_splitting=False,
            chunk_size=256,
            batch_size=64,
        ),
    ):
        fOut.write("{}\t{}\n".format(qid, translated_query.replace("\t", " ")))
        fOut.flush()
