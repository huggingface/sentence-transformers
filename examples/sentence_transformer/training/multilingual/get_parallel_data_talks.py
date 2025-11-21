"""
This script downloads the parallel sentences corpus and create parallel sentences tsv files that can be used to extend
existent sentence embedding models to new languages.

The parallel sentences corpus is a crawl of transcripts from talks, which are translated to 100+ languages.

The parallel sentences corpus cannot be downloaded automatically. It is available for research purposes only (CC-BY-NC).

The training procedure can be found in the files make_multilingual.py.

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://huggingface.co/papers/2004.09813
"""

import csv
import gzip
import os

from tqdm.autonotebook import tqdm

import sentence_transformers.util

source_languages = set(["en"])  # Languages our (monolingual) teacher model understands
target_languages = set(["de", "es", "it", "fr", "ar", "tr"])  # New languages we want to extend to


dev_sentences = 1000  # Number of sentences we want to use for development
download_url = "https://sbert.net/datasets/parallel-sentences.tsv.gz"  # Specify parallel sentences URL here
parallel_sentences_path = "../datasets/parallel-sentences.tsv.gz"  # Path of the parallel-sentences.tsv.gz file.
parallel_sentences_folder = "parallel-sentences/"


os.makedirs(os.path.dirname(parallel_sentences_path), exist_ok=True)
if not os.path.exists(parallel_sentences_path):
    print("parallel-sentences.tsv.gz does not exists. Try to download from server")
    sentence_transformers.util.http_get(download_url, parallel_sentences_path)


os.makedirs(parallel_sentences_folder, exist_ok=True)
train_files = []
dev_files = []
files_to_create = []
for source_lang in source_languages:
    for target_lang in target_languages:
        output_filename_train = os.path.join(
            parallel_sentences_folder, f"talks-{source_lang}-{target_lang}-train.tsv.gz"
        )
        output_filename_dev = os.path.join(parallel_sentences_folder, f"talks-{source_lang}-{target_lang}-dev.tsv.gz")
        train_files.append(output_filename_train)
        dev_files.append(output_filename_dev)
        if not os.path.exists(output_filename_train) or not os.path.exists(output_filename_dev):
            files_to_create.append(
                {
                    "src_lang": source_lang,
                    "trg_lang": target_lang,
                    "fTrain": gzip.open(output_filename_train, "wt", encoding="utf8"),
                    "fDev": gzip.open(output_filename_dev, "wt", encoding="utf8"),
                    "devCount": 0,
                }
            )

if len(files_to_create) > 0:
    print(
        "Parallel sentences files {} do not exist. Create these files now".format(
            ", ".join(map(lambda x: x["src_lang"] + "-" + x["trg_lang"], files_to_create))
        )
    )
    with gzip.open(parallel_sentences_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line in tqdm(reader, desc="Sentences"):
            for outfile in files_to_create:
                src_text = line[outfile["src_lang"]].strip()
                trg_text = line[outfile["trg_lang"]].strip()

                if src_text != "" and trg_text != "":
                    if outfile["devCount"] < dev_sentences:
                        outfile["devCount"] += 1
                        fOut = outfile["fDev"]
                    else:
                        fOut = outfile["fTrain"]

                    fOut.write(f"{src_text}\t{trg_text}\n")

    for outfile in files_to_create:
        outfile["fTrain"].close()
        outfile["fDev"].close()


print("---DONE---")
