"""
This script downloads the WikiMatrix corpus (https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix)
 and create parallel sentences tsv files that can be used to extend existent sentence embedding models to new languages.

The WikiMatrix mined parallel sentences from Wikipedia in various languages.

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://huggingface.co/papers/2004.09813
"""

import gzip
import os

import sentence_transformers.util

source_languages = set(["en"])  # Languages our (monolingual) teacher model understands
target_languages = set(["de", "es", "it", "fr", "ar", "tr"])  # New languages we want to extend to


num_dev_sentences = 1000  # Number of sentences we want to use for development
threshold = 1.075  # Only use sentences with a LASER similarity score above the threshold

download_url = "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/"
download_folder = "../datasets/WikiMatrix/"
parallel_sentences_folder = "parallel-sentences/"


os.makedirs(os.path.dirname(download_folder), exist_ok=True)
os.makedirs(parallel_sentences_folder, exist_ok=True)


for source_lang in source_languages:
    for target_lang in target_languages:
        filename_train = os.path.join(
            parallel_sentences_folder, f"WikiMatrix-{source_lang}-{target_lang}-train.tsv.gz"
        )
        filename_dev = os.path.join(parallel_sentences_folder, f"WikiMatrix-{source_lang}-{target_lang}-dev.tsv.gz")

        if not os.path.exists(filename_train) and not os.path.exists(filename_dev):
            langs_ordered = sorted([source_lang, target_lang])
            wikimatrix_filename = "WikiMatrix.{}-{}.tsv.gz".format(*langs_ordered)
            wikimatrix_filepath = os.path.join(download_folder, wikimatrix_filename)

            if not os.path.exists(wikimatrix_filepath):
                print("Download", download_url + wikimatrix_filename)
                try:
                    sentence_transformers.util.http_get(download_url + wikimatrix_filename, wikimatrix_filepath)
                except Exception:
                    print("Was not able to download", download_url + wikimatrix_filename)
                    continue

            if not os.path.exists(wikimatrix_filepath):
                continue

            train_sentences = []
            dev_sentences = []
            dev_sentences_set = set()
            extract_dev_sentences = True

            with gzip.open(wikimatrix_filepath, "rt", encoding="utf8") as fIn:
                for line in fIn:
                    score, sent1, sent2 = line.strip().split("\t")
                    sent1 = sent1.strip()
                    sent2 = sent2.strip()
                    score = float(score)

                    if score < threshold:
                        break

                    if sent1 == sent2:
                        continue

                    if langs_ordered.index(source_lang) == 1:  # Swap, so that src lang is sent1
                        sent1, sent2 = sent2, sent1

                    # Avoid duplicates in development set
                    if sent1 in dev_sentences_set or sent2 in dev_sentences_set:
                        continue

                    if extract_dev_sentences:
                        dev_sentences.append([sent1, sent2])
                        dev_sentences_set.add(sent1)
                        dev_sentences_set.add(sent2)

                        if len(dev_sentences) >= num_dev_sentences:
                            extract_dev_sentences = False
                    else:
                        train_sentences.append([sent1, sent2])

            print("Write", len(dev_sentences), "dev sentences", filename_dev)
            with gzip.open(filename_dev, "wt", encoding="utf8") as fOut:
                for sents in dev_sentences:
                    fOut.write("\t".join(sents))
                    fOut.write("\n")

            print("Write", len(train_sentences), "train sentences", filename_train)
            with gzip.open(filename_train, "wt", encoding="utf8") as fOut:
                for sents in train_sentences:
                    fOut.write("\t".join(sents))
                    fOut.write("\n")


print("---DONE---")
