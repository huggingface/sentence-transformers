"""
Tatoeba (https://tatoeba.org/) is a collection of sentences and translation, mainly aiming for language learning.
It is available for more than 300 languages.

This script downloads the Tatoeba corpus and extracts the sentences & translations in the languages you like
"""

import gzip
import os
import tarfile

from sentence_transformers.util import http_get

# Note: Tatoeba uses 3 letter languages codes (ISO-639-2),
# while other datasets like OPUS use 2 letter language codes (ISO-639-1)
# For training of sentence transformers, which type of language code is used doesn't matter.
# For language codes, see: https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
source_languages = set(["eng"])
target_languages = set(["deu", "ara", "tur", "spa", "ita", "fra"])

num_dev_sentences = 1000  # Number of sentences that are used to create a development set


tatoeba_folder = "../datasets/tatoeba"
output_folder = "parallel-sentences/"


sentences_file_bz2 = os.path.join(tatoeba_folder, "sentences.tar.bz2")
sentences_file = os.path.join(tatoeba_folder, "sentences.csv")
links_file_bz2 = os.path.join(tatoeba_folder, "links.tar.bz2")
links_file = os.path.join(tatoeba_folder, "links.csv")

download_url = "https://downloads.tatoeba.org/exports/"


os.makedirs(tatoeba_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Download files if needed
for filepath in [sentences_file_bz2, links_file_bz2]:
    if not os.path.exists(filepath):
        url = download_url + os.path.basename(filepath)
        print("Download", url)
        http_get(url, filepath)

# Extract files if needed
if not os.path.exists(sentences_file):
    print("Extract", sentences_file_bz2)
    tar = tarfile.open(sentences_file_bz2, "r:bz2")
    tar.extract("sentences.csv", path=tatoeba_folder)
    tar.close()

if not os.path.exists(links_file):
    print("Extract", links_file_bz2)
    tar = tarfile.open(links_file_bz2, "r:bz2")
    tar.extract("links.csv", path=tatoeba_folder)
    tar.close()


# Read sentences
sentences = {}
all_langs = target_languages.union(source_languages)
print("Read sentences.csv file")
with open(sentences_file, encoding="utf8") as fIn:
    for line in fIn:
        id, lang, sentence = line.strip().split("\t")
        if lang in all_langs:
            sentences[id] = (lang, sentence)

# Read links that map the translations between different languages
print("Read links.csv")
translations = {src_lang: {trg_lang: {} for trg_lang in target_languages} for src_lang in source_languages}
with open(links_file, encoding="utf8") as fIn:
    for line in fIn:
        src_id, target_id = line.strip().split()

        if src_id in sentences and target_id in sentences:
            src_lang, src_sent = sentences[src_id]
            trg_lang, trg_sent = sentences[target_id]

            if src_lang in source_languages and trg_lang in target_languages:
                if src_sent not in translations[src_lang][trg_lang]:
                    translations[src_lang][trg_lang][src_sent] = []
                translations[src_lang][trg_lang][src_sent].append(trg_sent)

# Write everything to the output folder
print("Write output files")
for src_lang in source_languages:
    for trg_lang in target_languages:
        source_sentences = list(translations[src_lang][trg_lang])
        train_sentences = source_sentences[num_dev_sentences:]
        dev_sentences = source_sentences[0:num_dev_sentences]

        print(f"{src_lang}-{trg_lang} has {len(source_sentences)} sentences")
        if len(dev_sentences) > 0:
            with gzip.open(
                os.path.join(output_folder, f"Tatoeba-{src_lang}-{trg_lang}-dev.tsv.gz"),
                "wt",
                encoding="utf8",
            ) as fOut:
                for sent in dev_sentences:
                    fOut.write("\t".join([sent] + translations[src_lang][trg_lang][sent]))
                    fOut.write("\n")

        if len(train_sentences) > 0:
            with gzip.open(
                os.path.join(output_folder, f"Tatoeba-{src_lang}-{trg_lang}-train.tsv.gz"),
                "wt",
                encoding="utf8",
            ) as fOut:
                for sent in train_sentences:
                    fOut.write("\t".join([sent] + translations[src_lang][trg_lang][sent]))
                    fOut.write("\n")


print("---DONE---")
