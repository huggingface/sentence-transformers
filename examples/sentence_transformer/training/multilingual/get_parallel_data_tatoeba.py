"""
Tatoeba (https://tatoeba.org/) is a collection of sentences and translations, mainly aiming for
language learning. It is available for more than 300 languages.

This script writes parallel sentence tsv files from the Hugging Face dataset
``sentence-transformers/parallel-sentences-tatoeba``. The training procedure can be found in
``make_multilingual.py``, which loads the same Hugging Face dataset directly.
"""

import gzip
import os

from datasets import load_dataset

# Tatoeba uses 3-letter language codes (ISO-639-2), while the Hugging Face dataset uses 2-letter
# codes (ISO-639-1) in its config names. The output filenames keep the 3-letter codes (Tatoeba's
# convention) and are mapped to the dataset's config names below.
iso3_to_iso2 = {"eng": "en", "deu": "de", "ara": "ar", "tur": "tr", "spa": "es", "ita": "it", "fra": "fr"}

hf_dataset = "sentence-transformers/parallel-sentences-tatoeba"
source_lang = "eng"
target_languages = ["deu", "ara", "tur", "spa", "ita", "fra"]

output_folder = "parallel-sentences/"
os.makedirs(output_folder, exist_ok=True)

for target_lang in target_languages:
    subset = f"{iso3_to_iso2[source_lang]}-{iso3_to_iso2[target_lang]}"
    for split in ("train", "dev"):
        output_filename = os.path.join(output_folder, f"Tatoeba-{source_lang}-{target_lang}-{split}.tsv.gz")
        if os.path.exists(output_filename):
            continue
        try:
            dataset = load_dataset(hf_dataset, subset, split=split)
        except Exception as e:
            print(f"Skipping {subset} ({split}): {e}")
            continue
        num_written = 0
        with gzip.open(output_filename, "wt", encoding="utf8") as fOut:
            for row in dataset:
                english = row["english"].strip()
                non_english = row["non_english"].strip()
                if english and non_english:
                    fOut.write(f"{english}\t{non_english}\n")
                    num_written += 1
        print(f"Wrote {output_filename}: {num_written} pairs")

print("---DONE---")
