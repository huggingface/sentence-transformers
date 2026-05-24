"""
This script writes parallel sentence tsv files that can be used to extend existing sentence
embedding models to new languages.

It loads the WikiMatrix parallel corpus (mined from Wikipedia, see
https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix) from the Hugging Face dataset
``sentence-transformers/parallel-sentences-wikimatrix``, which is already filtered by LASER
similarity score. The training procedure can be found in ``make_multilingual.py``, which loads the
same Hugging Face dataset directly.

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://huggingface.co/papers/2004.09813
"""

import gzip
import os

from datasets import load_dataset

hf_dataset = "sentence-transformers/parallel-sentences-wikimatrix"
source_lang = "en"  # Language our (monolingual) teacher model understands
target_languages = ["de", "es", "it", "fr", "ar", "tr"]  # New languages we want to extend to

output_folder = "parallel-sentences/"
os.makedirs(output_folder, exist_ok=True)

for target_lang in target_languages:
    subset = f"{source_lang}-{target_lang}"
    for split in ("train", "dev"):
        output_filename = os.path.join(output_folder, f"WikiMatrix-{source_lang}-{target_lang}-{split}.tsv.gz")
        if os.path.exists(output_filename):
            continue
        try:
            dataset = load_dataset(hf_dataset, subset, split=split)
        except Exception as e:
            print(f"Skipping {subset} ({split}): {e}")
            continue
        with gzip.open(output_filename, "wt", encoding="utf8") as fOut:
            for row in dataset:
                english = row["english"].strip()
                non_english = row["non_english"].strip()
                if english and non_english:
                    fOut.write(f"{english}\t{non_english}\n")
        print(f"Wrote {output_filename}: {len(dataset)} pairs")

print("---DONE---")
