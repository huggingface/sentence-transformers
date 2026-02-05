"""
In this example we train a semantic search model to search through Wikipedia
articles about programming articles & technologies.

We use the text paragraphs from the following Wikipedia articles:
Assembly language, C , C Sharp , C++, Go , Java , JavaScript, Keras, Laravel, MATLAB, Matplotlib, MongoDB, MySQL, Natural Language Toolkit, NumPy, pandas (software), Perl, PHP, PostgreSQL, Python , PyTorch, R , React, Rust , Scala , scikit-learn, SciPy, Swift , TensorFlow, Vue.js

In:
1_programming_query_generation.py - We generate queries for all paragraphs from these articles
2_programming_train_bi-encoder.py - We train a SentenceTransformer bi-encoder with these generated queries. This results in a model we can then use for semantic search (for the given Wikipedia articles).
3_programming_semantic_search.py - Shows how the trained model can be used for semantic search
"""

import gzip
import json
import os

import torch
import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from sentence_transformers import util

paragraphs = set()

# We use the Wikipedia articles of certain programming languages
corpus_filepath = "wiki-programmming-20210101.jsonl.gz"
if not os.path.exists(corpus_filepath):
    util.http_get("https://sbert.net/datasets/wiki-programmming-20210101.jsonl.gz", corpus_filepath)

with gzip.open(corpus_filepath, "rt") as fIn:
    for line in fIn:
        data = json.loads(line.strip())

        for p in data["paragraphs"]:
            if len(p) > 100:  # Only take paragraphs with at least 100 chars
                paragraphs.add(p)

paragraphs = list(paragraphs)
print("Paragraphs:", len(paragraphs))


# Now we load the model that is able to generate queries given a paragraph.
# This model was trained on the MS MARCO dataset, a dataset with 500k
# queries from Bing and the respective relevant passage
tokenizer = T5Tokenizer.from_pretrained("BeIR/query-gen-msmarco-t5-large-v1")
model = T5ForConditionalGeneration.from_pretrained("BeIR/query-gen-msmarco-t5-large-v1")
model.eval()

# Select the device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Parameters for generation
batch_size = 8  # Batch size
num_queries = 5  # Number of queries to generate for every paragraph
max_length_paragraph = 300  # Max length for paragraph
max_length_query = 64  # Max length for output query

# Now for every paragraph in our corpus, we generate the queries
with open("generated_queries.tsv", "w", encoding="utf-8") as fOut:
    for start_idx in tqdm.trange(0, len(paragraphs), batch_size):
        sub_paragraphs = paragraphs[start_idx : start_idx + batch_size]
        inputs = tokenizer.prepare_seq2seq_batch(
            sub_paragraphs, max_length=max_length_paragraph, truncation=True, return_tensors="pt"
        ).to(device)
        outputs = model.generate(
            **inputs, max_length=max_length_query, do_sample=True, top_p=0.95, num_return_sequences=num_queries
        )

        for idx, out in enumerate(outputs):
            query = tokenizer.decode(out, skip_special_tokens=True)
            para = sub_paragraphs[int(idx / num_queries)]
            fOut.write("{}\t{}\n".format(query.replace("\t", " ").strip(), para.replace("\t", " ").strip()))
