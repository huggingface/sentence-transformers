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

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import http_get, semantic_search

# Load the model we trained in 2_programming_train_bi-encoder.py
model = SentenceTransformer("output/programming-model")

# Load the corpus
docs = []
corpus_filepath = "wiki-programmming-20210101.jsonl.gz"
if not os.path.exists(corpus_filepath):
    http_get("https://sbert.net/datasets/wiki-programmming-20210101.jsonl.gz", corpus_filepath)

with gzip.open(corpus_filepath, "rt") as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        title = data["title"]
        for p in data["paragraphs"]:
            if len(p) > 100:  # Only take paragraphs with at least 100 chars
                docs.append((title, p))

paragraph_emb = model.encode([d[1] for d in docs], convert_to_tensor=True)

print("Available Wikipedia Articles:")
print(", ".join(sorted(list(set([d[0] for d in docs])))))

# Example for semantic search
while True:
    query = input("Query: ")
    query_emb = model.encode(query, convert_to_tensor=True)
    hits = semantic_search(query_emb, paragraph_emb, top_k=3)[0]

    for hit in hits:
        doc = docs[hit["corpus_id"]]
        print("{:.2f}\t{}\t\t{}".format(hit["score"], doc[0], doc[1]))

    print("\n=================\n")
