"""
This examples demonstrates the setup for Question-Answer-Retrieval.

You can input a query or a question. The script then uses semantic search
to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

As model, we use: nq-distilbert-base-v1

It was trained on the Natural Questions dataset, a dataset with real questions from Google Search
together with annotated data from Wikipedia providing the answer. For the passages, we encode the
Wikipedia article tile together with the individual text passages.

Google Colab Example: https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing
"""

import gzip
import json
import os
import time

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import get_device_name, http_get, semantic_search

# We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
model_name = "sentence-transformers/nq-distilbert-base-v1"
bi_encoder = SentenceTransformer(model_name)
top_k = 5  # Number of passages we want to retrieve with the bi-encoder

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = "data/simplewiki-2020-11-01.jsonl.gz"

if not os.path.exists(wikipedia_filepath):
    http_get("http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz", wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, "rt", encoding="utf8") as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        for paragraph in data["paragraphs"]:
            # We encode the passages as [title, text]
            passages.append([data["title"], paragraph])

# If you like, you can also limit the number of passages you want to use
print("Passages:", len(passages))

# To speed things up, pre-computed embeddings are downloaded.
# The provided file encoded the passages with the model 'nq-distilbert-base-v1'
if model_name == "sentence-transformers/nq-distilbert-base-v1":
    embeddings_filepath = "simplewiki-2020-11-01-nq-distilbert-base-v1.pt"
    if not os.path.exists(embeddings_filepath):
        http_get("http://sbert.net/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt", embeddings_filepath)

    corpus_embeddings = torch.load(embeddings_filepath)
    corpus_embeddings = corpus_embeddings.float()  # Convert embedding file to float
    device = get_device_name()
    corpus_embeddings = corpus_embeddings.to(device)
else:  # Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

while True:
    query = input("Please enter a question: ")

    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query)
    print(f"Results (after {end_time - start_time:.3f} seconds):")
    for hit in hits:
        print("\t{:.3f}\t{}".format(hit["score"], passages[hit["corpus_id"]]))

    print("\n\n========\n")
