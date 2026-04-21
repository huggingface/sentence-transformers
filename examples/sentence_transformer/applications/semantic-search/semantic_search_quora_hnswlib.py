"""
This example uses Approximate Nearest Neighbor Search (ANN) with Hnswlib  (https://github.com/nmslib/hnswlib/).

Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.

This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.

In this example, we use Hnswlib: It is a fast and easy to use library, with excellent results on common benchmarks.

Usually you can install Hnswlib by running:
pip install hnswlib

For more details, see https://github.com/nmslib/hnswlib/

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions (we only use 100k in this example):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs

As embeddings model, we use the SBERT model 'quora-distilbert-multilingual',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).
"""

import csv
import os
import pickle
import time

import hnswlib

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import http_get, semantic_search

model_name = "sentence-transformers/quora-distilbert-multilingual"
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 100000

embedding_cache_path = "quora-embeddings-{}-size-{}.pkl".format(model_name.replace("/", "_"), max_corpus_size)


embedding_size = 768  # Size of embeddings
top_k_hits = 10  # Output k hits

# Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            corpus_sentences.add(row["question1"])
            if len(corpus_sentences) >= max_corpus_size:
                break

            corpus_sentences.add(row["question2"])
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disk")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({"sentences": corpus_sentences, "embeddings": corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disk")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data["sentences"]
        corpus_embeddings = cache_data["embeddings"]

# Defining our hnswlib index
index_path = "./hnswlib.index"
# We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
index = hnswlib.Index(space="cosine", dim=embedding_size)

if os.path.exists(index_path):
    print("Loading index...")
    index.load_index(index_path)
else:
    ### Create the HNSWLIB index
    print("Start creating HNSWLIB index")
    index.init_index(max_elements=len(corpus_embeddings), ef_construction=400, M=64)

    # Then we train the index to find a suitable clustering
    index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))

    print("Saving index to:", index_path)
    index.save_index(index_path)

# Controlling the recall by setting ef:
index.set_ef(50)  # ef should always be > top_k_hits

######### Search in the index ###########

print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")

while True:
    inp_question = input("Please enter a question: ")

    start_time = time.time()
    question_embedding = model.encode(inp_question)

    # We use hnswlib knn_query method to find the top_k_hits
    corpus_ids, distances = index.knn_query(question_embedding, k=top_k_hits)

    # We extract corpus ids and scores for the first query
    hits = [{"corpus_id": id, "score": 1 - score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x["score"], reverse=True)
    end_time = time.time()

    print("Input question:", inp_question)
    print(f"Results (after {end_time - start_time:.3f} seconds):")
    for hit in hits[0:top_k_hits]:
        print("\t{:.3f}\t{}".format(hit["score"], corpus_sentences[hit["corpus_id"]]))

    # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
    # Here, we compute the recall of ANN compared to the exact results
    correct_hits = semantic_search(question_embedding, corpus_embeddings, top_k=top_k_hits)[0]
    correct_hits_ids = set([hit["corpus_id"] for hit in correct_hits])

    ann_corpus_ids = set([hit["corpus_id"] for hit in hits])
    if len(ann_corpus_ids) != len(correct_hits_ids):
        print("Approximate Nearest Neighbor returned a different number of results than expected")

    recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
    print(f"\nApproximate Nearest Neighbor Recall@{top_k_hits}: {recall * 100:.2f}")

    if recall < 1:
        print("Missing results:")
        for hit in correct_hits[0:top_k_hits]:
            if hit["corpus_id"] not in ann_corpus_ids:
                print("\t{:.3f}\t{}".format(hit["score"], corpus_sentences[hit["corpus_id"]]))
    print("\n\n========\n")
