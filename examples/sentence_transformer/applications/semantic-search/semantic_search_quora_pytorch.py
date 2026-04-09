"""
This script contains an example how to perform semantic search with PyTorch. It performs exact nearest neighborh search.

As dataset, we use the Quora Duplicate Questions dataset, which contains about 500k questions (we only use about 100k):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs


As embeddings model, we use the SBERT model 'quora-distilbert-multilingual',
that it aligned for 100 languages. I.e., you can type in a question in various languages and it will
return the closest questions in the corpus (questions in the corpus are mainly in English).


Google Colab example: https://colab.research.google.com/drive/12cn5Oo0v3HfQQ8Tv6-ukgxXSmT3zl35A?usp=sharing
"""

import csv
import os
import pickle
import time

from sentence_transformers import SentenceTransformer, util

model_name = "sentence-transformers/quora-distilbert-multilingual"
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 100000


embedding_cache_path = "quora-embeddings-{}-size-{}.pkl".format(model_name.replace("/", "_"), max_corpus_size)


# Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

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
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_tensor=True)

    print("Store file on disk")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({"sentences": corpus_sentences, "embeddings": corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disk")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data["sentences"][0:max_corpus_size]
        corpus_embeddings = cache_data["embeddings"][0:max_corpus_size]

###############################
print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")

# Move embeddings to the target device of the model
corpus_embeddings = corpus_embeddings.to(model.device)

while True:
    inp_question = input("Please enter a question: ")

    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings)
    end_time = time.time()
    hits = hits[0]  # Get the hits for the first query

    print("Input question:", inp_question)
    print(f"Results (after {end_time - start_time:.3f} seconds):")
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit["score"], corpus_sentences[hit["corpus_id"]]))

    print("\n\n========\n")
