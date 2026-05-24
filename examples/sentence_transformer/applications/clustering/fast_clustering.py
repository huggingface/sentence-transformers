"""
This is a more complex example on performing clustering on large scale dataset.

This examples find in a large set of sentences local communities, i.e., groups of sentences that are highly
similar. You can freely configure the threshold what is considered as similar. A high threshold will
only find extremely similar sentences, a lower threshold will find more sentence that are less similar.

A second parameter is 'min_community_size': Only communities with at least a certain number of sentences will be returned.

The method for finding the communities is extremely fast, for clustering 50k sentences it requires only 5 seconds (plus embedding computation).

In this example, we download a large set of questions from Quora and then find similar questions in this set.
"""

import time

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# We use the Quora Duplicate Questions Dataset (https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
# and find similar questions in it
max_corpus_size = 50000  # We limit our corpus to only the first 50k questions


# Get all unique sentences from the dataset
dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
corpus_sentences = set()
for row in dataset:
    corpus_sentences.add(row["sentence1"])
    corpus_sentences.add(row["sentence2"])
    if len(corpus_sentences) >= max_corpus_size:
        break

corpus_sentences = list(corpus_sentences)
print("Encode the corpus. This might take a while")
corpus_embeddings = model.encode(corpus_sentences, batch_size=64, show_progress_bar=True, convert_to_tensor=True)


print("Start clustering")
start_time = time.time()

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)

print(f"Clustering done after {time.time() - start_time:.2f} sec")

# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    for sentence_id in cluster[0:3]:
        print("\t", corpus_sentences[sentence_id])
    print("\t", "...")
    for sentence_id in cluster[-3:]:
        print("\t", corpus_sentences[sentence_id])
