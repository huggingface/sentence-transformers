"""
This scripts show how to mine parallel (translated) sentences from two list of monolingual sentences.

As input, you specific two text files that have sentences in every line. Then, the
sentence-transformers/LaBSE model is used to find parallel (translated) across these two files.

The result is written to disk.

A large source for monolingual sentences in different languages is:
http://data.statmt.org/cc-100/

This script requires that you have FAISS installed:
https://github.com/facebookresearch/faiss
"""

import gzip

import numpy as np
import torch
import tqdm
from bitext_mining_utils import file_open, kNN, score_candidates
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Dense

# Model we want to use for bitext mining. sentence-transformers/LaBSE achieves state-of-the-art performance
model_name = "sentence-transformers/LaBSE"
model = SentenceTransformer(model_name)

# Input files. We interpret every line as sentence.
source_file = "data/so.txt.xz"
target_file = "data/yi.txt.xz"

# Only consider sentences that are between min_sent_len and max_sent_len characters long
min_sent_len = 10
max_sent_len = 200

# We base the scoring on k nearest neighbors for each element
knn_neighbors = 4

# Min score for text pairs. Note, score can be larger than 1
min_threshold = 1

# Do we want to use exact search of approximate nearest neighbor search (ANN)
# Exact search: Slower, but we don't miss any parallel sentences
# ANN: Faster, but the recall will be lower
use_ann_search = True

# Number of clusters for ANN. Each cluster should have at least 10k entries
ann_num_clusters = 32768

# How many cluster to explorer for search. Higher number = better recall, slower
ann_num_cluster_probe = 3

# To save memory, we can use PCA to reduce the dimensionality from 768 to for example 128 dimensions
# The encoded embeddings will hence require 6 times less memory. However, we observe a small drop in performance.
use_pca = True
pca_dimensions = 128


if use_pca:
    # We use a smaller number of training sentences to learn the PCA
    train_sent = []
    num_train_sent = 20000

    with file_open(source_file) as fSource, file_open(target_file) as fTarget:
        for line_source, line_target in zip(fSource, fTarget):
            if min_sent_len <= len(line_source.strip()) <= max_sent_len:
                sentence = line_source.strip()
                train_sent.append(sentence)

            if min_sent_len <= len(line_target.strip()) <= max_sent_len:
                sentence = line_target.strip()
                train_sent.append(sentence)

            if len(train_sent) >= num_train_sent:
                break

    print("Encode training embeddings for PCA")
    train_matrix = model.encode(train_sent, show_progress_bar=True, convert_to_numpy=True)
    pca = PCA(n_components=pca_dimensions)
    pca.fit(train_matrix)

    dense = Dense(
        in_features=model.get_embedding_dimension(),
        out_features=pca_dimensions,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    model.add_module("dense", dense)


print("Read source file")
source_sentences = set()
with file_open(source_file) as fIn:
    for line in tqdm.tqdm(fIn):
        line = line.strip()
        if len(line) >= min_sent_len and len(line) <= max_sent_len:
            source_sentences.add(line)

print("Read target file")
target_sentences = set()
with file_open(target_file) as fIn:
    for line in tqdm.tqdm(fIn):
        line = line.strip()
        if len(line) >= min_sent_len and len(line) <= max_sent_len:
            target_sentences.add(line)

print("Source Sentences:", len(source_sentences))
print("Target Sentences:", len(target_sentences))


### Encode source sentences
source_sentences = list(source_sentences)


print("Encode source sentences")
source_embeddings = model.encode(source_sentences, show_progress_bar=True, convert_to_numpy=True)


### Encode target sentences
target_sentences = list(target_sentences)

print("Encode target sentences")
target_embeddings = model.encode(target_sentences, show_progress_bar=True, convert_to_numpy=True)


# Normalize embeddings
x = source_embeddings
x = x / np.linalg.norm(x, axis=1, keepdims=True)

y = target_embeddings
y = y / np.linalg.norm(y, axis=1, keepdims=True)

# Perform kNN in both directions
x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
x2y_mean = x2y_sim.mean(axis=1)

y2x_sim, y2x_ind = kNN(y, x, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
y2x_mean = y2x_sim.mean(axis=1)

# Compute forward and backward scores
margin = lambda a, b: a / b
fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

indices = np.stack(
    [np.concatenate([np.arange(x.shape[0]), bwd_best]), np.concatenate([fwd_best, np.arange(y.shape[0])])], axis=1
)
scores = np.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
seen_src, seen_trg = set(), set()

# Extract list of parallel sentences
print("Write sentences to disk")
sentences_written = 0
with gzip.open("parallel-sentences-out.tsv.gz", "wt", encoding="utf8") as fOut:
    for i in np.argsort(-scores):
        src_ind, trg_ind = indices[i]
        src_ind = int(src_ind)
        trg_ind = int(trg_ind)

        if scores[i] < min_threshold:
            break

        if src_ind not in seen_src and trg_ind not in seen_trg:
            seen_src.add(src_ind)
            seen_trg.add(trg_ind)
            fOut.write(
                "{:.4f}\t{}\t{}\n".format(
                    scores[i],
                    source_sentences[src_ind].replace("\t", " "),
                    target_sentences[trg_ind].replace("\t", " "),
                )
            )
            sentences_written += 1

print(f"Done. {sentences_written} sentences written")
