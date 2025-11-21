"""
This script tests the approach on the BUCC 2018 shared task on finding parallel sentences:
https://comparable.limsi.fr/bucc2018/bucc2018-task.html

You can download the necessary files from there.

We have used it in our paper (https://huggingface.co/papers/2004.09813) in Section 4.2 to evaluate different multilingual models.

This script requires that you have FAISS installed:
https://github.com/facebookresearch/faiss
"""

import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from bitext_mining_utils import kNN, score_candidates
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, models

# Model we want to use for bitext mining. LaBSE achieves state-of-the-art performance
model_name = "LaBSE"
model = SentenceTransformer(model_name)

# Input files for BUCC2018 shared task
source_file = "bucc2018/de-en/de-en.training.de"
target_file = "bucc2018/de-en/de-en.training.en"
labels_file = "bucc2018/de-en/de-en.training.gold"


# We base the scoring on k nearest neighbors for each element
knn_neighbors = 4

# Min score for text pairs. Note, score can be larger than 1
min_threshold = 1

# Do we want to use exact search of approximate nearest neighbor search (ANN)
# Exact search: Slower, but we don't miss any parallel sentences
# ANN: Faster, but the recall will be lower
use_ann_search = True

# Number of clusters for ANN. Optimal number depends on dataset size
ann_num_clusters = 32768

# How many cluster to explorer for search. Higher number = better recall, slower
ann_num_cluster_probe = 5

# To save memory, we can use PCA to reduce the dimensionality from 768 to for example 128 dimensions
# The encoded embeddings will hence require 6 times less memory. However, we observe a small drop in performance.
use_pca = False
pca_dimensions = 128

# We store the embeddings on disk, so that they can later be loaded from disk
source_embedding_file = f"{model_name}_{os.path.basename(source_file)}_{pca_dimensions if use_pca else model.get_sentence_embedding_dimension()}.emb"
target_embedding_file = f"{model_name}_{os.path.basename(target_file)}_{pca_dimensions if use_pca else model.get_sentence_embedding_dimension()}.emb"


# Use PCA to reduce the dimensionality of the sentence embedding model
if use_pca:
    # We use a smaller number of training sentences to learn the PCA
    train_sent = []
    num_train_sent = 20000

    with open(source_file, encoding="utf8") as fSource, open(target_file, encoding="utf8") as fTarget:
        for line_source, line_target in zip(fSource, fTarget):
            id, sentence = line_source.strip().split("\t", maxsplit=1)
            train_sent.append(sentence)

            id, sentence = line_target.strip().split("\t", maxsplit=1)
            train_sent.append(sentence)

            if len(train_sent) >= num_train_sent:
                break

    print("Encode training embeddings for PCA")
    train_matrix = model.encode(train_sent, show_progress_bar=True, convert_to_numpy=True)
    pca = PCA(n_components=pca_dimensions)
    pca.fit(train_matrix)

    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=pca_dimensions,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
    model.add_module("dense", dense)


print("Read source file")
source = {}
with open(source_file, encoding="utf8") as fIn:
    for line in fIn:
        id, sentence = line.strip().split("\t", maxsplit=1)
        source[id] = sentence

print("Read target file")
target = {}
with open(target_file, encoding="utf8") as fIn:
    for line in fIn:
        id, sentence = line.strip().split("\t", maxsplit=1)
        target[id] = sentence

labels = defaultdict(lambda: defaultdict(bool))
num_total_parallel = 0
with open(labels_file) as fIn:
    for line in fIn:
        src_id, trg_id = line.strip().split("\t")
        if src_id in source and trg_id in target:
            labels[src_id][trg_id] = True
            labels[trg_id][src_id] = True
            num_total_parallel += 1

print("Source Sentences:", len(source))
print("Target Sentences:", len(target))
print("Num Parallel:", num_total_parallel)

### Encode source sentences
source_ids = list(source.keys())
source_sentences = [source[id] for id in source_ids]

if not os.path.exists(source_embedding_file):
    print("Encode source sentences")
    source_embeddings = model.encode(source_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open(source_embedding_file, "wb") as fOut:
        pickle.dump(source_embeddings, fOut)
else:
    with open(source_embedding_file, "rb") as fIn:
        source_embeddings = pickle.load(fIn)

### Encode target sentences
target_ids = list(target.keys())
target_sentences = [target[id] for id in target_ids]

if not os.path.exists(target_embedding_file):
    print("Encode target sentences")
    target_embeddings = model.encode(target_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open(target_embedding_file, "wb") as fOut:
        pickle.dump(target_embeddings, fOut)
else:
    with open(target_embedding_file, "rb") as fIn:
        target_embeddings = pickle.load(fIn)

##### Now we start to search for parallel (translated) sentences

# Normalize embeddings
x = source_embeddings
y = target_embeddings

print("Shape Source:", x.shape)
print("Shape Target:", y.shape)

x = x / np.linalg.norm(x, axis=1, keepdims=True)
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
bitext_list = []
for i in np.argsort(-scores):
    src_ind, trg_ind = indices[i]
    src_ind = int(src_ind)
    trg_ind = int(trg_ind)

    if scores[i] < min_threshold:
        break

    if src_ind not in seen_src and trg_ind not in seen_trg:
        seen_src.add(src_ind)
        seen_trg.add(trg_ind)
        bitext_list.append([scores[i], source_ids[src_ind], target_ids[trg_ind]])


# Measure Performance by computing the threshold
# that leads to the best F1 score performance
bitext_list = sorted(bitext_list, key=lambda x: x[0], reverse=True)

n_extract = n_correct = 0
threshold = 0
best_f1 = best_recall = best_precision = 0
average_precision = 0

for idx in range(len(bitext_list)):
    score, id1, id2 = bitext_list[idx]
    n_extract += 1
    if labels[id1][id2] or labels[id2][id1]:
        n_correct += 1
        precision = n_correct / n_extract
        recall = n_correct / num_total_parallel
        f1 = 2 * precision * recall / (precision + recall)
        average_precision += precision
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            threshold = (bitext_list[idx][0] + bitext_list[min(idx + 1, len(bitext_list) - 1)][0]) / 2

print("Best Threshold:", threshold)
print("Recall:", best_recall)
print("Precision:", best_precision)
print("F1:", best_f1)
