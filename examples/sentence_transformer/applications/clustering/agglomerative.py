"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
"""

from sklearn.cluster import AgglomerativeClustering

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Corpus with example sentences
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "The baby is carried by the woman",
    "A man is riding a horse.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah is running behind its prey.",
    "A cheetah chases prey on across a field.",
]
corpus_embeddings = embedder.encode(corpus)

# Some models don't automatically normalize the embeddings, in which case you should normalize the embeddings:
# corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

# Perform agglomerative clustering
clustering_model = AgglomerativeClustering(
    n_clusters=None, distance_threshold=1.5
)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in clustered_sentences.items():
    print("Cluster ", i + 1)
    print(cluster)
    print("")
