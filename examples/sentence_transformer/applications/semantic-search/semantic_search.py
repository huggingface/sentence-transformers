"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

import torch

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Corpus with example documents
corpus = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
    "Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.",
    "The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.",
    "SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.",
    "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.",
    "Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.",
]
# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode_document(corpus, convert_to_tensor=True)

# Query sentences:
queries = [
    "How do artificial neural networks work?",
    "What technology is used for modern space exploration?",
    "How can we address climate change challenges?",
]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode_query(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(f"(Score: {score:.4f})", corpus[idx])

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
