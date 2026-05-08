"""
This is a simple application for sparse encoder: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""

from sentence_transformers import SparseEncoder
from sentence_transformers.util import semantic_search

# 1. Load a pretrained SparseEncoder model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# 2. Encode a corpus of texts using the SparseEncoder model
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
corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True)

# 3. Encode the user queries using the same SparseEncoder model
queries = [
    "How do artificial neural networks work?",
    "What technology is used for modern space exploration?",
    "How can we address climate change challenges?",
]
query_embeddings = model.encode_query(queries, convert_to_tensor=True)

# 4. Use the similarity function to compute the similarity scores between the query and corpus embeddings
top_k = min(5, len(corpus))  # Find at most 5 sentences of the corpus for each query sentence
results = semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=model.similarity)

# 5. Sort the results and print the top 5 most similar sentences for each query
for query_id, query in enumerate(queries):
    pointwise_scores = model.intersection(query_embeddings[query_id], corpus_embeddings)

    print(f"Query: {query}")
    for res in results[query_id]:
        corpus_id, score = res.values()
        sentence = corpus[corpus_id]

        pointwise_score = model.decode(pointwise_scores[corpus_id], top_k=10)

        token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in pointwise_score])

        print(f"Score: {score:.4f} - Sentence: {sentence} - Top influential tokens: {token_scores}")
    print("")

"""
Query: How do artificial neural networks work?
Score: 16.9053 - Sentence: Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. - Top influential tokens: ("neural", 5.71), ("networks", 3.24), ("network", 2.93), ("brain", 2.10), ("computer", 0.50), ("##uron", 0.32), ("artificial", 0.27), ("technology", 0.27), ("communication", 0.27), ("connection", 0.21)
Score: 13.6119 - Sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. - Top influential tokens: ("artificial", 3.71), ("neural", 3.15), ("networks", 1.78), ("brain", 1.22), ("network", 1.12), ("ai", 1.07), ("machine", 0.39), ("robot", 0.20), ("technology", 0.20), ("algorithm", 0.18)
Score: 2.7373 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("machine", 0.78), ("computer", 0.50), ("technology", 0.32), ("artificial", 0.22), ("robot", 0.21), ("ai", 0.20), ("process", 0.16), ("theory", 0.11), ("technique", 0.11), ("fuzzy", 0.06)
Score: 2.1430 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("technology", 0.42), ("function", 0.41), ("mechanism", 0.21), ("sensor", 0.21), ("device", 0.18), ("process", 0.18), ("generator", 0.13), ("detection", 0.10), ("technique", 0.10), ("tracking", 0.05)
Score: 2.0195 - Sentence: Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments. - Top influential tokens: ("robot", 0.67), ("function", 0.34), ("technology", 0.29), ("device", 0.23), ("experiment", 0.20), ("machine", 0.10), ("artificial", 0.08), ("design", 0.04), ("useful", 0.03), ("they", 0.02)

Query: What technology is used for modern space exploration?
Score: 10.4748 - Sentence: SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond. - Top influential tokens: ("space", 4.40), ("technology", 1.15), ("nasa", 1.06), ("mars", 0.63), ("exploration", 0.52), ("spacecraft", 0.44), ("robot", 0.32), ("rocket", 0.28), ("astronomy", 0.27), ("travel", 0.26)
Score: 9.3818 - Sentence: The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy. - Top influential tokens: ("space", 3.89), ("nasa", 1.09), ("astronomy", 0.93), ("discovery", 0.48), ("instrument", 0.47), ("technology", 0.35), ("device", 0.26), ("spacecraft", 0.25), ("invented", 0.22), ("equipment", 0.22)
Score: 8.5147 - Sentence: Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments. - Top influential tokens: ("technology", 1.39), ("mars", 0.79), ("exploration", 0.78), ("robot", 0.67), ("used", 0.66), ("nasa", 0.52), ("spacecraft", 0.44), ("device", 0.39), ("explore", 0.38), ("travel", 0.25)
Score: 7.6993 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("technology", 1.99), ("tech", 1.76), ("technologies", 1.74), ("equipment", 0.32), ("device", 0.31), ("technological", 0.28), ("mining", 0.22), ("sensor", 0.19), ("tool", 0.18), ("software", 0.11)
Score: 2.5526 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("technology", 1.52), ("machine", 0.27), ("robot", 0.21), ("computer", 0.18), ("engineering", 0.12), ("technique", 0.11), ("science", 0.05), ("technological", 0.05), ("techniques", 0.02), ("innovation", 0.01)

Query: How can we address climate change challenges?
Score: 9.5587 - Sentence: Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities. - Top influential tokens: ("climate", 3.21), ("warming", 2.87), ("weather", 1.58), ("change", 0.46), ("global", 0.41), ("environmental", 0.39), ("storm", 0.19), ("pollution", 0.15), ("environment", 0.11), ("adaptation", 0.08)
Score: 1.3191 - Sentence: Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground. - Top influential tokens: ("warming", 0.39), ("pollution", 0.34), ("environmental", 0.15), ("goal", 0.12), ("strategy", 0.07), ("monitoring", 0.07), ("protection", 0.06), ("greenhouse", 0.05), ("safety", 0.02), ("escape", 0.01)
Score: 1.0774 - Sentence: Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time. - Top influential tokens: ("conservation", 0.39), ("sustainability", 0.18), ("environmental", 0.18), ("sustainable", 0.13), ("agriculture", 0.13), ("alternative", 0.07), ("recycling", 0.00)
Score: 0.2401 - Sentence: Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. - Top influential tokens: ("strategy", 0.10), ("success", 0.06), ("foster", 0.04), ("engineering", 0.03), ("innovation", 0.00), ("research", 0.00)
Score: 0.1516 - Sentence: Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. - Top influential tokens: ("strategy", 0.09), ("foster", 0.04), ("research", 0.01), ("approach", 0.01), ("engineering", 0.01)
"""
