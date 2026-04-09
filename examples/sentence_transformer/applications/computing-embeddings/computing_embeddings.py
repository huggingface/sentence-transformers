"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

import logging

import numpy as np

from sentence_transformers import LoggingHandler, SentenceTransformer

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout


# Load pre-trained Sentence Transformer Model. It will be downloaded automatically
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embed a list of sentences
sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]
sentence_embeddings = model.encode(sentences)

# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
