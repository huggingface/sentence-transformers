"""
This example demonstrates how to use parallel computing with the tqdm package
to show progress while encoding sentences using the SentenceTransformer model.
"""

import logging
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sentence_transformers import LoggingHandler, SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


def encode_sentences(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(sentences)


if __name__ == "__main__":
    # Create a large list of sentences
    sentences = [f"This is sentence {i}" for i in range(100_000)]

    # Split sentences into chunks for parallel processing
    num_chunks = cpu_count()
    chunk_size = len(sentences) // num_chunks
    sentence_chunks = [sentences[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    # Use multiprocessing Pool to encode sentences in parallel
    with Pool(processes=num_chunks) as pool:
        results = list(tqdm(pool.imap(encode_sentences, sentence_chunks), total=num_chunks))

    # Flatten the list of results
    sentence_embeddings = [embedding for result in results for embedding in result]

    print("Embeddings computed. Number of embeddings:", len(sentence_embeddings))
