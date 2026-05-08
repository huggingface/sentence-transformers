"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

import logging

from sentence_transformers import LoggingHandler, SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == "__main__":
    # Create a large list of 100k sentences
    sentences = [f"This is sentence {i}" for i in range(100000)]

    # Define the model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Compute the embeddings using the multi-process pool
    emb = model.encode(sentences, pool=pool)
    print("Embeddings computed. Shape:", emb.shape)

    # Optional: Stop the processes in the pool
    model.stop_multi_process_pool(pool)
