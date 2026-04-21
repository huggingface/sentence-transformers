"""
This examples measures the inference speed of a certain model

Usage:
python evaluation_inference_speed.py
OR
python evaluation_inference_speed.py model_name
"""

import sys
import time

import torch
from datasets import load_dataset

from sentence_transformers import SentenceTransformer

# Limit torch to 4 threads
torch.set_num_threads(4)


model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/bert-base-nli-mean-tokens"

# Load a sentence transformer model
model = SentenceTransformer(model_name)

max_sentences = 100_000
all_nli_dataset = load_dataset("sentence-transformers/all-nli", "pair", split="train")
sentences = list(set(all_nli_dataset["anchor"]))[:max_sentences]

print("Model Name:", model_name)
print("Number of sentences:", len(sentences))

for i in range(3):
    print("Run", i)
    start_time = time.time()
    emb = model.encode(sentences, batch_size=32)
    end_time = time.time()
    diff_time = end_time - start_time
    print(f"Done after {diff_time:.2f} seconds")
    print(f"Speed: {len(sentences) / diff_time:.2f} sentences / second")
    print("=====")
