"""
This is a simple application for sparse encoder: Computing embeddings.

we have multiple sentences and we want to compute their embeddings.
The embeddings are sparse, meaning that most of the values are zero.
The embeddings are stored in a sparse matrix format, which is more efficient for storage and computation.
we can also visualize the top tokens for each text."""

from sentence_transformers import SparseEncoder

# Initialize the SPLADE model
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Embed a list of sentences
sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]

# Generate embeddings
embeddings = model.encode(sentences)

# Print embedding sim and sparsity
print(f"Embedding dim: {model.get_embedding_dimension()}")

stats = model.sparsity(embeddings)
print(f"Embedding sparsity: {stats}")
print(f"Average non-zero dimensions: {stats['active_dims']:.2f}")
print(f"Sparsity percentage: {stats['sparsity_ratio']:.2%}")


"""
Embedding dim: 30522
Embedding sparsity: {'active_dims': 56.333335876464844, 'sparsity_ratio': 0.9981543366792325}
Average non-zero dimensions: 56.33
Sparsity percentage: 99.82%
"""
# Visualize top tokens for each text
top_k = 10

token_weights = model.decode(embeddings, top_k=top_k)

print(f"\nTop tokens {top_k} for each text:")
# The result is a list of sentence embeddings as numpy arrays
for i, sentence in enumerate(sentences):
    token_scores = ", ".join([f'("{token.strip()}", {value:.2f})' for token, value in token_weights[i]])
    print(f"{i}: {sentence} -> Top tokens:  {token_scores}")

"""
Top tokens 10 for each text:
0: This framework generates embeddings for each input sentence -> Top tokens:  ("framework", 2.19), ("##bed", 2.12), ("input", 1.99), ("each", 1.60), ("em", 1.58), ("sentence", 1.49), ("generate", 1.42), ("##ding", 1.33), ("sentences", 1.10), ("create", 0.93)
1: Sentences are passed as a list of string. -> Top tokens:  ("string", 2.72), ("pass", 2.24), ("sentences", 2.15), ("passed", 2.07), ("sentence", 1.90), ("strings", 1.86), ("list", 1.84), ("lists", 1.49), ("as", 1.18), ("passing", 0.73)
2: The quick brown fox jumps over the lazy dog. -> Top tokens:  ("lazy", 2.18), ("fox", 1.67), ("brown", 1.56), ("over", 1.52), ("dog", 1.50), ("quick", 1.49), ("jump", 1.39), ("dogs", 1.25), ("foxes", 0.99), ("jumping", 0.84)
"""

# Example of using max_active_dims during encoding
print("\n--- Using max_active_dims during encoding ---")
# Generate embeddings with limited active dimensions
embeddings_limited = model.encode(sentences, max_active_dims=32)
stats_limited = model.sparsity(embeddings_limited)
print(f"Limited embedding sparsity: {stats_limited}")
print(f"Average non-zero dimensions: {stats_limited['active_dims']:.2f}")
print(f"Sparsity percentage: {stats_limited['sparsity_ratio']:.2%}")

"""
--- Using max_active_dims during encoding ---
Limited embedding sparsity: {'active_dims': 32.0, 'sparsity_ratio': 0.9989516139030457}
Average non-zero dimensions: 32.00
Sparsity percentage: 99.90%
"""

# Comparing memory usage
print("\n--- Comparing memory usage ---")


def get_memory_size(tensor):
    if tensor.is_sparse:
        # For sparse tensors, only count non-zero elements
        return (
            tensor._values().element_size() * tensor._values().nelement()
            + tensor._indices().element_size() * tensor._indices().nelement()
        )
    else:
        return tensor.element_size() * tensor.nelement()


print(f"Original embeddings memory: {get_memory_size(embeddings) / 1024:.2f} KB")
print(f"Embeddings with max_active_dims=32 memory: {get_memory_size(embeddings_limited) / 1024:.2f} KB")

"""
--- Comparing memory usage ---
Original embeddings memory: 3.32 KB
Embeddings with max_active_dims=32 memory: 1.88 KB
"""
