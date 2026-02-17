import time

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.util.quantization import quantize_embeddings, semantic_search_usearch

# 1. Load the quora corpus with questions
dataset = load_dataset("quora", split="train", trust_remote_code=True).map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_000
corpus = dataset["text"][:max_corpus_size]

# 2. Come up with some queries
queries = [
    "How do I become a good programmer?",
    "How do I become a good data scientist?",
]

# 3. Load the model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# 4. Choose a target precision for the corpus embeddings
corpus_precision = "binary"
# Valid options are: "float32", "uint8", "int8", "ubinary", and "binary"
# But usearch only supports "float32", "int8", "binary" and "ubinary"

# 5. Encode the corpus
full_corpus_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)
corpus_embeddings = quantize_embeddings(full_corpus_embeddings, precision=corpus_precision)
# NOTE: We can also pass "precision=..." to the encode method to quantize the embeddings directly,
# but we want to keep the full precision embeddings to act as a calibration dataset for quantizing
# the query embeddings. This is important only if you are using uint8 or int8 precision

# Initially, we don't have a usearch index yet, we can use semantic_search_usearch to create it
corpus_index = None
while True:
    # 7. Encode the queries using the full precision
    start_time = time.time()
    query_embeddings = model.encode(queries, normalize_embeddings=True)
    print(f"Encoding time: {time.time() - start_time:.6f} seconds")

    # 8. Perform semantic search using usearch
    results, search_time, corpus_index = semantic_search_usearch(
        query_embeddings,
        corpus_index=corpus_index,
        corpus_embeddings=corpus_embeddings if corpus_index is None else None,
        corpus_precision=corpus_precision,
        top_k=10,
        calibration_embeddings=full_corpus_embeddings,
        rescore=corpus_precision != "float32",
        rescore_multiplier=4,
        exact=True,
        output_index=True,
    )
    # This is a helper function to showcase how usearch can be used with quantized embeddings.
    # You must either provide the `corpus_embeddings` or the `corpus_index` usearch index, but not both.
    # In the first call we'll provide the `corpus_embeddings` and get the `corpus_index` back, which
    # we'll use in the next call. In practice, the index is stored in RAM or saved to disk, and not
    # recalculated for every query.

    # This function will 1) quantize the query embeddings to the same precision as the corpus embeddings,
    # 2) perform the semantic search using usearch, 3) rescore the results using the full precision embeddings,
    # and 4) return the results and the search time (and perhaps the usearch index).

    # `corpus_precision` must be the same as the precision used to quantize the corpus embeddings.
    # It is used to convert the query embeddings to the same precision as the corpus embeddings.
    # `top_k` determines how many results are returned for each query.
    # `rescore_multiplier` is a parameter for the rescoring step. Rather than searching for the top_k results,
    # we search for top_k * rescore_multiplier results and rescore the top_k results using the full precision embeddings.
    # So, higher values of rescore_multiplier will give better results, but will be slower.

    # `calibration_embeddings` is a set of embeddings used to calibrate the quantization of the query embeddings.
    # This is important only if you are using uint8 or int8 precision. In practice, this is used to calculate
    # the minimum and maximum values of each of the embedding dimensions, which are then used to determine the
    # quantization thresholds.

    # `rescore` determines whether to rescore the results using the full precision embeddings, if False & the
    # corpus is quantized, the results will be very poor. `exact` determines whether to use the exact search
    # or the approximate search method in usearch.

    # 9. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}")
        print("")

    # 10. Prompt for more queries
    queries = [input("Please enter a question: ")]
