"""Trace a text MaxSim ranking back to the tokens that produced it: semantic search, with similarity maps.

The image counterpart is heatmap.py, which overlays a heatmap on a page image. Text documents have no
patch grid to overlay, but the underlying quantity is the same: the similarity of every query token
against every document token. Because MaxSim sums the per-query-token maxima, that matrix decomposes
the ranking score exactly, so this script can rank a corpus (as semantic_search.py does) and then
attribute every point of the top hit's score to one query token and one document token.

Special tokens ([CLS], the [Q]/[D] markers, [SEP], and any query-expansion [MASK] tokens) are part of
the score too, and often absorb query tokens that have no lexical match in the document. They are
kept out of the table (that is what real_query_token_slice selects) and summarised in one line.
"""

from __future__ import annotations

import time

import torch
from datasets import load_dataset
from torch import Tensor

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.interpretability import real_query_token_slice


def highlight(text: str) -> str:
    """Invert the terminal's own foreground and background. Inverting rather than picking a colour
    keeps the highlight legible on light and dark themes alike."""
    return f"\033[7m{text}\033[0m"


def token_labels(model: MultiVectorEncoder, input_ids: Tensor) -> list[str]:
    """Readable labels for token ids: byte-level pieces decoded back to text, and the markers that
    tokenizers use to encode word boundaries (a leading space, ``##``) dropped."""
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    labels = (model.tokenizer.convert_tokens_to_string([token]).strip() for token in tokens)
    return [label.removeprefix("##") or "_" for label in labels]


def print_similarity_map(model: MultiVectorEncoder, query: str, document: str, score: float) -> None:
    # output_value=None returns the raw per-input features instead of just the embeddings: the token
    # ids that produced each embedding come along, so every row can be labelled with its token.
    query_features = model.encode_query([query], output_value=None)[0]
    document_features = model.encode_document([document], output_value=None)[0]

    # The document's attention_mask is MultiVectorMask's scoring mask, so skiplisted punctuation is
    # already excluded: masking both keys keeps ids and embeddings aligned with what MaxSim saw.
    document_mask = document_features["attention_mask"].bool()
    document_embeddings = document_features["token_embeddings"][document_mask].float().cpu()
    document_labels = token_labels(model, document_features["input_ids"][document_mask])

    query_slice = real_query_token_slice(model, query)
    query_embeddings = query_features["token_embeddings"][query_slice].float().cpu()
    query_tokens = token_labels(model, query_features["input_ids"][query_slice])

    # The similarity map: (query tokens, document tokens). Both sides are already L2-normalized by
    # the model's Normalize module, so these are cosine similarities.
    similarity_map = query_embeddings @ document_embeddings.T
    best_similarity, best_index = similarity_map.max(dim=1)

    matches = [document_labels[index] for index in best_index.tolist()]
    n_special = int(query_features["attention_mask"].sum()) - len(query_tokens)
    special_label = f"{n_special} special tokens"
    token_width = max(len(text) for text in [*query_tokens, "query token", special_label, "MaxSim score"])
    match_width = max(len(text) for text in [*matches, "best document token"])

    def row(name: str, match: str, value: float) -> None:
        print(f"  {name:<{token_width}}  {match:<{match_width}}  {value:7.4f}  {value / score:6.1%}")

    print(f"\n  {'query token':<{token_width}}  {'best document token':<{match_width}}      sim   share")
    for token, match, similarity in zip(query_tokens, matches, best_similarity.tolist()):
        row(token, match, similarity)
    print(f"  {'-' * (token_width + match_width + 21)}")
    row(special_label, "", score - best_similarity.sum().item())
    row("MaxSim score", "", score)

    # Each query token's maximum lands on one document token, so these counts are where the score came from.
    wins = torch.bincount(best_index, minlength=len(document_labels)).tolist()

    print(
        f"\n  Similarity map: {len(query_tokens)} query tokens over {len(document_labels)} "
        "document tokens, (n) = won n query tokens"
    )
    cells, width = [], 0
    for label, count in zip(document_labels, wins):
        text = label if count < 2 else f"{label}({count})"
        if width + len(text) > 96:
            print("   ", " ".join(cells))
            cells, width = [], 0
        cells.append(highlight(text) if count else text)
        width += len(text) + 1
    print("   ", " ".join(cells))


def main() -> None:
    # A small corpus: the (deduplicated) answer passages of 5000 Natural Questions rows.
    dataset = load_dataset("sentence-transformers/natural-questions", split="train[:5000]")
    corpus = list(dict.fromkeys(dataset["answer"]))

    model = MultiVectorEncoder("lightonai/LateOn")
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)

    def search(query: str) -> None:
        start = time.perf_counter()
        query_embeddings = model.encode_query([query], convert_to_tensor=True)
        scores = model.similarity(query_embeddings, corpus_embeddings)[0]
        top_scores, top_indices = scores.topk(3)
        search_time = (time.perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(f"Top 3 of {len(corpus)} documents by exhaustive MaxSim ({search_time:.1f}ms):")
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            print(f"  {score:.4f}  {corpus[index][:100]}")

        print_similarity_map(model, query, corpus[top_indices[0]], top_scores[0].item())

    search(dataset["query"][0])

    while True:
        try:
            query = input("\nPlease enter a question (or press Enter to quit): ").strip()
            # * "Where is the most wool produced?",
            # * "Where do hamsters live?",
            # * "What is the most common type of cheese in the world?"
        except EOFError:
            break
        if not query:
            break
        search(query)


if __name__ == "__main__":
    main()
