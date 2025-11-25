# Paraphrase Mining

Paraphrase mining is the task of finding paraphrases (texts with identical / similar meaning) in a large corpus of sentences. In [Semantic Textual Similarity](../../../../docs/sentence_transformer/usage/semantic_textual_similarity.rst) we saw a simplified version of finding paraphrases in a list of sentences. The approach presented there used a brute-force approach to score and rank all pairs.

```{eval-rst}
However, as this has a quadratic runtime, it fails to scale to large (10,000 and more) collections of sentences. For larger collections, the :func:`~sentence_transformers.util.paraphrase_mining` function can be used::

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import paraphrase_mining

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Single list of sentences - Possible tens of thousands of sentences
    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]

    paraphrases = paraphrase_mining(model, sentences)

    for paraphrase in paraphrases[0:10]:
        score, i, j = paraphrase
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))

The :func:`~sentence_transformers.util.paraphrase_mining` accepts the following parameters:

.. autofunction:: sentence_transformers.util.paraphrase_mining

To optimize memory and computation time, paraphrase mining is performed in chunks, as specified by ``query_chunk_size`` and ``corpus_chunk_size``.
To be specific, only ``query_chunk_size * corpus_chunk_size`` pairs will be compared at a time, rather than ``len(sentences) * len(sentences)``. This is more time- and memory-efficient. Additionally, :func:`~sentence_transformers.util.paraphrase_mining` only considers the ``top_k`` best scores per sentences per chunk. You can experiment with this value as an efficiency-performance trade-off.

For example, for each sentence you will get only the one most relevant sentence in this script.

::

    paraphrases = paraphrase_mining(model, sentences, corpus_chunk_size=len(sentences), top_k=1)

The final key parameter is ``max_pairs``, which determines the maximum number of paraphrase pairs that the function returns. Usually, you get fewer pairs returned because the list is cleaned of duplicates, e.g., if it contains (A, B) and (B, A), then only one is returned.

.. note::
    
    If B is the most similar sentence for A, A is not necessarily the most similar sentence for B. So it can happen that the returned list contains entries like (A, B) and (B, C).
```
