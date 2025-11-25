# Text Summarization

SentenceTransformers can be used for (extractive) text summarization: The document is broken down into sentences and embedded by SentenceTransformers. Then, we can compute the cosine similarity across all possible sentence combinations.

We then use [LexRank](https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf) to find the most central sentences in the document. These central sentences form a good basis for a summarization of the document.

An example is shown in [text-summarization.py](text-summarization.py)
