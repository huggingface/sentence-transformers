# Applications

SentenceTransformers can be used for various use-cases. In these folders, you find several example scripts that show case how SentenceTransformers can be used

## Computing Embeddings

The [computing-embeddings](computing-embeddings/) folder contains examples how to compute sentence embeddings using SentenceTransformers.

## Clustering

The [clustering](clustering/) folder shows how SentenceTransformers can be used for text clustering, i.e., grouping sentences together based on their similarity.

## Cross-Encoder

SentenceTransformers also support training and inference of [Cross-Encoders](cross-encoder/). There, two sentences are presented simultaneously to the transformer network and a score (0...1) is derived indicating the similarity or a label.

## Parallel Sentence Mining

The [parallel-sentence-mining](parallel-sentence-mining/) folder contains examples of how parallel (translated) sentences can be found in two corpora of different languages. For example, you take the English and the Spanish Wikipedia and the script finds and returns all translated English-Spanish sentence pairs.

## Paraphrase Mining

The [paraphrase-mining](paraphrase-mining/) folder contains examples to find all paraphrase sentences in a large set of sentences. The example can be used to find e.g. duplicate questions or duplicate sentences in a set of Millions of questions / sentences.

## Semantic Search

The [semantic-search](semantic-search/) folder shows examples for semantic search: Given a sentence, find in a large collection semantically similar sentences.

## Retrieve & Rerank

The [retrieve_rerank](retrieve_rerank/) folder shows how to combine a bi-encoder for semantic search retrieval and a more powerful re-ranking stage with a cross-encoder.

## Image Search

The [image-search](image-search/) folder shows how to use the image&text-models, which can map images and text to the same vector space. This allows for an image search given a user query.

## Text Summarization

The [text-summarization](text-summarization/) folder shows how SentenceTransformers can be used for extractive summarization: Give a long document, find the k sentences that give a good and short summary of the content.

## Compilation

The [compilation](compilation/) folder shows how to speed up batch-1 CPU-bound inference on GPU by compiling the model into CUDA graphs, padding inputs to a small set of token buckets so a fixed set of compiled graphs covers all sequence lengths.
