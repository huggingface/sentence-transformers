# MS MARCO - Multilingual Training

This folder demonstrates how to train a multi-lingual SBERT model for [semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html) / [information retrieval](https://www.sbert.net/examples/applications/retrieve_rerank/README.html).

As dataset, we use the [MS Marco Passage Ranking dataset](https://github.com/microsoft/MSMARCO-Passage-Ranking). It is a large dataset consisting of search queries from Bing search engine with the relevant text passage that answers the query.

Sadly this dataset is only available in English. As there are no large, multi-lingual datasets available suitable to train a semantic search model, we will use **machine translation** to translate the training data.

## Translating Data

We will translate the queries and the passages using [EasyNMT](https://github.com/UKPLab/EasyNMT), which provides state-of-the-art machine translation to 150+ languages.

Then, we will use [Multilingual Knowledge Distillation](https://www.sbert.net/examples/training/multilingual/README.html) and transform the English model trained on MS MARCO to a multi-lingual model.
