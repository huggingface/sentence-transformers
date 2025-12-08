# MS MARCO

[MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) is a large dataset to train models for information retrieval. It consists of about 500k real search queries from Bing search engine with the relevant text passage that answers the query. This page shows how to **train** Cross Encoder models on this dataset so that it can be used for searching text passages given queries (key words, phrases or questions).

If you are interested in how to use these models, see [Application - Retrieve & Re-Rank](../../../sentence_transformer/applications/retrieve_rerank/README.md). There are **pre-trained models** available, which you can directly use without the need of training your own models. For more information, see [Pretrained Cross-Encoders for MS MARCO](../../../../docs/cross_encoder/pretrained_models.md#ms-marco).

## Cross Encoder

```{eval-rst}
A `Cross Encoder <../../applications/README.html>`_ accepts both a query and a possible relevant passage and returns a score denoting how relevant the passage is for the given query. Often times, a :class:`torch.nn.Sigmoid` is applied over the raw output prediction, casting it to a value between 0 and 1.
```

![CrossEncoder](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/CrossEncoder.png)

```{eval-rst}
:class:`~sentence_transformers.cross_encoder.CrossEncoder` models are often used for **re-ranking**: Given a list with possible relevant passages for a query, for example retrieved from a :class:`~sentence_transformers.SentenceTransformer` model / BM25 / Elasticsearch, the cross-encoder re-ranks this list so that the most relevant passages are the top of the result list. 
```

## Training Scripts

```{eval-rst}
We provide several training scripts with various loss functions to train a :class:`~sentence_transformers.cross_encoder.CrossEncoder` on MS MARCO.

In all scripts, the model is evaluated on subsets of `MS MARCO <https://huggingface.co/datasets/sentence-transformers/NanoMSMARCO-bm25>`_, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ via the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`.
```

- **[training_ms_marco_bce_preprocessed.py](training_ms_marco_bce_preprocessed.py)**:
  ```{eval-rst}
  This example uses :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` on a `pre-processed MS MARCO dataset <https://huggingface.co/datasets/sentence-transformers/msmarco>`_.
  ```
- **[training_ms_marco_bce.py](training_ms_marco_bce.py)**:
  ```{eval-rst}
  This example also uses the :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`, but now the dataset pre-processing into ``(query, answer)`` with ``label`` as 1 or 0 is done in the training script. 
  ```
- **[training_ms_marco_cmnrl.py](training_ms_marco_cmnrl.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`. The script applies dataset pre-processing into ``(query, answer, negative_1, negative_2, negative_3, negative_4, negative_5)``.
  ```
- **[training_ms_marco_listnet.py](training_ms_marco_listnet.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.ListNetLoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
  ```
- **[training_ms_marco_lambda.py](training_ms_marco_lambda.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` with the :class:`~sentence_transformers.cross_encoder.losses.NDCGLoss2PPScheme` loss scheme. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
  ```
- **[training_ms_marco_lambda_preprocessed.py](training_ms_marco_lambda_preprocessed.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` with the :class:`~sentence_transformers.cross_encoder.losses.NDCGLoss2PPScheme` loss scheme on a `pre-processed MS MARCO dataset <https://huggingface.co/datasets/sentence-transformers/msmarco>`_.
  ```
- **[training_ms_marco_lambda_hard_neg.py](training_ms_marco_lambda_hard_neg.py)**:
  ```{eval-rst}
  This example extends the above example by increasing the size of the training dataset by mining hard negatives with :func:`~sentence_transformers.util.mine_hard_negatives`.
  ```
- **[training_ms_marco_listmle.py](training_ms_marco_listmle.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.ListMLELoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
  ```
- **[training_ms_marco_plistmle.py](training_ms_marco_plistmle.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.PListMLELoss` with the default :class:`~sentence_transformers.cross_encoder.losses.PListMLELambdaWeight` position weighting. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
  ```
- **[training_ms_marco_ranknet.py](training_ms_marco_ranknet.py)**:
  ```{eval-rst}
  This example uses the :class:`~sentence_transformers.cross_encoder.losses.RankNetLoss`. The script applies dataset pre-processing into ``(query, [doc1, doc2, ..., docN])`` with ``labels`` as ``[score1, score2, ..., scoreN]``.
  ```

Out of these training scripts, I suspect that **[training_ms_marco_lambda_preprocessed.py](training_ms_marco_lambda_preprocessed.py)**, **[training_ms_marco_lambda_hard_neg.py](training_ms_marco_lambda_hard_neg.py)** or **[training_ms_marco_bce_preprocessed.py](training_ms_marco_bce_preprocessed.py)** produces the strongest model, as anecdotally `LambdaLoss` and `BinaryCrossEntropyLoss` are quite strong. It seems that `LambdaLoss` > `PListMLELoss` > `ListNetLoss` > `RankNetLoss` > `ListMLELoss` out of all learning to rank losses, but your milage may vary.

Additionally, you can also train with Distillation. See [Cross Encoder > Training Examples > Distillation](../distillation/README.md) for more details.

## Inference

You can perform inference using any of the [pre-trained CrossEncoder models for MS MARCO](../../../../docs/cross_encoder/pretrained_models.md#ms-marco) like so:

```python
from sentence_transformers import CrossEncoder

# 1. Load a pre-trained CrossEncoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# 2. Predict scores for a pair of sentences
scores = model.predict([
    ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
    ("How many people live in Berlin?", "Berlin is well known for its museums."),
])
# => array([ 8.607138 , -4.3200774], dtype=float32)

# 3. Rank a list of passages for a query
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
    "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
    "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
    "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
    "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
    "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
    "In 2015, the total labour force in Berlin was 1.85 million.",
    "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
    "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
]
ranks = model.rank(query, passages)

# Print the scores
print("Query:", query)
for rank in ranks:
    print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")
"""
Query: How many people live in Berlin?
8.92    The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.
8.61    Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
8.24    An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.
7.60    In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.
6.35    In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
5.42    Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.
3.45    In 2015, the total labour force in Berlin was 1.85 million.
0.33    Berlin is subdivided into 12 boroughs or districts (Bezirke).
-4.24   The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019
-4.32   Berlin is well known for its museums.
"""
```
