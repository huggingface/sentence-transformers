# Rerankers

```{eval-rst}
Reranker models are often :class:`~sentence_transformers.cross_encoder.CrossEncoder` models with 1 output class, i.e. given a pair of texts (query, answer), the model outputs one score. This score, either a float score that reasonably ranges between -10.0 and 10.0, or a score that's bound to 0...1, denotes to what extent the answer can help answer the query.

Many reranker models are trained on MS MARCO:

- `MS MARCO Pre-trained Cross Encoders <../../../../docs/cross_encoder/pretrained_models.html#ms-marco>`_
- `Cross Encoder > Training Examples > MS MARCO <../ms_marco/README.html>`_
```

But most likely, you will get the best results when training on your dataset. Because of this, this page includes some examples training scripts that you can adopt for your own data:

- **[training_gooaq_bce.py](training_gooaq_bce.py)**:
  ```{eval-rst}
  This example uses :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` on labeled pair data that was mined from the `GooAQ <https://huggingface.co/datasets/sentence-transformers/gooaq>`_ dataset using an efficient :class:`~sentence_transformer.SentenceTransformers`.

  The model is evaluated on subsets of `MS MARCO <https://huggingface.co/datasets/sentence-transformers/NanoMSMARCO-bm25>`_, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ via the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`. Additionally, it is evaluated on the performance gain when reranking the top 100 results from an efficient :class:`~sentence_transformer.SentenceTransformers` on the GooAQ development set.
  ```
- **[training_gooaq_cmnrl.py](training_gooaq_cmnrl.py)**:
  ```{eval-rst}
  This example uses :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss` on positive pair data loaded from the `GooAQ <https://huggingface.co/datasets/sentence-transformers/gooaq>`_ dataset.

  The model is evaluated on subsets of `MS MARCO <https://huggingface.co/datasets/sentence-transformers/NanoMSMARCO-bm25>`_, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ via the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`.
  ```
- **[training_gooaq_lambda.py](training_gooaq_lambda.py)**:
  ```{eval-rst}
  This example uses :class:`~sentence_transformers.cross_encoder.losses.LambdaLoss` on labeled list data that was mined from the `GooAQ <https://huggingface.co/datasets/sentence-transformers/gooaq>`_ dataset using an efficient :class:`~sentence_transformer.SentenceTransformers`.

  The model is evaluated on subsets of `MS MARCO <https://huggingface.co/datasets/sentence-transformers/NanoMSMARCO-bm25>`_, `NFCorpus <https://huggingface.co/datasets/sentence-transformers/NanoNFCorpus-bm25>`_, `NQ <https://huggingface.co/datasets/sentence-transformers/NanoNQ-bm25>`_ via the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`. Additionally, it is evaluated on the performance gain when reranking the top 100 results from an efficient :class:`~sentence_transformer.SentenceTransformers` on the GooAQ development set.
  ```
- **[training_nq_bce.py](training_nq_bce.py)**:
  ```{eval-rst}
  This example uses a near-identical training script as ``training_gooaq_bce.py``, except on the smaller `NQ (natural questions) <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_ dataset.
  ```

## BinaryCrossEntropyLoss

```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` is a very strong yet simple loss. Given pairs of texts (e.g. (query, answer) pairs), this loss uses the :class:`~sentence_transformers.cross_encoder.CrossEncoder` model to compute prediction scores. It compares these against the gold (or silver, a.k.a. determined with some model) labels, and computes a lower loss the better the model is doing.
```

## CachedMultipleNegativesRankingLoss

```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss` (a.k.a. InfoNCE with GradCache) is more complex than the common :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`. It accepts positive pairs (i.e. (query, answer) pairs) or triplets (i.e. (query, right_answer, wrong_answer) triplets), and will then randomly find ``num_negatives`` extra incorrect answers per query by taking answers from other questions in the batch. This is often referred to as "in-batch negatives".

The loss will then compute scores for all (query, answer) pairs, *including* the incorrect answers ones it just selected. The loss will then use a Cross Entropy Loss to ensure that the score of the (query, correct_answer) is higher than (query, wrong_answer) for all (randomly selected) wrong answers.

The :class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss` uses an approach called `GradCache <https://huggingface.co/papers/2101.06983>`_ to allow computing the scores in mini-batches without increasing the memory usage excessively. This loss is recommended over the "standard" :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss` (a.k.a. InfoNCE) loss, which does not have this clever mini-batching support and thus requires a lot of memory.

Experimentation with an ``activation_fn`` and ``scale`` is warranted for this loss. :class:`torch.nn.Sigmoid` with ``scale=10.0`` works okay, :class:`torch.nn.Identity`` with ``scale=1.0`` also works, and the `mGTE <https://huggingface.co/papers/2407.19669>`_ paper authors suggest using :class:`torch.nn.Tanh` with ``scale=10.0``.
```

## Inference

The [tomaarsen/reranker-ModernBERT-base-gooaq-bce](https://huggingface.co/tomaarsen/reranker-ModernBERT-base-gooaq-bce) model was trained with the first script. If you want to try out the model before training something yourself, feel free to use this script:

```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("tomaarsen/reranker-ModernBERT-base-gooaq-bce")

# Get scores for pairs of texts
pairs = [
    ["how to obtain a teacher's certificate in texas?", 'Some aspiring educators may be confused about the difference between teaching certification and teaching certificates. Teacher certification is another term for the licensure required to teach in public schools, while a teaching certificate is awarded upon completion of an academic program.'],
    ["how to obtain a teacher's certificate in texas?", '["Step 1: Obtain a Bachelor\'s Degree. One of the most important Texas teacher qualifications is a bachelor\'s degree. ... ", \'Step 2: Complete an Educator Preparation Program (EPP) ... \', \'Step 3: Pass Texas Teacher Certification Exams. ... \', \'Step 4: Complete a Final Application and Background Check.\']'],
    ["how to obtain a teacher's certificate in texas?", "Washington Teachers Licensing Application Process Official transcripts showing proof of bachelor's degree. Proof of teacher program completion at an approved teacher preparation school. Passing scores on the required examinations. Completed application for teacher certification in Washington."],
    ["how to obtain a teacher's certificate in texas?", 'Teacher education programs may take 4 years to complete after which certification plans are prepared for a three year period. During this plan period, the teacher must obtain a Standard Certification within 1-2 years. Learn how to get certified to teach in Texas.'],
    ["how to obtain a teacher's certificate in texas?", 'In Texas, the minimum age to work is 14. Unlike some states, Texas does not require juvenile workers to obtain a child employment certificate or an age certificate to work. A prospective employer that wants one can request a certificate of age for any minors it employs, obtainable from the Texas Workforce Commission.'],
]
scores = model.predict(pairs)
print(scores)
# [0.00121048 0.97105724 0.00536712 0.8632406  0.00168043]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "how to obtain a teacher's certificate in texas?",
    [
        "[\"Step 1: Obtain a Bachelor's Degree. One of the most important Texas teacher qualifications is a bachelor's degree. ... \", 'Step 2: Complete an Educator Preparation Program (EPP) ... ', 'Step 3: Pass Texas Teacher Certification Exams. ... ', 'Step 4: Complete a Final Application and Background Check.']",
        "Teacher education programs may take 4 years to complete after which certification plans are prepared for a three year period. During this plan period, the teacher must obtain a Standard Certification within 1-2 years. Learn how to get certified to teach in Texas.",
        "Washington Teachers Licensing Application Process Official transcripts showing proof of bachelor's degree. Proof of teacher program completion at an approved teacher preparation school. Passing scores on the required examinations. Completed application for teacher certification in Washington.",
        "Some aspiring educators may be confused about the difference between teaching certification and teaching certificates. Teacher certification is another term for the licensure required to teach in public schools, while a teaching certificate is awarded upon completion of an academic program.",
        "In Texas, the minimum age to work is 14. Unlike some states, Texas does not require juvenile workers to obtain a child employment certificate or an age certificate to work. A prospective employer that wants one can request a certificate of age for any minors it employs, obtainable from the Texas Workforce Commission.",
    ],
)
print(ranks)
# [
#     {'corpus_id': 0, 'score': 0.97105724},
#     {'corpus_id': 1, 'score': 0.8632406},
#     {'corpus_id': 2, 'score': 0.0053671156},
#     {'corpus_id': 4, 'score': 0.0016804343},
#     {'corpus_id': 3, 'score': 0.0012104829},
# ]
```
