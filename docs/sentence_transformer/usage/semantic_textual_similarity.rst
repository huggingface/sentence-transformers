Semantic Textual Similarity
===========================

For Semantic Textual Similarity (STS), we want to produce embeddings for all texts involved and calculate the similarities between them. The text pairs with the highest similarity score are most semantically similar. See also the `Computing Embeddings <../../../examples/sentence_transformer/applications/computing-embeddings/README.html>`_ documentation for more advanced details on getting embedding scores.

.. sidebar:: Documentation

   1. :class:`SentenceTransformer <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
   2. :meth:`SentenceTransformer.encode <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`
   3. :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`

::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Two lists of sentences
    sentences1 = [
        "The new movie is awesome",
        "The cat sits outside",
        "A man is playing guitar",
    ]

    sentences2 = [
        "The dog plays in the garden",
        "The new movie is so great",
        "A woman watches TV",
    ]

    # Compute embeddings for both lists
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)

    # Output the pairs with their score
    for idx_i, sentence1 in enumerate(sentences1):
        print(sentence1)
        for idx_j, sentence2 in enumerate(sentences2):
            print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")

.. code-block:: text
    :emphasize-lines: 3

    The new movie is awesome
    - The dog plays in the garden   : 0.0543
    - The new movie is so great     : 0.8939
    - A woman watches TV            : -0.0502
    The cat sits outside
    - The dog plays in the garden   : 0.2838
    - The new movie is so great     : -0.0029
    - A woman watches TV            : 0.1310
    A man is playing guitar
    - The dog plays in the garden   : 0.2277
    - The new movie is so great     : -0.0136
    - A woman watches TV            : -0.0327

In this example, the :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>` method returns a 3x3 matrix with the respective cosine similarity scores for all possible pairs between ``embeddings1`` and ``embeddings2``.

Similarity Calculation
----------------------

The similarity metric that is used is stored in the SentenceTransformer instance under :attr:`SentenceTransformer.similarity_fn_name <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity_fn_name>`. Valid options are:

- ``SimilarityFunction.COSINE`` (a.k.a `"cosine"`): Cosine Similarity (**default**)
- ``SimilarityFunction.DOT_PRODUCT`` (a.k.a `"dot"`): Dot Product
- ``SimilarityFunction.EUCLIDEAN`` (a.k.a `"euclidean"`): Negative Euclidean Distance
- ``SimilarityFunction.MANHATTAN`` (a.k.a. `"manhattan"`): Negative Manhattan Distance

This value can be changed in a handful of ways:

1. By initializing the SentenceTransformer instance with the desired similarity function::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)

2. By setting the value directly on the SentenceTransformer instance::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.similarity_fn_name = SimilarityFunction.DOT_PRODUCT

3. By setting the value under the ``"similarity_fn_name"`` key in the ``config_sentence_transformers.json`` file of a saved model. When you save a Sentence Transformer model, this value will be automatically saved as well.

Sentence Transformers implements two methods to calculate the similarity between embeddings:

- :meth:`SentenceTransformer.similarity <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity>`: Calculates the similarity between all pairs of embeddings.
- :meth:`SentenceTransformer.similarity_pairwise <sentence_transformers.sentence_transformer.model.SentenceTransformer.similarity_pairwise>`: Calculates the similarity between embeddings in a pairwise fashion.

::

    from sentence_transformers import SentenceTransformer, SimilarityFunction

    # Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed some sentences
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])

    # Change the similarity function to Manhattan distance
    model.similarity_fn_name = SimilarityFunction.MANHATTAN
    print(model.similarity_fn_name)
    # => "manhattan"

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[ -0.0000, -12.6269, -20.2167],
    #         [-12.6269,  -0.0000, -20.1288],
    #         [-20.2167, -20.1288,  -0.0000]])

.. note::

   If a Sentence Transformer instance ends with a :class:`~sentence_transformers.sentence_transformer.models.Normalize` module, then it is sensible to choose the "dot" metric instead of "cosine".

   Dot product on normalized embeddings is equivalent to cosine similarity, but "cosine" will re-normalize the embeddings again. As a result, the "dot" metric will be faster than "cosine".

If you want find the highest scoring pairs in a long list of sentences, have a look at `Paraphrase Mining <../../../examples/sentence_transformer/applications/paraphrase-mining/README.html>`_.
