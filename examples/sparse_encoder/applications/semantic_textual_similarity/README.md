## Semantic Textual Similarity

For Semantic Textual Similarity (STS), we want to generate sparse embeddings for all texts involved and calculate the similarities between them. The text pairs with the highest similarity score are most semantically similar.

```{eval-rst}
.. sidebar:: Documentation

   1. :class:`SparseEncoder <sentence_transformers.sparse_encoder.SparseEncoder>`
   2. :meth:`SparseEncoder.encode <sentence_transformers.sparse_encoder.SparseEncoder.encode>`
   3. :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>`

::

    from sentence_transformers import SparseEncoder

    # Initialize the SPLADE model
    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

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
    - The dog plays in the garden   : 1.1750
    - The new movie is so great     : 24.0100
    - A woman watches TV            : 0.1358
    The cat sits outside
    - The dog plays in the garden   : 2.7264
    - The new movie is so great     : 0.6256
    - A woman watches TV            : 0.2129
    A man is playing guitar
    - The dog plays in the garden   : 7.5841
    - The new movie is so great     : 0.0316
    - A woman watches TV            : 1.5672

In this example, the :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>` method returns a 3x3 matrix with the respective cosine similarity scores for all possible pairs between ``embeddings1`` and ``embeddings2``.
```

### Similarity Calculation

```{eval-rst}
The similarity metric that is used is stored in the SparseEncoder instance under :attr:`SparseEncoder.similarity_fn_name <sentence_transformers.sparse_encoder.SparseEncoder.similarity_fn_name>`. Valid options are:

- ``SimilarityFunction.DOT_PRODUCT`` (a.k.a `"dot"`): Dot Product (**default**)
- ``SimilarityFunction.COSINE`` (a.k.a `"cosine"`): Cosine Similarity
- ``SimilarityFunction.EUCLIDEAN`` (a.k.a `"euclidean"`): Negative Euclidean Distance
- ``SimilarityFunction.MANHATTAN`` (a.k.a. `"manhattan"`): Negative Manhattan Distance

This value can be changed in a handful of ways:

1. By initializing the :class:`~sentence_transformers.sparse_encoder.SparseEncoder` instance with the desired similarity function::

    from sentence_transformers import SparseEncoder, SimilarityFunction

    model = SparseEncoder(
        "naver/splade-cocondenser-ensembledistil",
        similarity_fn_name=SimilarityFunction.COSINE,
    )

2. By setting the value directly on the :class:`~sentence_transformers.sparse_encoder.SparseEncoder` instance::

    from sentence_transformers import SparseEncoder, SimilarityFunction

    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
    model.similarity_fn_name = SimilarityFunction.COSINE


3. By setting the value under the ``"similarity_fn_name"`` key in the ``config_sentence_transformers.json`` file of a saved model. When you save a Sparse Encoder model, this value will be automatically saved as well.

The :class:`~sentence_transformers.sparse_encoder.SparseEncoder` class implements two methods to calculate the similarity between embeddings:

- :meth:`SparseEncoder.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>`: Calculates the similarity between all pairs of embeddings.
- :meth:`SparseEncoder.similarity_pairwise <sentence_transformers.sparse_encoder.SparseEncoder.similarity_pairwise>`: Calculates the similarity between embeddings in a pairwise fashion.

::

    from sentence_transformers import SparseEncoder, SimilarityFunction

    # Load a pretrained Sparse Encoder model
    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

    # Embed some sentences
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    embeddings = model.encode(sentences)

    similarities = model.similarity(embeddings, embeddings)
    print(model.similarity_fn_name)
    # => "dot"
    print(similarities)
    # tensor([[   35.629,     9.154,     0.098],
    #         [    9.154,    27.478,     0.019],
    #         [    0.098,     0.019,    29.553]])

    # Change the similarity function to Manhattan distance
    model.similarity_fn_name = SimilarityFunction.COSINE
    print(model.similarity_fn_name)
    # => "cosine"

    similarities = model.similarity(embeddings, embeddings)
    print(similarities)
    # tensor([[    1.000,     0.293,     0.003],
    #         [    0.293,     1.000,     0.001],
    #         [    0.003,     0.001,     1.000]])

```
