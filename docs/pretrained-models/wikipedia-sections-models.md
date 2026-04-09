# Wikipedia Sections Models
The `wikipedia-sections-models` implement the idea from Ein Dor et al., 2018, [Learning Thematic Similarity Metric Using Triplet Networks](https://aclweb.org/anthology/P18-2009).

It was trained with a triplet-loss: The anchor and the positive example were sentences from the same section from an wikipedia article, for example, from the History section of the London article. The negative example came from a different section from the same article, for example, from the Education section of the London article.

# Dataset
We use dataset from Ein Dor et al., 2018, [Learning Thematic Similarity Metric Using Triplet Networks](https://aclweb.org/anthology/P18-2009).

See [examples/sentence_transformer/training/other/training_wikipedia_sections.py](../../examples/sentence_transformer/training/other/training_wikipedia_sections.py) for how to train on this dataset.


# Pre-trained models
We provide the following pre-trained models:

- **bert-base-wikipedia-sections-mean-tokens**: 80.42% accuracy on test set.

You can use them in the following way:
```
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/bert-base-wikipedia-sections-mean-tokens")
```

# Performance Comparison
Performance (accuracy) reported by Dor et al.:
- mean-vectors: 0.65
- skip-thoughts-CS: 0.615
- skip-thoughts-SICK: 0.547
- triplet-sen: 0.74


# Applications
The models achieve a rather low performance on the STS benchmark dataset. The reason for this is the training objective: An anchor, a positive and a negative example are presented. The network must only learn to differentiate what the positive and what the negative example is by ensuring that the negative example is further away from the anchor than the positive example.

However, it does not matter how far the negative example is away, it can be little or really far away. This makes this model rather bad for deciding if a pair is somewhat similar. It learns only to recognize similar pairs (high scores) and dissimilar pairs (low scores).

However, this model works well for **fine-grained clustering**. 

