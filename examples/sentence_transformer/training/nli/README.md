# Natural Language Inference

Given two sentence (premise and hypothesis), Natural Language Inference (NLI) is the task of deciding if the premise entails the hypothesis, if they are contradiction, or if they are neutral. Commonly used NLI dataset are [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli).

[Conneau et al.](https://huggingface.co/papers/1705.02364) showed that NLI data can be quite useful when training Sentence Embedding methods. We also found this in our [Sentence-BERT-Paper](https://huggingface.co/papers/1908.10084) and often use NLI as a first fine-tuning step for sentence embedding methods.

To train on NLI, see the following example files:

1. **[training_nli.py](training_nli.py)**:
   ```{eval-rst}
   This example uses :class:`~sentence_transformers.losses.SoftmaxLoss` as described in the original [Sentence Transformers paper](https://huggingface.co/papers/1908.10084).
   ```
1. **[training_nli_v2.py](training_nli_v2.py)**:
   ```{eval-rst}
   The :class:`~sentence_transformers.losses.SoftmaxLoss` as used in our original SBERT paper does not yield optimal performance. A better loss is :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`, where we provide pairs or triplets. In this script, we provide a triplet of the format: (anchor, entailment_sentence, contradiction_sentence). The NLI data provides such triplets. The :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` yields much higher performances and is more intuitive than :class:`~sentence_transformers.losses.SoftmaxLoss`. We have used this loss to train the paraphrase model in our `Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation <https://huggingface.co/papers/2004.09813>`_ paper.
   ```
1. **[training_nli_v3.py](training_nli_v3.py)**
   ```{eval-rst}
   Following the `GISTEmbed <https://huggingface.co/papers/2402.16829>`_ paper, we can modify the in-batch negative selection from :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` using a guiding model. Candidate negative pairs are ignored during training if the guiding model considers the pair to be too similar. In practice, the :class:`~sentence_transformers.losses.GISTEmbedLoss` tends to produce a stronger training signal than :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` at the cost of some training overhead for running inference on the guiding model.
   ```

```{eval-rst}
You can also train and use :class:`~sentence_transformers.cross_encoder.CrossEncoder` models for this task. See `Cross Encoder > Training Examples > Natural Language Inference <../../../cross_encoder/training/nli/README.html>`_ for more details.
```

## Data

We combine [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) and [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) into a dataset we call [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli). These two datasets contain sentence pairs and one of three labels: entailment, neutral, contradiction:

| Sentence A (Premise) | Sentence B (Hypothesis) | Label |
| --- | --- | --- |
| A soccer game with multiple males playing. | Some men are playing a sport. | entailment |
| An older and younger man smiling. | Two men are smiling and laughing at the cats playing on the floor. | neutral |
| A man inspects the uniform of a figure in some East Asian country. | The man is sleeping. | contradiction |

We format AllNLI in a few different subsets, compatible with different loss functions. See for example the [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet).

## SoftmaxLoss

```{eval-rst}
`Conneau et al. <https://huggingface.co/papers/1705.02364>`_ described how a softmax classifier on top of a `siamese network <https://en.wikipedia.org/wiki/Siamese_neural_network>`_ can be used to learn meaningful sentence representation. We can achieve this by using :class:`~sentence_transformers.losses.SoftmaxLoss`:
```

<img src="https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/SBERT_SoftmaxLoss.png" alt="SBERT SoftmaxLoss" width="250"/>

We pass the two sentences through our SentenceTransformer model and get the sentence embeddings *u* and *v*. We then concatenate *u*, *v* and *|u-v|* to form one long vector. This vector is then passed to a softmax classifier, which predicts our three classes (entailment, neutral, contradiction).

This setup learns sentence embeddings that can later be used for wide variety of tasks.

## MultipleNegativesRankingLoss

```{eval-rst}
That the :class:`~sentence_transformers.losses.SoftmaxLoss` with NLI data produces (relatively) good sentence embeddings is rather coincidental. The :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` is much more intuitive and produces significantly better sentence representations.
```

The training data for MultipleNegativesRankingLoss consists of sentence pairs [(a<sub>1</sub>, b<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>)] where we assume that (a<sub>i</sub>, b<sub>i</sub>) are similar sentences and (a<sub>i</sub>, b<sub>j</sub>) are dissimilar sentences for i != j. The minimizes the distance between (a<sub>i</sub>, b<sub>i</sub>) while it simultaneously maximizes the distance (a<sub>i</sub>, b<sub>j</sub>) for all i != j. For example, in the following picture:

<img src="https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/MultipleNegativeRankingLoss.png" alt="SBERT MultipleNegativeRankingLoss" width="350"/>

The distance between (a<sub>1</sub>, b<sub>1</sub>) is reduced, while the distance between (a<sub>1</sub>, b<sub>2...5</sub>) will be increased. The same is done for a<sub>2</sub>, ..., a<sub>5</sub>.

```{eval-rst}
Using :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` with NLI is rather easy: We define sentences that have an *entailment* label as positive pairs. E.g, we have pairs like (*"A soccer game with multiple males playing."*, *"Some men are playing a sport."*) and want that these pairs are close in vector space. The `pair subset of AllNLI <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair>`_ has been prepared in this format.
```

### MultipleNegativesRankingLoss with Hard Negatives

We can further improve MultipleNegativesRankingLoss by providing triplets rather than pairs: [(a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>), ..., (a<sub>n</sub>, b<sub>n</sub>, c<sub>n</sub>)]. The samples for c<sub>i</sub> are so-called hard-negatives: On a lexical level, they are similar to a<sub>i</sub> and b<sub>i</sub>, but on a semantic level, they mean different things and should not be close to a<sub>i</sub> in the vector space.

For NLI data, we can use the contradiction-label to create such triplets with a hard negative. So our triplets look like this:
("*A soccer game with multiple males playing."*, *"Some men are playing a sport."*, *"A group of men playing a baseball game."*). We want the sentences *"A soccer game with multiple males playing."* and *"Some men are playing a sport."* to be close in the vector space, while there should be a larger distance between *"A soccer game with multiple males playing."* and "*A group of men playing a baseball game."*. The [triplet subset of AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet) has been prepared in this format.

### GISTEmbedLoss

```{eval-rst}

:class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` can be extended even further by recognizing that the in-batch negative sampling as shown in `this example <#multiplenegativesrankingloss>`_ is a bit flawed. In particular, we automatically assume that the pairs (a\ :sub:`1`\ , b\ :sub:`2`\ ), ..., (a\ :sub:`1`\ , b\ :sub:`n`\ ) are negative, but that does not strictly have to be true.

To address this, :class:`~sentence_transformers.losses.GISTEmbedLoss` uses a Sentence Transformer model to guide the in-batch negative sample selection. In particular, if the guide model considers the similarity of (a\ :sub:`1`\ , b\ :sub:`n`\ ) to be larger than (a\ :sub:`1`\ , b\ :sub:`1`\ ), then the (a\ :sub:`1`\ , b\ :sub:`n`\ ) pair is considered a false negative and consequently ignored in the training process. In essence, this results in higher quality training data for the model.
```
