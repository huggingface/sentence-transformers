# NLI Models
Conneau et al., 2017, show in the InferSent-Paper ([Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://huggingface.co/papers/1705.02364)) that training on Natural Language Inference (NLI) data can produce universal sentence embeddings.

The datasets labeled sentence pairs with the labels *entail*, *contradict*, and *neutral*. For both sentences, we compute a sentence embedding. These two embeddings are concatenated and passed to softmax classifier to derive the final label.

As shown, this produces sentence embeddings that can be used for various use cases like clustering or semantic search.

# Datasets
We train the models on the [SNLI](https://nlp.stanford.edu/projects/snli/) and on the [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) dataset. We call the combination of the two datasets AllNLI.

For a training example, see [examples/sentence_transformer/training/nli/training_nli.py](../../examples/sentence_transformer/training/nli/training_nli.py). 

# Pretrained models
We provide the various pre-trained models. The performance was evaluated on the test set of the STS benchmark dataset ([docs](https://web.archive.org/web/20231128064114/http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark), [dataset](https://huggingface.co/datasets/sentence-transformers/stsb)) using Spearman rank correlation.

[» Full List of NLI & STS Models](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0)

# Performance Comparison
Here are the performances on the STS benchmark for other sentence embeddings methods. They were also computed by using cosine-similarity and Spearman rank correlation:
- Avg. GloVe embeddings:  58.02 
- BERT-as-a-service avg. embeddings:  46.35 
- BERT-as-a-service CLS-vector: 16.50 
- InferSent - GloVe: 68.03 
- Universal Sentence Encoder: 74.92

# Applications
This model works well in accessing the coarse-grained similarity between sentences. For application examples, see [semantic_textual_similarity](../sentence_transformer/usage/semantic_textual_similarity.rst) and [semantic search](../../examples/sentence_transformer/applications/semantic-search/README.md).