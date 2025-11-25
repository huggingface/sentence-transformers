# Publications

If you find this repository helpful, feel free to cite our publication [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://huggingface.co/papers/1908.10084):

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```

If you use one of the multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://huggingface.co/papers/2004.09813):

```bibtex
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
```

If you use the code for [data augmentation](https://github.com/huggingface/sentence-transformers/tree/main/examples/sentence_transformer/training/data_augmentation), feel free to cite our publication [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://huggingface.co/papers/2010.08240):

```bibtex
@inproceedings{thakur-2020-AugSBERT,
    title = "Augmented {SBERT}: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks",
    author = "Thakur, Nandan and Reimers, Nils and Daxenberger, Johannes  and Gurevych, Iryna",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = "6",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2010.08240",
    pages = "296--310",
}
```

If you use the models for [MS MARCO](pretrained-models/msmarco-v2.md), feel free to cite the paper: [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes](https://huggingface.co/papers/2012.14210)

```bibtex
@inproceedings{reimers-2020-Curse_Dense_Retrieval,
    title = "The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes",
    author = "Reimers, Nils  and Gurevych, Iryna",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = "8",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2012.14210",
    pages = "605--611",
}
```

When you use the unsupervised learning example, please have a look at: [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning](https://huggingface.co/papers/2104.06979):

```bibtex
@inproceedings{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and Gurevych, Iryna", 
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "671--688",
    url = "https://arxiv.org/abs/2104.06979",
}
```

When you use the GenQ learning example, please have a look at: [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://huggingface.co/papers/2104.08663):

```bibtex
@inproceedings{thakur-2021-BEIR,
    title = "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
    author = {Thakur, Nandan and Reimers, Nils and R{\"{u}}ckl{\'{e}}, Andreas and Srivastava, Abhishek and Gurevych, Iryna}, 
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) - Datasets and Benchmarks Track (Round 2)},
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2104.08663",
}
```

When you use GPL, please have a look at: [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval](https://huggingface.co/papers/2112.07577):

```bibtex
@inproceedings{wang-2021-GPL,
    title = "GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval",
    author = "Wang, Kexin and Thakur, Nandan and Reimers, Nils and Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2112.07577",
    month = "12",
    year = "2021",
    url = "https://arxiv.org/abs/2112.07577",
}
```

**Repositories using SentenceTransformers**

- **[haystack](https://github.com/deepset-ai/haystack)** - Neural Search / Q&A
- **[Top2Vec](https://github.com/ddangelov/Top2Vec)** - Topic modeling
- **[txtai](https://github.com/neuml/txtai)** - AI-powered search engine
- **[BERTTopic](https://github.com/MaartenGr/BERTopic)** - Topic model using SBERT embeddings
- **[KeyBERT](https://github.com/MaartenGr/KeyBERT)** - Key phrase extraction using SBERT
- **[contextualized-topic-models](https://github.com/MilaNLProc/contextualized-topic-models)** - Cross-Lingual Topic Modeling
- **[covid-papers-browser](https://github.com/gsarti/covid-papers-browser)** - Semantic Search for Covid-19 papers
- **[backprop](https://github.com/backprop-ai/backprop)** - Natural Language Engine that makes using state-of-the-art language models easy, accessible and scalable.

**SentenceTransformers in Articles**

In the following you find a (selective) list of articles / applications using SentenceTransformers to do amazing stuff. Feel free to contact me (info@nils-reimers.de) to add you application here.

- **December 2021 - [Sentence Transformer Fine-Tuning (SetFit): Outperforming GPT-3 on few-shot Text-Classification while being 1600 times smaller](https://towardsdatascience.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e?gi=4bdbaff416e3)**
- **October 2021: [Natural Language Processing (NLP) for Semantic Search](https://www.pinecone.io/learn/nlp)**
- **January 2021 - [Advance BERT model via transferring knowledge from Cross-Encoders to Bi-Encoders](https://resources.experfy.com/ai-ml/bert-model-transferring-knowledge-cross-encoders-bi-encoders/)**
- **November 2020 - [How to Build a Semantic Search Engine With Transformers and Faiss](https://towardsdatascience.com/how-to-build-a-semantic-search-engine-with-transformers-and-faiss-dcbea307a0e8)**
- **October 2020 - [Topic Modeling with BERT](https://medium.com/data-science/topic-modeling-with-bert-779f7db187e6)**
- **September 2020 - [Elastic Transformers -
  Making BERT stretchy - Scalable Semantic Search on a Jupyter Notebook](https://medium.com/@mihail.dungarov/elastic-transformers-ae011e8f5b88)**
- **July 2020 - [Simple Sentence Similarity Search with SentenceBERT](https://laptrinhx.com/simple-sentence-similarity-search-with-sentencebert-800684405/?fbclid=IwAR0rxdYS2DBGuHhijIRO_lsXqGc9BbjtDA-dDQM5Ng_StahT9xrHdRZuP9M)**
- **May 2020 - [HN Time Machine: finally some Hacker News history!](https://peltarion.com/blog/applied-ai/hacker-news-time-machine)**
- **May 2020 - [A complete guide to transfer learning from English to other Languages using Sentence Embeddings BERT Models](https://medium.com/data-science/a-complete-guide-to-transfer-learning-from-english-to-other-languages-using-sentence-embeddings-8c427f8804a9)**
- **March 2020 - [Building a k-NN Similarity Search Engine using Amazon Elasticsearch and SageMaker](https://medium.com/data-science/building-a-k-nn-similarity-search-engine-using-amazon-elasticsearch-and-sagemaker-98df18d883bd)**
- **February 2020 - [Semantic Search Engine with Sentence BERT](https://medium.com/@evergreenllc2020/semantic-search-engine-with-s-abbfb3cd9377)**

**SentenceTransformers used in Research**

SentenceTransformers is used in hundreds of research projects. For a list of publications, see [Google Scholar](https://scholar.google.com/scholar?oi=bibs&hl=de&cites=12599223809118664426) or [Semantic Scholar](https://www.semanticscholar.org/paper/Sentence-BERT%3A-Sentence-Embeddings-using-Siamese-Reimers-Gurevych/93d63ec754f29fa22572615320afe0521f7ec66d).
