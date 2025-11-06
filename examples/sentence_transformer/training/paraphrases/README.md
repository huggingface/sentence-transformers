# Paraphrase Data

```{eval-rst}
In our paper `Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation <https://huggingface.co/papers/2004.09813>`_, we showed that paraphrase data together with :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` is a powerful combination to learn sentence embeddings models. Read `NLI > MultipleNegativesRankingLoss <../nli/README.html#multiplenegativesrankingloss>`_ for more information on this loss function.
```

The [training.py](training.py) script loads various datasets from the [Dataset Overview](../../../../docs/sentence_transformer/dataset_overview.md#pre-existing-datasets). We construct batches by sampling examples from the respective dataset. So far, examples are not mixed between the datasets, i.e., a batch consists only of examples from a single dataset.

As the dataset sizes are quite different in size, we perform [round-robin sampling](../../../../docs/package_reference/sentence_transformer/sampler.md#sentence_transformers.training_args.MultiDatasetBatchSamplers) to train using the same amount of batches from each dataset.

## Pre-Trained Models

Have a look at [pre-trained models](../../../../docs/sentence_transformer/pretrained_models.md) to view all models that were trained on these paraphrase datasets.

- [paraphrase-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- [paraphrase-distilroberta-base-v2](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v2) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions,flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions, wiki-atomic-edits
- [paraphrase-distilroberta-base-v1](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1) - Trained on the following datasets: AllNLI, sentence-compression, SimpleWiki, altlex, quora_duplicates, wiki-atomic-edits, wiki-split
- [paraphrase-xlm-r-multilingual-v1](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1) - Multilingual version of paraphrase-distilroberta-base-v1, trained on parallel data for 50+ languages. (Teacher: [paraphrase-distilroberta-base-v1](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1), Student: [xlm-r-base](https://huggingface.co/FacebookAI/xlm-roberta-base))
