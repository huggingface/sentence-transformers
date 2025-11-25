# Training

This folder contains various examples to fine-tune `SentenceTransformers` for specific tasks.

For the beginning, I can recommend to have a look at the Semantic Textual Similarity ([STS](sts/)) or the Natural Language Inference ([NLI](nli/)) examples.

For the documentation how to train your own models, see [Training Overview](http://www.sbert.net/docs/sentence_transformer/training_overview.html).

## Training Examples

- [adaptive_layer](adaptive_layer/) - Examples to train models whose layers can be removed on the fly for faster inference.
- [avg_word_embeddings](avg_word_embeddings/) - This folder contains examples to train models based on classical word embeddings like GloVe. These models are extremely fast, but are a more inaccurate than transformers based models.
- [clip](clip/) - Examples to train CLIP image models.
- [cross-encoder](cross-encoder/) - Examples to train [CrossEncoder](http://www.sbert.net/docs/cross_encoder/usage/usage.html) models.
- [data_augmentation](data_augmentation/) Examples of how to apply data augmentation strategies to improve embedding models.
- [distillation](distillation/) - Examples to make models smaller, faster and lighter.
- [hpo](hpo/) - Examples with hyperparameter search to find the best hyperparameters for your task.
- [matryoshka](matryoshka/) - Examples with training embedding models whose embeddings can be truncated (allowing for faster search) with minimal performance loss.
- [ms_marco](ms_marco/) - Example training scripts for training on the MS MARCO information retrieval dataset.
- [multilingual](multilingual/) - Existent monolingual models can be extend to various languages ([paper](https://huggingface.co/papers/2004.09813)). This folder contains a step-by-step guide to extend existent models to new languages.
- [nli](nli/) - Natural Language Inference (NLI) data can be quite helpful to pre-train and fine-tune models to create meaningful sentence embeddings.
- [other](other/) - Various tiny examples for show-casing one specific training case.
- [paraphrases](paraphrases/) - Examples for training models capable of recognizing paraphrases, i.e. understand when texts have the same meaning despite using different words.
- [quora_duplicate_questions](quora_duplicate_questions/) - Quora Duplicate Questions is large set corpus with duplicate questions from the Quora community. The folder contains examples how to train models for duplicate questions mining and for semantic search.
- [sts](sts/) - The most basic method to train models is using Semantic Textual Similarity (STS) data. Here, we have a sentence pair and a score indicating the semantic similarity.
