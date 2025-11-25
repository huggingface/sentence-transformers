# Training

This folder contains various examples to fine-tune `CrossEncoder` models for specific tasks.

For the beginning, I can recommend to have a look at the [MS MARCO](ms_marco/) examples.

For the documentation how to train your own models, see [Training Overview](http://www.sbert.net/docs/cross_encoder/training_overview.html).

## Training Examples

- [distillation](distillation/) - Examples to make models smaller, faster and lighter.
- [ms_marco](ms_marco/) - Numerous example training scripts for training on the MS MARCO information retrieval dataset.
- [nli](nli/) - Natural Language Inference (NLI) data involves pair classification using the "contradiction", "entailment", and "neutral" classes.
- [quora_duplicate_questions](quora_duplicate_questions/) - Quora Duplicate Questions is large set corpus with duplicate questions from the Quora community. The folder contains examples how to train models for duplicate questions mining and for semantic search.
- [rerankers](rerankers/) - Example training scripts for training on generic information retrieval datasets.
- [sts](sts/) - The most basic method to train models is using Semantic Textual Similarity (STS) data. Here, we have a sentence pair and a score indicating the semantic similarity.
