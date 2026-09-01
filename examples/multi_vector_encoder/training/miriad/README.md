# Domain-specific MultiVectorEncoder training on MIRIAD

[training_contrastive.py](training_contrastive.py) trains a ColBERT-style late-interaction model for
medical question -> passage retrieval on [MIRIAD](https://huggingface.co/datasets/tomaarsen/miriad-4.4M-split),
as an example of adapting a multi-vector model to a specific domain with plain contrastive learning:
no teacher scores or mined negatives are required, only (question, passage) pairs.

Because MIRIAD passages average ~940 tokens, the script doubles as a long-document late-interaction
example: the `max_length` training argument caps tokenization during training while the saved model
keeps the backbone's full context, and the evaluation calls run at full passage length without any
manual memory tuning, because MaxSim scoring packs documents under an element budget on its own.

Reference results (nDCG@10, deduplicated passage corpus per split, single RTX 3090, ~9.5 hours):

| Model | eval | test |
|---|---|---|
| [multivector-gte-modernbert-base-miriad](https://huggingface.co/tomaarsen/multivector-gte-modernbert-base-miriad) (this script) | **0.9827** | **0.9826** |
| [splade-modernbert-base-miriad](https://huggingface.co/tomaarsen/splade-modernbert-base-miriad) | 0.8609 | 0.8626 |
| pre-training baseline (warm-started backbone, random projection) | 0.9176 | |
