# Visual Document Retrieval

ColPali-style late interaction is the state of the art for visual document retrieval: text queries are matched directly against document page images, with every query token compared against every image patch embedding. This skips OCR, layout parsing, and chunking entirely, so tables, figures, and charts are searched just like body text. This page shows two ways to build such a model: finetuning a pretrained ColVLM (e.g. to adapt it to your own document collection), or training a fresh late-interaction model from a multimodal embedding backbone.

There are pre-trained models available, which you can directly use without the need of training your own models. For more information, see [Pretrained Models > Visual Document Retrieval Models](../../../../docs/multi_vector_encoder/pretrained_models.md#visual-document-retrieval-models).

## Finetuning a pretrained ColVLM

**Training code: [finetuning_colqwen2.py](finetuning_colqwen2.py)**

```{eval-rst}
:class:`~sentence_transformers.MultiVectorEncoder` auto-detects transformers-native ``*ForRetrieval`` checkpoints such as `vidore/colqwen2-v1.0-hf <https://huggingface.co/vidore/colqwen2-v1.0-hf>`_, and image documents are encoded exactly like text: pass PIL images, file paths, or URLs to ``encode_document`` or use them as a document column in your training dataset. The `vidore/syntheticDocQA_energy_train <https://huggingface.co/datasets/vidore/syntheticDocQA_energy_train>`_ dataset (one of the domain components of the original ColPali training mix) provides ``(query, page_image)`` pairs from scraped energy-domain PDFs, and :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss` optimizes the MaxSim score of each pair to be higher than the scores against all other in-batch pages.

The 2.2B parameter backbone is too large for full finetuning on a 24GB consumer GPU, so the script freezes everything except the top layers of the language model and the multi-vector projection. Keeping the vision tower frozen is standard ColVLM practice (the ViDoRe Col* checkpoints themselves were trained with a frozen vision tower and LoRA adapters on the language model), and the frozen bottom layers need no activation storage or optimizer state, keeping the peak VRAM usage well below 24GB with bf16 weights. Evaluation uses :class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorInformationRetrievalEvaluator` with held-out queries against a corpus of held-out pages.
```

## Training from a multimodal embedding backbone

**Training code: [training_visual_document_retrieval.py](training_visual_document_retrieval.py)**

```{eval-rst}
You do not have to start from an existing Col* checkpoint: :class:`~sentence_transformers.MultiVectorEncoder` appends a freshly initialised 128-dim token projection to any multimodal embedding backbone, the same construction that turned VLMs into ColPali / ColQwen in the first place. This script builds a late-interaction model from `Qwen/Qwen3-VL-Embedding-2B <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>`_ and trains it on `(query, page image, hard negative image)` triplets from the `tomaarsen/llamaindex-vdr-en-train-preprocessed <https://huggingface.co/datasets/tomaarsen/llamaindex-vdr-en-train-preprocessed>`_ dataset. It mirrors the single-vector recipe from the `Training and Finetuning Multimodal Embedding & Reranker Models <https://huggingface.co/blog/train-multimodal-sentence-transformers>`_ blog post (`training_visual_document_retrieval.py <https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/multimodal/training_visual_document_retrieval.py>`_) on the same data, so the two families can be compared directly. :class:`~sentence_transformers.multi_vector_encoder.losses.CachedMultiVectorMultipleNegativesRankingLoss` keeps the effective batch large while only encoding a few samples at a time.

The whole backbone is finetuned, with two training arguments doing the heavy lifting: ``learning_rate_mapping`` gives the freshly initialised projection 10x the backbone learning rate, and ``max_grad_norm=30`` stops the fresh head's large gradients from letting default clipping throttle every update.
```
