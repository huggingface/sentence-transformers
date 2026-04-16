# Multimodal Training

```{eval-rst}
Sentence Transformer models can handle multimodal inputs (text, images, audio, and video), enabling cross-modal retrieval tasks such as text-to-image search or audio-to-text matching. The key enabler is the :class:`~sentence_transformers.base.modules.Transformer` module's automatic modality detection: it inspects the underlying model's processor to determine which modalities are supported, then handles preprocessing for each modality transparently.

This means multimodal training uses the exact same pipeline as text-only training: the same losses, the same trainer, and the same evaluation tools. The data collator handles multimodal preprocessing automatically.
```

## Supported Input Types

```{eval-rst}
You can check which modalities a model supports using the :attr:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.modalities` property and the :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.supports` method:
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-VL-Embedding-2B")
print(model.modalities)
# => ['text', 'image', 'message']
print(model.supports("image"))
# => True
print(model.supports("audio"))
# => False
```

```{eval-rst}
The :meth:`~sentence_transformers.sentence_transformer.model.SentenceTransformer.encode` method accepts a variety of input types depending on the model's supported modalities:
```

- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Audio**: file paths, numpy/torch arrays, dicts with `"array"` and `"sampling_rate"` keys, or `torchcodec.AudioDecoder` instances.
- **Video**: file paths, numpy/torch arrays, dicts with `"array"` and `"video_metadata"` keys, or `torchcodec.VideoDecoder` instances.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. `{"text": ..., "audio": ...}`. The keys must be `"text"`, `"image"`, `"audio"`, or `"video"`.
- **Chat messages**: a list of dicts with `"role"` and `"content"` keys for multimodal models that use an uncommon chat template to combine text and non-text inputs.

## Training

Training a multimodal model follows the same steps as training a text-only model. You can use any compatible loss function, and the trainer and data collator handle multimodal inputs without any special configuration. Datasets can mix modalities across columns, for example a "query" column containing text strings and a "document" column containing PIL images.

### Training Example: Document Screenshot Embedding

The [training_visual_document_retrieval.py](training_visual_document_retrieval.py) script finetunes [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) on query-document screenshot pairs for visual document retrieval. Here is how it works:

```{eval-rst}
**1. Load the model** with efficient training settings::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "Qwen/Qwen3-VL-Embedding-2B",
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"},
        processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
    )

The ``model_kwargs`` enable Flash Attention 2 and bfloat16 precision for faster training. The ``processor_kwargs`` control image resolution bounds; smaller ``max_pixels`` reduces memory usage at the cost of image detail.

**2. Load the dataset** from the `tomaarsen/llamaindex-vdr-en-train-preprocessed <https://huggingface.co/datasets/tomaarsen/llamaindex-vdr-en-train-preprocessed>`_ dataset, which contains text queries paired with document screenshot images::

    from datasets import load_dataset

    train_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "train", split="train")
    eval_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "eval", split="train")

**3. Define the loss function** using :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss` wrapped in :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss`. This combination trains the model for retrieval with in-batch negatives while producing embeddings that remain effective after truncation to smaller dimensions::

    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MatryoshkaLoss

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=1)
    loss = MatryoshkaLoss(model, loss, matryoshka_dims=[2048, 1536, 1024, 512, 256, 128, 64])

**4. Evaluate** using :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator` with text queries against an image corpus, measuring cross-modal retrieval performance::

    from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

    eval_evaluator = InformationRetrievalEvaluator(
        queries=eval_queries,       # dict of text queries
        corpus=eval_corpus,         # dict of PIL images
        relevant_docs=eval_relevant_docs,
        name="vdr-eval-hard",
    )

**5. Train** using the standard :class:`~sentence_transformers.sentence_transformer.trainer.SentenceTransformerTrainer`::

    from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=eval_evaluator,
    )
    trainer.train()

After training, the model can be evaluated at each Matryoshka dimension separately to measure the performance-efficiency tradeoff.
```

## References

```{eval-rst}
- :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss`
- :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss`
- :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator`
- `Training Overview <../../../../docs/sentence_transformer/training_overview.html>`_
- `Loss Overview <../../../../docs/sentence_transformer/loss_overview.html>`_
- `Pretrained Models <../../../../docs/sentence_transformer/pretrained_models.html>`_
```
