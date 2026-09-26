# Multimodal Training

```{eval-rst}
.. seealso::
   See the `Multimodal Embedding & Reranker Models <https://huggingface.co/blog/multimodal-sentence-transformers>`_ blogpost for an inference walkthrough, and the `Training and Finetuning Multimodal Embedding & Reranker Models <https://huggingface.co/blog/train-multimodal-sentence-transformers>`_ blogpost for a full Visual Document Retrieval training example built on the script described on this page.
```

```{eval-rst}
Sentence Transformer models can handle multimodal inputs (text, images, audio, and video), enabling cross-modal retrieval tasks such as text-to-image search or audio-to-text matching. The key enabler is the :class:`~sentence_transformers.base.modules.Transformer` module's automatic modality detection: it inspects the underlying model's processor to determine which modalities are supported, then handles preprocessing for each modality transparently.

This means multimodal training uses the exact same pipeline as text-only training: the same losses, the same trainer, and the same evaluation tools. The data collator handles multimodal preprocessing automatically.
```

## Supported Input Types

Use {attr}`model.modalities <sentence_transformers.sentence_transformer.model.SentenceTransformer.modalities>` and {meth}`model.supports() <sentence_transformers.sentence_transformer.model.SentenceTransformer.supports>` to check modality support. See [Input Formats](../../../../docs/input_formats.rst) for accepted representations, metadata, and examples of combining modalities.

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

## Experimental: preprocess images per GradCache mini-batch

[training_lazy_images.py](training_lazy_images.py) is a single-device Trainer example for
[issue #3991](https://github.com/huggingface/sentence-transformers/issues/3991). It keeps text and
local image paths in the outer training batch, opens images only for the current mini-batch,
and calls the model's existing processor. The same images are reopened and preprocessed during
GradCache's backward replay. No processed image batch is cached between the two passes.

Enable this experimental path with `SentenceTransformerTrainingArguments(lazy_preprocessing=True)`
and `CachedMultipleNegativesRankingLoss`. The default collator keeps raw inputs and resolves the
usual per-column/per-dataset prompts and Router tasks. GradCache calls the existing model processor
on each mini-batch, then moves only that mini-batch to the model's device. The default remains
`False`, preserving eager preprocessing.

```python
args = SentenceTransformerTrainingArguments(
    output_dir="output",
    per_device_train_batch_size=64,
    lazy_preprocessing=True,
)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=4),
)
trainer.train()
```

The dataset should contain text strings and image references such as `{"image": "/path/to/image.jpg"}`.
Use the input format supported by the model's processor. A `datasets.Image` column with automatic
decoding enabled loads images before collation, so use path dictionaries to defer image decoding.

Create a JSONL file with one positive text/image pair per line. Image paths are relative to the
JSONL file (absolute paths also work):

```json
{"query": "A dog running on grass", "image": "images/dog.jpg"}
{"query": "A page containing a revenue chart", "image": "images/chart.jpg"}
```

Choose a multimodal model that accepts text and image inputs, then run:

```bash
python training_lazy_images.py \
    --model /path/to/model --data pairs.jsonl --output output-lazy \
    --batch-size 64 --mini-batch-size 4 --device cuda
```

For an eager reference, repeat the command with `--eager --output output-eager`. Both modes
print elapsed training time and, on CUDA, peak allocated memory (including the model, activations,
and optimizer state). Compare runs from the same original model and data; the first step includes
startup costs, so these timings are a smoke benchmark rather than steady-state throughput.

Scope of this experimental integration:

- Deterministic image preprocessing and tokenization; no random data augmentation. Model dropout
  is supported because the existing GradCache machinery replays the model's random state.
- Fixed-size mini-batches with `CachedMultipleNegativesRankingLoss`. Token-budget batching requires
  lengths that are unavailable before preprocessing, so combining it with lazy preprocessing raises
  an error. Other losses and loss wrappers are not enabled in this first version.
- Validated with single-device, full-precision training and evaluation. Distributed training,
  compiled models, mixed precision, and asynchronous mini-batch prefetching are not validated here.
- Text and local image paths only. Keep the dataset as paths, rather than decoding all images
  before passing them to the loss. Inputs must remain unchanged until backward finishes.
- Existing `prompts` and `router_mapping` training arguments are carried through to each mini-batch.
- With a custom data collator, enable its `lazy_preprocessing` option too and keep
  `preprocess_fn=model.preprocess`. A different preprocessing callable would be bypassed by the loss,
  so Trainer rejects it; configure processor options on the model instead. The built-in collator
  emits a small sample-index tensor so Trainer can count even a partial evaluation batch; image
  tensors are still created only inside the loss.
- Padding and other batch-dependent preprocessing now use each mini-batch. Determinism guarantees
  that the two GradCache passes see the same inputs, but does not guarantee identical results to
  eager whole-batch preprocessing. In particular, dynamic left padding changes absolute token
  positions in some models. Use fixed-length padding when preserving those positions is required.
- Images are decoded and processed twice during training. This trades extra CPU/I/O work for
  bounded processed-image memory; whole-batch embeddings and their cached gradients still grow
  with the outer batch size. This is not a claim that total training memory is constant.

The example saves the trained model locally. It does not publish it to the Hub.

## References

```{eval-rst}
- :class:`~sentence_transformers.sentence_transformer.losses.CachedMultipleNegativesRankingLoss`
- :class:`~sentence_transformers.sentence_transformer.losses.MatryoshkaLoss`
- :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator`
- `Training Overview <../../../../docs/sentence_transformer/training_overview.html>`_
- `Loss Overview <../../../../docs/sentence_transformer/loss_overview.html>`_
- `Pretrained Models <../../../../docs/sentence_transformer/pretrained_models.html>`_
```
