# Multimodal Training

```{eval-rst}
:class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models can be trained on multimodal data, enabling cross-modal reranking where the model scores pairs involving different modalities. Each element in a pair can be:

- **Text**: strings.
- **Image**: PIL images, file paths, URLs, or numpy/torch arrays.
- **Audio**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"sampling_rate"`` keys, or ``torchcodec.AudioDecoder`` instances.
- **Video**: file paths, numpy/torch arrays, dicts with ``"array"`` and ``"video_metadata"`` keys, or ``torchcodec.VideoDecoder`` instances.
- **Multimodal dicts**: a dict mapping modality names to values, e.g. ``{"text": ..., "image": ...}``. The keys must be ``"text"``, ``"image"``, ``"audio"``, or ``"video"``.

Two architectural approaches are demonstrated here, both training on the `doodles-captions-manual <https://huggingface.co/datasets/julianmoraes/doodles-captions-manual>`_ dataset with :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` and multi-dataset training (image-to-text and text-to-image directions).
```

## Transformer (Any-to-Any) + LogitScore

- **[training_doodles_any_to_any.py](training_doodles_any_to_any.py)**:
  ```{eval-rst}
  This example builds a multimodal :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` from ``Qwen/Qwen3.5-0.8B`` using the module chain :class:`Transformer(transformer_task="any-to-any") <sentence_transformers.base.modules.Transformer>` + :class:`~sentence_transformers.cross_encoder.modules.LogitScore`.

  The ``"any-to-any"`` task loads the full causal LM via :class:`~transformers.AutoModelForMultimodalLM` **with** its language model head, and ``add_generation_prompt=True`` appends the assistant turn start token so the model generates from the right position. :class:`~sentence_transformers.cross_encoder.modules.LogitScore` then takes the next-token logits and computes a relevance score as the log-odds of generating ``"1"`` (match) vs ``"0"`` (no match).

  The model is trained with :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` using multi-dataset training with two sub-datasets:

  - **image_to_text**: given an image query, score text candidates
  - **text_to_image**: given a text query, score image candidates

  Each sample is expanded with negatives at a 1:4 positive-to-negative ratio. Evaluation uses :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator` on both directions.
  ```

## Transformer (Feature Extraction) + Pooling + Dense

- **[training_doodles_feature_extraction.py](training_doodles_feature_extraction.py)**:
  ```{eval-rst}
  This example builds a multimodal :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` from ``Qwen/Qwen3.5-0.8B`` using the module chain :class:`Transformer(transformer_task="feature-extraction") <sentence_transformers.base.modules.Transformer>` + :class:`~sentence_transformers.sentence_transformer.modules.Pooling` (``lasttoken``) + :class:`~sentence_transformers.base.modules.Dense`.

  The ``"feature-extraction"`` task loads only the base model via :class:`~transformers.AutoModel` **without** the LM head, making this approach more memory-efficient. The :class:`~sentence_transformers.sentence_transformer.modules.Pooling` layer extracts the last token's hidden state, and the :class:`~sentence_transformers.base.modules.Dense` layer projects it to a single score.

  To approximate the :class:`~sentence_transformers.cross_encoder.modules.LogitScore` behavior at initialization, the Dense layer's weight is initialized as ``embed("1") - embed("0")`` using the model's input embeddings. Because most causal LMs tie input embeddings with the LM head weights, this gives a starting point equivalent to computing log-odds over the ``"1"`` and ``"0"`` tokens.

  The dataset, loss, and evaluation setup are identical to the LogitScore variant above.
  ```

## Comparing the Two Approaches

| | Any-to-Any + LogitScore | Feature Extraction + Pooling + Dense |
|---|---|---|
| **LM head** | Loaded (full vocabulary logits) | Not loaded (hidden states only) |
| **Memory usage** | Higher | Lower |
| **Score mechanism** | Log-odds from generative output | Learned Dense projection |
| **Initialization** | Uses pretrained LM head directly | Approximates LM head via embedding init |

For large models where GPU memory is a concern, the feature extraction approach may be preferred. Both approaches produce comparable results.

## Other Module Chains

```{eval-rst}
These two approaches are not the only options. :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` supports several module chain patterns depending on the task:

- **Transformer (Sequence Classification)**: The traditional encoder-based approach (e.g. BERT, RoBERTa). A single :class:`~sentence_transformers.base.modules.Transformer` module loads a model via :class:`~transformers.AutoModelForSequenceClassification` with a pretrained classification head, which produces scores without any subsequent modules. This is the default for text-only reranking.
- **Transformer (Text Generation) + LogitScore**: Like the Any-to-Any variant above, but for text-only CausalLM rerankers loaded with :class:`~transformers.AutoModelForCausalLM`. Uses ``transformer_task="text-generation"`` instead of ``"any-to-any"``.

See `Creating Custom CrossEncoder Models <../../../../docs/cross_encoder/usage/custom_models.html>`_ for details on the modular architecture.
```

## References

```{eval-rst}
- :class:`~sentence_transformers.base.modules.Transformer` ``transformer_task`` parameter
- :class:`~sentence_transformers.cross_encoder.modules.LogitScore` API reference
- `CrossEncoder Training Overview <../../../../docs/cross_encoder/training_overview.html>`_
- `CrossEncoder Loss Overview <../../../../docs/cross_encoder/loss_overview.html>`_
```
