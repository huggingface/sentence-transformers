# Training with PEFT Adapters

Sentence Transformers has been integrated with [PEFT](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning), allowing you to finetune CrossEncoder reranker models without fine-tuning all of the model parameters. Instead, with PEFT methods you are only finetuning a fraction of (extra) model parameters with only a minor hit in performance compared to full model finetuning.

## Compatibility Methods

```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` supports the following methods for interacting with the PEFT Adapters:

   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.add_adapter`: Adds a fresh new adapter to the current model for training.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.load_adapter`: Load adapter weights from a file or Hugging Face Hub repository.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.active_adapters`: Gets the current active adapters.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.set_adapter`: Tell your model to use a specific adapter and disable all others.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.enable_adapters`: Enable all adapters.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.disable_adapters`: Disable all adapters.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.get_adapter_state_dict`: Get the adapter state dict with the weights.
   * :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.delete_adapter`: Delete an adapter from the model.
```

## Adding a New Adapter

```{eval-rst}
Adding a new adapter to a model is as simple as calling :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.add_adapter` with a (subclass of) :class:`~peft.PeftConfig` on an initialized CrossEncoder model. In the following example, we use a :class:`~peft.LoraConfig` instance with ``task_type=SEQ_CLS``, matching the sequence classification setup that CrossEncoder models use.

```

```python
from peft import LoraConfig, TaskType

from sentence_transformers import CrossEncoder

# 1. Load a model to finetune, e.g. an existing reranker or any backbone
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", num_labels=1)

# 2. Create a LoRA adapter for the model & add it
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    target_modules=["query", "value"],  # Query and Value attention linears for BERT-family models
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

# Proceed as usual... See https://sbert.net/docs/cross_encoder/training_overview.html
```

`target_modules` refers to the names of the modules to which the adapters are applied: e.g. `["query", "value"]` for BERT/MiniLM-based models, `["q_lin", "v_lin"]` for DistilBERT-based models, or `["Wqkv", "Wo"]` for ModernBERT models. Consider setting it explicitly rather than leaving it unset, as the valid names differ per architecture.

Note that with this setup the classification head remains frozen. That is convenient when starting from an existing reranker trained for relevance scoring, as in the training script below: the LoRA adapter then nudges the representations on top of the relevance behaviour that the model has already learned. If you train a fresh head over a plain backbone, consider whether you need PEFT's `modules_to_save` so the head is saved together with the adapter, and verify that saving and reloading reproduces your trained head weights.

## Using ``get_peft_model`` Directly

Not recommended. Some community finetuning snippets wrap the model by reassigning the wrapped transformer directly, e.g. `model.model = get_peft_model(model.model, peft_config)`. With `SentenceTransformer` and `CrossEncoder`, this breaks with a confusing error on the first forward pass: the `CrossEncoder` iterates over its modules and calls each with a single preprocessed model input (e.g. the tokenized pairs), while the PEFT model's `forward` expects keyword arguments such as `input_ids` and `attention_mask`, so the preprocessed input ends up bound to the wrong keyword. `add_adapter` (or `load_adapter`) applies the adapter to the underlying transformers model exactly as the upstream [PEFT integration](https://huggingface.co/docs/transformers/en/main_classes/peft#transformers.integrations.PeftAdapterMixin) expects, keeping the usual training pipeline intact. See [#3916](https://github.com/huggingface/sentence-transformers/issues/3916) for details.

## Loading a Pretrained Adapter

Adapter checkpoints are tiny: saving only stores the adapter weights, not the full model, and loading only requires the base model:

```python
from sentence_transformers import CrossEncoder

# Load a checkpoint that contains adapter weights
model = CrossEncoder("path/to/adapter_checkpoint")
scores = model.predict([("How to bake bread?", "Knead dough and bake at 220C.")])
```

Or you can load the base model and load the adapter into it:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
model.load_adapter("path/to/adapter_checkpoint")
```

In most cases, the former is easiest, as it will work regardless of whether the model is an adapter model or not.

## Training Script

See the following example file for a full example of how PEFT can be used with CrossEncoder models:

- **[training_gooaq_lora.py](training_gooaq_lora.py)**: This is the same recipe as [training_gooaq_bce.py](../rerankers/training_gooaq_bce.py), finetuning the [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) reranker on GooAQ question-answer pairs with hard negatives using BinaryCrossEntropyLoss, but it has been adapted to use a [LoRA adapter](https://huggingface.co/docs/peft/en/package_reference/lora) from PEFT, so only a small fraction of the parameters is trained.
