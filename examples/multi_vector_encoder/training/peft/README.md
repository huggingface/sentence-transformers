# Training with PEFT Adapters

Sentence Transformers has been integrated with [PEFT](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning), allowing you to finetune multi-vector (late-interaction) models without fine-tuning all of the model parameters. Instead, with PEFT methods you are only finetuning a fraction of (extra) model parameters with only a minor hit in performance compared to full model finetuning.

## Compatibility Methods

```{eval-rst}
The :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` supports the following methods for interacting with the PEFT Adapters:

   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.add_adapter`: Adds a fresh new adapter to the current model for training.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.load_adapter`: Load adapter weights from a file or Hugging Face Hub repository.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.active_adapters`: Gets the current active adapters.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.set_adapter`: Tell your model to use a specific adapter and disable all others.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.enable_adapters`: Enable all adapters.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.disable_adapters`: Disable all adapters.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.get_adapter_state_dict`: Get the adapter state dict with the weights.
   * :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.delete_adapter`: Delete an adapter from the model.
```

## Adding a New Adapter

```{eval-rst}
Adding a new adapter to a model is as simple as calling :meth:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder.add_adapter` with a (subclass of) :class:`~peft.PeftConfig` on an initialized Multi-Vector Encoder model. In the following example, we use a :class:`~peft.LoraConfig` instance.

The adapter is applied to the Transformer backbone only. The projection layer that maps token embeddings to the multi-vector dimension sits outside the backbone and remains fully trainable, which is convenient as it is randomly initialized when starting from a plain backbone like ``answerdotai/ModernBERT-base``.
```

```python
from peft import LoraConfig, TaskType

from sentence_transformers import MultiVectorEncoder

# 1. Load a model to finetune
model = MultiVectorEncoder("answerdotai/ModernBERT-base")

# 2. Create a LoRA adapter for the model & add it
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    target_modules=["Wqkv", "Wo", "Wi"],  # ModernBERT attention and MLP linear layers
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

# Proceed as usual... See https://sbert.net/docs/multi_vector_encoder/training_overview.html
```

## Training Script

See the following example file for a full example of how PEFT can be used with Multi-Vector Encoder models:

- **[training_msmarco_lora.py](training_msmarco_lora.py)**: This is the same recipe as [training_contrastive.py](../msmarco/training_contrastive.py), finetuning [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on MS MARCO triplets with MultiVectorMultipleNegativesRankingLoss, but it has been adapted to use a [LoRA adapter](https://huggingface.co/docs/peft/en/package_reference/lora) from PEFT. Only the LoRA matrices and the projection layer are trained (8.4% of the parameters), which considerably reduces the gradient and optimizer memory.
