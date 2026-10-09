# Training CrossEncoders with PEFT Adapters

Sentence Transformers has been integrated with [PEFT](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning), allowing you to finetune reranker models without fine-tuning all of the model parameters. Instead, with PEFT methods you are only finetuning a fraction of (extra) model parameters with only a minor hit in performance compared to full model finetuning.

PEFT Adapter models can be loaded just like any others, but the saved checkpoint only contains the tiny adapter weights rather than a full copy of the base model:

```python
from sentence_transformers.cross_encoder import CrossEncoder

# Load a CrossEncoder that was trained with e.g. a LoRA adapter
model = CrossEncoder("models/reranker-msmarco-v1.1-ms-marco-MiniLM-L6-v2-lora/final")
# Run inference
query = "How many people live in Berlin?"
passages = [
    "Berlin had a population of 3,520,061 registered inhabitants in an area of 891.82 square kilometers.",
    "Berlin is well known for its museums.",
]
scores = model.predict([(query, passage) for passage in passages])
print(scores)
```

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
Adding a new adapter to a model is as simple as calling :meth:`~sentence_transformers.cross_encoder.model.CrossEncoder.add_adapter` with a (subclass of) :class:`~peft.PeftConfig` on an initialized CrossEncoder model. In the following example, we use a :class:`~peft.LoraConfig` instance.
```

```python
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from peft import LoraConfig, TaskType

# 1. Load a model to finetune
# Loading in fp32 is preferred for training if your memory can handle it
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    model_card_data=CrossEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ms-marco-MiniLM-L6-v2 adapter finetuned on MS MARCO",
    ),
    model_kwargs={"torch_dtype": "float32"},
)

# 2. Create a LoRA adapter for the model & add it
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)

# Proceed as usual, e.g. with CrossEncoderTrainer and BinaryCrossEntropyLoss...
# See https://sbert.net/docs/cross_encoder/training_overview.html
```

## Loading a Pretrained Adapter

To load an adapter model, you can either load the adapter checkpoint directly:

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder("models/reranker-msmarco-v1.1-ms-marco-MiniLM-L6-v2-lora/final")
scores = model.predict([("How many people live in Berlin?", "Berlin had a population of 3,520,061.")])
print(scores)
```

Or you can load the base model and load the adapter into it:

```python
from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
model.load_adapter("models/reranker-msmarco-v1.1-ms-marco-MiniLM-L6-v2-lora/final")
scores = model.predict([("How many people live in Berlin?", "Berlin had a population of 3,520,061.")])
print(scores)
```

In most cases, the former is easiest, as it will work regardless of whether the model is an adapter model or not.

## Training Script

See the following example file for a full example of how PEFT can be used with CrossEncoders:

- **[training_ms_marco_lora.py](training_ms_marco_lora.py)**: This is a simple recipe for finetuning [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) on the MS MARCO passage ranking dataset with BinaryCrossEntropyLoss, adapted to train only a [LoRA adapter](https://huggingface.co/docs/peft/en/package_reference/lora) from PEFT instead of the full model.
