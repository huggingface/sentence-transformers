# Training with PEFT Adapters

CrossEncoder supports [PEFT](https://huggingface.co/docs/peft/en/index) (Parameter-Efficient Fine-Tuning), allowing you to fine-tune a small set of adapter parameters instead of every model parameter.

Add adapters through `CrossEncoder.add_adapter()`. Do not replace `model.model` with the result of `get_peft_model()`: that replaces the complete CrossEncoder pipeline and bypasses its input handling.

```python
from peft import LoraConfig, TaskType
from sentence_transformers import CrossEncoder

model = CrossEncoder("distilbert/distilroberta-base", num_labels=1)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
)
model.add_adapter(peft_config)
```

After adding the adapter, train the model normally with `CrossEncoderTrainer`. Saving the model writes the adapter weights and configuration alongside the CrossEncoder metadata, so it can later be loaded with `CrossEncoder`.

## Training Script

- **[training_quora_duplicate_questions_lora.py](training_quora_duplicate_questions_lora.py)** fine-tunes `distilbert/distilroberta-base` on Quora duplicate-question pairs with a LoRA adapter and `BinaryCrossEntropyLoss`.
