"""
This example trains a multimodal CrossEncoder on the doodles-captions-manual dataset,
learning to match images with their correct text captions (and vice versa) using
BinaryCrossEntropyLoss with multi-dataset training.

Two sub-datasets are created:
- image_to_text: given an image query, rerank text candidates
- text_to_image: given a text query, rerank image candidates

Each sample is expanded with negatives at a 1:4 positive-to-negative ratio.

Usage:
python training_doodles.py
"""

import logging
import random
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.modules import CausalScoreHead, Transformer

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
# TODO: Ignore all httpx INFO logs

# Config
train_batch_size = 4
num_epochs = 1
neg_to_pos_ratio = 4  # 4 negatives per positive
num_eval_negatives = 10
eval_fraction = 0.1
seed = 42
output_dir = "output/training_ce_doodles-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Load the model
# NOTE: It's recommended to use ``pip install kernels`` to avoid having to install the separate ``flash_attn`` package
model_name = "Qwen/Qwen3.5-0.8B"
transformer = Transformer(
    model_name,
    transformer_task="any-to-any",
    model_kwargs={"torch_dtype": "bfloat16", "device_map": "auto", "attn_implementation": "flash_attention_2"},
    processing_kwargs={"chat_template": {"add_generation_prompt": True}},
)
transformer.processor.chat_template = transformer.processor.chat_template.replace(
    'message.role == "user"', 'message.role in ["user", "query", "document"]'
)
score_head = CausalScoreHead(
    true_token_id=transformer.tokenizer.convert_tokens_to_ids("1"),
    false_token_id=transformer.tokenizer.convert_tokens_to_ids("0"),
)
model = CrossEncoder(
    modules=[transformer, score_head],
    num_labels=1,
    prompts={"default": "Judge whether the image and text match. Respond with 1 if they match, 0 if they don't."},
    default_prompt_name="default",
)

# 2. Load dataset and create train/eval split
max_image_size = 128
logging.info("Loading doodles-captions-manual dataset")
full_dataset = load_dataset("julianmoraes/doodles-captions-manual", split="train")
full_dataset = full_dataset.map(
    lambda example: {"image": example["image"].resize((max_image_size, max_image_size))},
)
full_dataset = full_dataset.train_test_split(test_size=eval_fraction, seed=seed)
train_split = full_dataset["train"]
eval_split = full_dataset["test"]
logging.info(f"Train: {len(train_split)} samples, Eval: {len(eval_split)} samples")


# 3. Expand dataset with negatives using dataset.map with batching
# For each sample, keep the positive pair and randomly sample negatives from a large pool.
rng = random.Random(seed)


def expand_with_negatives(batch, col_a, col_b):
    batch_size = len(batch[col_a])
    expanded_a, expanded_b, labels = [], [], []
    for i in range(batch_size):
        expanded_a.append(batch[col_a][i])
        expanded_b.append(batch[col_b][i])
        labels.append(1)
        neg_indices = rng.sample([j for j in range(batch_size) if j != i], min(neg_to_pos_ratio, batch_size - 1))
        for j in neg_indices:
            expanded_a.append(batch[col_a][i])
            expanded_b.append(batch[col_b][j])
            labels.append(0)
    return {col_a: expanded_a, col_b: expanded_b, "label": labels}


def build_pair_dataset(split, col_a_name, col_b_name):
    """Build a dataset of (col_a, col_b, label) pairs with negatives."""
    return split.select_columns([col_a_name, col_b_name]).map(
        expand_with_negatives,
        batched=True,
        fn_kwargs={"col_a": col_a_name, "col_b": col_b_name},
    )


logging.info("Building image-to-text pair datasets")
train_image_to_text = build_pair_dataset(train_split, "image", "text")
eval_image_to_text = build_pair_dataset(eval_split, "image", "text")

logging.info("Building text-to-image pair datasets")
train_text_to_image = build_pair_dataset(train_split, "text", "image")
eval_text_to_image = build_pair_dataset(eval_split, "text", "image")

logging.info(f"Image-to-text train: {len(train_image_to_text)}, eval: {len(eval_image_to_text)}")
logging.info(f"Text-to-image train: {len(train_text_to_image)}, eval: {len(eval_text_to_image)}")

# 4. Build reranking evaluators
eval_images = eval_split["image"]
eval_texts = eval_split["text"]

rng = random.Random(seed)

image_to_text_samples = []
for i in range(len(eval_split)):
    image_to_text_samples.append(
        {
            "query": eval_images[i],
            "positive": [eval_texts[i]],
            "negative": [eval_texts[j] for j in range(len(eval_texts)) if j != i],
        }
    )

text_to_image_samples = []
for i in range(len(eval_split)):
    text_to_image_samples.append(
        {
            "query": eval_texts[i],
            "positive": [eval_images[i]],
            "negative": [eval_images[j] for j in range(len(eval_images)) if j != i],
        }
    )

image_to_text_evaluator = CrossEncoderRerankingEvaluator(
    samples=image_to_text_samples,
    name="doodles-image-to-text-eval",
    show_progress_bar=True,
)

text_to_image_evaluator = CrossEncoderRerankingEvaluator(
    samples=text_to_image_samples,
    name="doodles-text-to-image-eval",
    show_progress_bar=True,
)

# Evaluate before training to get a baseline
print("Evaluating before training:")
image_to_text_evaluator(model)
text_to_image_evaluator(model)

# 5. Define loss
loss = BinaryCrossEntropyLoss(model)

# 6. Training arguments
short_model_name = model_name.split("/")[-1]
run_name = f"reranker-{short_model_name}-doodles-image-text-to-text"
args = CrossEncoderTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    prompts=model.prompts.get("default"),
    warmup_ratio=0.1,
    learning_rate=5e-6,
    fp16=False,
    bf16=True,
    eval_strategy="steps",
    eval_steps=0.25,
    save_strategy="steps",
    save_steps=0.25,
    save_total_limit=2,
    logging_steps=0.1,
    run_name=run_name,
)

# 7. Multi-dataset training: pass dicts of datasets
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset={
        "image_to_text": train_image_to_text,
        "text_to_image": train_text_to_image,
    },
    eval_dataset={
        "image_to_text": eval_image_to_text,
        "text_to_image": eval_text_to_image,
    },
    loss=loss,
    evaluator=[image_to_text_evaluator, text_to_image_evaluator],
)
trainer.train()

# 8. Final evaluation
logging.info("Final evaluation")
image_to_text_evaluator(model)
text_to_image_evaluator(model)

# 9. Save model
final_output_dir = f"{output_dir}/final"
model.save_pretrained(final_output_dir)

# 10. (Optional) Push to Hub
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}"
        f"To upload it manually, you can run `huggingface-cli login`, followed by loading the model "
        f"using `model = CrossEncoder({final_output_dir!r})` and saving it using `model.push_to_hub('{run_name}')`."
    )
