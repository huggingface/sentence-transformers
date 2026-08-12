"""
This example shows how to train a multi-vector (late-interaction) model for document screenshot retrieval
without starting from an existing Col* checkpoint. It mirrors the single-vector recipe from the
"Training and Finetuning Multimodal Embedding & Reranker Models" blog post
(https://huggingface.co/blog/train-multimodal-sentence-transformers, see also
examples/sentence_transformer/training/multimodal/training_visual_document_retrieval.py) on the same dataset,
so the two families can be compared directly.

MultiVectorEncoder appends a freshly initialised 128-dim token projection to the multimodal embedding
backbone, the same construction that turned VLMs into ColPali / ColQwen in the first place. Training is
required before the model is useful.

The whole backbone is finetuned, with two training arguments doing the heavy lifting. The fresh
projection gets 10x the backbone learning rate via learning_rate_mapping, since it starts from random
weights. Those random weights also produce gradients that put the total gradient norm far above the
default max_grad_norm=1.0, where clipping would rescale every update to a small fraction of the nominal
learning rate and the backbone would barely move. max_grad_norm=30 keeps clipping as a spike guard
without the constant throttle. Expect roughly 22GB of VRAM plus 15GB of system RAM. On smaller hardware, train a
LoRA adapter on the language model instead (see ../peft/training_msmarco_lora.py for the pattern): a
LoRA reference run on this dataset reached 0.9164 NDCG@10, uploaded as
https://huggingface.co/tomaarsen/ColQwen3-VL-Embedding-2B-vdr-lora.

A reference run of this script trains the model from 0.8244 to 0.9278 NDCG@10 on the held-out eval set
(the untrained projection makes the starting score vary between draws), uploaded as
https://huggingface.co/tomaarsen/ColQwen3-VL-Embedding-2B-vdr. For comparison, the single-vector recipe
from the blog post reaches 0.888 zero-shot and 0.947 with a full finetune on the same data
(https://huggingface.co/tomaarsen/Qwen3-VL-Embedding-2B-vdr).

Usage:
python training_visual_document_retrieval.py
"""

import logging
import traceback

from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
from sentence_transformers.multi_vector_encoder.losses import CachedMultiVectorMultipleNegativesRankingLoss
from sentence_transformers.multi_vector_encoder.model_card import MultiVectorEncoderModelCardData
from sentence_transformers.multi_vector_encoder.trainer import MultiVectorEncoderTrainer
from sentence_transformers.multi_vector_encoder.training_args import MultiVectorEncoderTrainingArguments

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
for logger_name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# 1. Load a model to finetune with (optional) model card data. The multimodal embedding backbone is
# converted into a multi-vector model with a fresh 128-dim token projection.
model = MultiVectorEncoder(
    "Qwen/Qwen3-VL-Embedding-2B",
    model_card_data=MultiVectorEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ColQwen3-VL-Embedding-2B model trained on Visual Document Retrieval query-document screenshot pairs",
    ),
    model_kwargs={"attn_implementation": "kernels-community/flash-attn2@v2", "torch_dtype": "bfloat16"},
    processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 600 * 600},
)

# 2. Load a dataset to finetune on: https://huggingface.co/datasets/tomaarsen/llamaindex-vdr-en-train-preprocessed
# This script uses just one negative per query for training, but 4 are available, and 4 are used for evaluation
train_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "train", split="train")
train_dataset = train_dataset.select_columns(["query", "image", "negative_0"])
eval_dataset = load_dataset("tomaarsen/llamaindex-vdr-en-train-preprocessed", "eval", split="train")
logging.info(train_dataset)
logging.info(eval_dataset)

# 3. Define a loss function. MaxSim scores every query token against every image patch, so the cached
# loss keeps the effective batch large while only mini_batch_size samples are encoded at once.
loss = CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=1)

# 4. (Optional) Specify training arguments
run_name = "ColQwen3-VL-Embedding-2B-vdr"
args = MultiVectorEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    # The fresh projection starts from random weights and needs roughly 10x the backbone lr (the
    # regex also matches the loss-wrapped module path) plus a loosened clip, see the docstring.
    learning_rate_mapping={r"^(model\.)?1\.linear": 2e-4},
    max_grad_norm=30.0,
    warmup_steps=0.1,  # Warm up over the first 10% of training steps
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # BatchSamplers.NO_DUPLICATES is not used here: its duplicate detection does not work on image columns
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    # Checkpoints of a 2B model carry optimizer states roughly the model's size again, and gathering
    # them spikes system RAM at every save. This short run never needs resuming: save weights only.
    save_only_model=True,
    logging_steps=0.05,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 5. (Optional) Create an evaluator & evaluate the base model
eval_queries = {str(qid): sample["query"] for qid, sample in enumerate(eval_dataset)}
eval_corpus = {str(did): sample["image"] for did, sample in enumerate(eval_dataset)}
num_eval = len(eval_dataset)
negative_columns = ["negative_0", "negative_1", "negative_2", "negative_3"]
for neg_idx, neg_col in enumerate(negative_columns):
    for did, sample in enumerate(eval_dataset):
        eval_corpus[str(num_eval * (neg_idx + 1) + did)] = sample[neg_col]
eval_relevant_docs = {str(idx): {str(idx)} for idx in range(len(eval_dataset))}
eval_evaluator = MultiVectorInformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    batch_size=1,
    show_progress_bar=True,
    name="vdr-eval-hard",
)
eval_evaluator(model)

# 6. Create a trainer & train
# The retrieval evaluator is the model-selection signal. An eval_dataset would add an eval-loss pass
# that collates dozens of images per batch for little extra information, so it is omitted.
trainer = MultiVectorEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=eval_evaluator,
)
trainer.train()

# 7. (Optional) Evaluate the trained model after training
eval_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) Push it to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = MultiVectorEncoder({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
