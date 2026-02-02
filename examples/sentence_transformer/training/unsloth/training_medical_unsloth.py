import logging
import traceback

import torch
from datasets import Dataset, load_dataset
from unsloth import FastSentenceTransformer, is_bf16_supported

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

# 1. Load a base model to finetune using FastSentenceTransformer
model = FastSentenceTransformer.from_pretrained(
    model_name="google/embeddinggemma-300m",
    max_seq_length=1024,
    full_finetuning=False,
)

# 2. Add LoRA adapters so we only need to update a small amount of parameters
model = FastSentenceTransformer.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    task_type="FEATURE_EXTRACTION",
)
model.bfloat16()

# 3. Load a medical retrieval dataset (MIRIAD) with streaming for low memory usage
stream_train = list(load_dataset("tomaarsen/miriad-4.4M-split", split="train", streaming=True).take(10000))
stream_eval = list(load_dataset("tomaarsen/miriad-4.4M-split", split="eval", streaming=True).take(2000))

train_dataset = Dataset.from_generator(lambda: (yield from stream_train))
eval_dataset = Dataset.from_generator(lambda: (yield from stream_eval))
print(train_dataset)
print(train_dataset[0])

# 4. Build an Information Retrieval evaluator for MIRIAD
queries = dict(enumerate(eval_dataset["question"]))
corpus = dict(enumerate(list(eval_dataset["passage_text"]) + train_dataset["passage_text"][:2000]))
relevant_docs = {idx: [idx] for idx in queries}
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    batch_size=64,
)
evaluator(model)

# 5. Define a loss function
# This will use other positives in the same batch as negative examples
loss = losses.MultipleNegativesRankingLoss(model)

# 6. Define training arguments
run_name = "embeddinggemma-300m-miriad-unsloth"
args = SentenceTransformerTrainingArguments(
    # num_train_epochs=1,  # You can use num_train_epochs or max_steps to limit training
    # max_steps=30,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,  # Use GA to mimic a larger batch size, although the loss doesn't benefit from it
    learning_rate=2e-5,
    logging_steps=0.01,
    warmup_ratio=0.03,
    prompts={  # Map training column names to model prompts
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    eval_strategy="steps",
    eval_steps=0.2,
    save_strategy="steps",
    save_steps=0.2,
    bf16=is_bf16_supported(),  # Use BF16 when the GPU supports it
    output_dir="output",
    lr_scheduler_type="linear",
    # Because we have duplicate anchors in the dataset, we don't want
    # to accidentally use them for negative examples
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

# 7. Create a trainer & train the model
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    args=args,
    evaluator=evaluator,  # Optional, will make training slower
)

# Track GPU memory usage for this LoRA run
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# (Optional) Evaluate the trained model on the evaluator after training
evaluator(model)

# 8. Run an inference example on a clinical question
query = "Patient presents with sharp chest pain that improves when leaning forward."

candidates = [
    "Acute Pericarditis often involves pleuritic chest pain relieved by sitting up and leaning forward.",
    "Myocardial Infarction typically presents with crushing substernal pressure and radiation to the left arm.",
    "Pneumothorax is characterized by sudden onset shortness of breath and unilateral chest pain.",
    "Gastroesophageal Reflux Disease (GERD) causes burning retrosternal pain usually after meals.",
]

query_emb = model.encode(query, convert_to_tensor=True)
candidate_embs = model.encode(candidates, convert_to_tensor=True)
similarities = model.similarity(query_emb, candidate_embs)

ranking = similarities.argsort(descending=True)[0]

for idx in ranking.tolist():
    score = similarities[0][idx].item()
    text = candidates[idx]
    print(f"{score:.4f} | {text}")

# 9. Save the trained model, only the LoRA adapters. Use save_pretrained_merged to save a full model.
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 10. (Optional) Save the model to the Hugging Face Hub (adapters only). Use push_to_hub_merged to push a full model.
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
