import logging
import traceback

from datasets import load_dataset

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "microsoft/MiniLM-L12-H384-uncased"
train_batch_size = 64
num_epochs = 1
num_rand_negatives = 5  # How many random negatives should be used for each question-answer pair

# 1a. Load a model to finetune with 1b. (Optional) model card data
model = CrossEncoder(
    model_name,
    model_card_data=CrossEncoderModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MiniLM-L12-H384 trained on GooAQ",
    ),
)
print("Model max length:", model.max_length)
print("Model num labels:", model.num_labels)

# 2. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
logging.info("Read the gooaq training dataset")
full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
logging.info(train_dataset)
logging.info(eval_dataset)

# 3. Define our training loss.
loss = CachedMultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
    mini_batch_size=16,  # Informs the memory usage
)

# 4. Use CrossEncoderNanoBEIREvaluator, a light-weight evaluator for English reranking
evaluator = CrossEncoderNanoBEIREvaluator(
    dataset_names=["msmarco", "nfcorpus", "nq"],
    batch_size=train_batch_size,
)
evaluator(model)

# 5. Define the training arguments
short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
run_name = f"reranker-{short_model_name}-gooaq-cmnrl"
args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    learning_rate=2e-5,
    warmup_steps=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    logging_steps=100,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=12,
)

# 6. Create the trainer & start training
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 7. Evaluate the final model, useful to include these in the model card
evaluator(model)

# 8. Save the final model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
try:
    model.push_to_hub(run_name)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = CrossEncoder({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{run_name}')`."
    )
