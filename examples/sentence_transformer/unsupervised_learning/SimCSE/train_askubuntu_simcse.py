import logging
from datetime import datetime

from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Some training parameters. For the example, we use a batch_size of 128, a max sentence length (max_seq_length)
# of 32 word pieces and as model roberta-base
model_name = "roberta-base"
batch_size = 128
max_seq_length = 32
num_epochs = 1

output_path = "output/askubuntu-simcse-{}-{}-{}".format(
    model_name, batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Read eval/test datasets, skipping samples without positive examples
test_dataset = load_dataset("sentence-transformers/askubuntu", split="test").filter(lambda x: x["positive"])
eval_dataset = load_dataset("sentence-transformers/askubuntu", split="dev").filter(lambda x: x["positive"])

dev_or_test_questions = set()
for dataset in (eval_dataset, test_dataset):
    dev_or_test_questions.update(dataset["query"])
    dev_or_test_questions.update(question for question_list in dataset["positive"] for question in question_list)
    dev_or_test_questions.update(question for question_list in dataset["negative"] for question in question_list)

# Load questions for training, skipping those that are part of dev or test sets
train_dataset = load_dataset("sentence-transformers/askubuntu-questions", split="train").filter(
    lambda x: x["text"] not in dev_or_test_questions
)
logging.info(train_dataset)

# Initialize an SBERT model


word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# As Loss function, we use MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(eval_dataset, name="AskUbuntu dev")
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")

logging.info("Dev performance before training")
dev_evaluator(model)

# Warmup: 10% of steps
steps_per_epoch = len(train_dataset) // batch_size
warmup_steps = int(num_epochs * steps_per_epoch * 0.1)

# Prepare training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0.1,
    eval_steps=0.1,
    logging_steps=0.01,
    learning_rate=5e-5,
    save_strategy="no",
    fp16=True,  # If your GPU does not have FP16 cores, set fp16=False
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=train_loss,
)

logging.info("Start training")
trainer.train()

# Run test evaluation
test_evaluator(model)

latest_output_path = output_path + "-latest"
model.save_pretrained(latest_output_path)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-simcse")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\nTo upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({latest_output_path!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-simcse')`."
    )
