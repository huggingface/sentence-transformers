import logging
import random
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import ContrastiveTensionLoss
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# print debug information to stdout

# Some training parameters. We use a batch size of 16, for every positive example we include 8-1=7 negative examples
# Sentences are truncated to 75 word pieces
model_name = "distilbert-base-uncased"
batch_size = 16
pos_neg_ratio = 8  # batch_size must be divisible by pos_neg_ratio
max_seq_length = 75
num_epochs = 1

output_path = "output/train_askubuntu_ct-{}-{}-{}".format(
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


# Generate sentence pairs for ContrastiveTensionLoss
def to_ct_pairs(sample, pos_neg_ratio=8):
    pos_neg_ratio = 1 / pos_neg_ratio
    return {
        "text1": sample["text"],
        "text2": sample["text"] if random.random() < pos_neg_ratio else random.choice(train_dataset)["text"],
    }


train_dataset = train_dataset.map(to_ct_pairs, fn_kwargs={"pos_neg_ratio": pos_neg_ratio}, remove_columns=["text"])
logging.info(train_dataset)
logging.info(train_dataset[0])

# Initialize an SBERT model
word_embedding_model = Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# As loss, we use ContrastiveTensionLoss
train_loss = ContrastiveTensionLoss(model)

# Create a dev evaluator
dev_evaluator = RerankingEvaluator(eval_dataset, name="askubuntu-dev")
test_evaluator = RerankingEvaluator(test_dataset, name="askubuntu-test")

# Evaluate the model before training
dev_evaluator(model)
test_evaluator(model)

logging.info("Start training")
# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir=output_path,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_steps=0,
    learning_rate=2e-6,
    weight_decay=0,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=100,
    fp16=False,  # Set to True, if your GPU has optimized FP16 cores
    optim="rmsprop",
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

logging.info("Evaluating after training")
dev_evaluator(model)
test_evaluator(model)

# Save the trained & evaluated model locally
final_output_dir = f"{output_path}/final"
model.save_pretrained(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-askubuntu-ct")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-askubuntu-ct')`."
    )
