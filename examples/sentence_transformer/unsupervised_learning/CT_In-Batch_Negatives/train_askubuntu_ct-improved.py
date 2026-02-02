import logging
from datetime import datetime

from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models

# Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
# print debug information to stdout

# Some training parameters. We use a batch size of 16, for every positive example we include 8-1=7 negative examples
# Sentences are truncated to 75 word pieces
# Training parameters
model_name = "distilbert-base-uncased"
batch_size = 128
epochs = 1
max_seq_length = 75

output_path = "output/train_askubuntu_ct-improved-{}-{}-{}".format(
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
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Train the model

# For ContrastiveTension we need a special data loader to construct batches with the desired properties
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# As loss, we losses.ContrastiveTensionLoss
train_loss = losses.ContrastiveTensionLossInBatchNegatives(model)

# Create a dev evaluator
dev_evaluator = evaluation.RerankingEvaluator(eval_dataset, name="AskUbuntu dev")
test_evaluator = evaluation.RerankingEvaluator(test_dataset, name="AskUbuntu test")


logging.info("Start training")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    evaluation_steps=100,
    epochs=1,
    warmup_steps=100,
    use_amp=True,  # Set to True, if your GPU has optimized FP16 cores
)

latest_output_path = output_path + "-latest"
model.save(latest_output_path)

# Run test evaluation on the latest model. This is equivalent to not having a dev dataset
model = SentenceTransformer(latest_output_path)
test_evaluator(model)
