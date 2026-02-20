import logging
import random
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import RerankingEvaluator
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# Training parameters
model_name = "bert-base-uncased"
train_batch_size = 8
num_epochs = 1
max_seq_length = 75

output_dir = f"output/training_stsb_tsdae-{model_name.replace('/', '-')}-{train_batch_size}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Defining our sentence transformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# or to load a pre-trained SentenceTransformer model OR use mean pooling
# model = SentenceTransformer(model_name)
# model.max_seq_length = max_seq_length


# 2. Read eval/test datasets, skipping samples without positive examples
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


def noise_transform(batch, del_ratio=0.6):
    """
    Applies noise by randomly deleting words.

    WARNING: nltk's tokenization/detokenization is designed primarily for English.
    For other languages, especially those without clear word boundaries (e.g., Chinese),
    custom tokenization and detokenization are strongly recommended.

    Args:
        batch (Dict[str, List[str]]): A dictionary with the structure
            {column_name: [string1, string2, ...]}, where each list contains
            the batch data for the respective column.
        del_ratio (float): The ratio of words to delete. Defaults to 0.6.
    """
    from nltk import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    assert 0.0 <= del_ratio < 1.0, "del_ratio must be in the range [0, 1)"
    assert isinstance(batch, dict) and "text" in batch, "batch must be a dictionary with a 'text' key."

    texts = batch["text"]
    noisy_texts = []
    for text in texts:
        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            noisy_texts.append(text)
            continue

        kept_words = [word for word in words if random.random() < del_ratio]
        # Guarantee that at least one word remains
        if len(kept_words) == 0:
            noisy_texts.append(random.choice(words))
            continue

        noisy_texts.append(TreebankWordDetokenizer().detokenize(kept_words))
    return {"noisy": noisy_texts, "text": texts}


# TSDAE requires a dataset with 2 columns: a noisified text column and a text column
# We use a function to delete some words, but you can customize `noise_transform` to noisify your text some other way.
# We use `set_transform` instead of `map` so the noisified text differs each epoch.
train_dataset.set_transform(transform=lambda batch: noise_transform(batch), columns=["text"], output_all_columns=True)
print(train_dataset)
print(train_dataset[0])
"""
Dataset({
    features: ['text'],
    num_rows: 160436
})
{
    'noisy': 'how to get "battery is broken go?',
    'text': "how to get the `` your battery is broken '' message to go away ?",
}
"""
# As you can see, the noisy text is applied on the fly when the sample is accessed.
print(eval_dataset)
print(test_dataset)
"""
Dataset({
    features: ['query', 'positive', 'negative'],
    num_rows: 189
})
Dataset({
    features: ['query', 'positive', 'negative'],
    num_rows: 186
})
"""

# 3. Define our training loss: https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderLoss
# Note that this will likely result in warnings as we're loading 'model_name' as a decoder, but it likely won't
# have weights for that yet. This is fine, as we'll be training it from scratch.
train_loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
logging.info("Evaluation before training:")
dev_evaluator = RerankingEvaluator(eval_dataset, name="AskUbuntu-dev")
dev_evaluator(model)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    learning_rate=3e-5,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
    run_name="tsdae-askubuntu",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the test set after training
logging.info("Evaluation after training:")
test_evaluator = RerankingEvaluator(test_dataset, name="AskUbuntu-test")
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-tsdae-askubuntu")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-tsdae-askubuntu')`."
    )
