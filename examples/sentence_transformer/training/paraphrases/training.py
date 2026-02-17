"""
Note: This script was modified with the v3 release of Sentence Transformers.
As a result, it does not produce exactly the same behaviour as the original script.
"""

import logging
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.trainer import SentenceTransformerTrainer
from sentence_transformers.base.training_args import (
    BaseTrainingArguments,
    BatchSamplers,
    MultiDatasetBatchSamplers,
)
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.util.similarity import SimilarityFunction

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

model_name = "distilroberta-base"
num_epochs = 1
batch_size = 128
max_seq_length = 128

# Save path of the model
output_dir = (
    "output/training_paraphrases_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# 2. Load some training dataset from: https://huggingface.co/datasets?other=sentence-transformers
# Notably, we are looking for datasets compatible with MultipleNegativesRankingLoss, which accepts
# triplets of sentences (anchor, positive, negative) and pairs of sentences (anchor, positive).
all_nli_train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
sentence_compression_train_dataset = load_dataset("sentence-transformers/sentence-compression", split="train")
simple_wiki_train_dataset = load_dataset("sentence-transformers/simple-wiki", split="train")
altlex_train_dataset = load_dataset("sentence-transformers/altlex", split="train")
quora_train_dataset = load_dataset("sentence-transformers/quora-duplicates", "triplet", split="train")
coco_train_dataset = load_dataset("sentence-transformers/coco-captions", split="train")
flickr_train_dataset = load_dataset("sentence-transformers/flickr30k-captions", split="train")
yahoo_answers_train_dataset = load_dataset(
    "sentence-transformers/yahoo-answers", "title-question-answer-pair", split="train"
)
stack_exchange_train_dataset = load_dataset(
    "sentence-transformers/stackexchange-duplicates", "title-title-pair", split="train"
)

train_dataset_dict = {
    "all-nli": all_nli_train_dataset,
    "sentence-compression": sentence_compression_train_dataset,
    "simple-wiki": simple_wiki_train_dataset,
    "altlex": altlex_train_dataset,
    "quora-duplicates": quora_train_dataset,
    "coco-captions": coco_train_dataset,
    "flickr30k-captions": flickr_train_dataset,
    "yahoo-answers": yahoo_answers_train_dataset,
    "stack-exchange": stack_exchange_train_dataset,
}
print(train_dataset_dict)

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
model.max_seq_length = max_seq_length
logging.info(model)

# 3. Define our training loss
train_loss = MultipleNegativesRankingLoss(model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# 5. Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # We can use ROUND_ROBIN or PROPORTIONAL - to avoid focusing too much on one dataset, we will
    # use round robin, which samples the same amount of batches from each dataset, until one dataset is empty
    multi_dataset_batch_sampler=MultiDatasetBatchSamplers.ROUND_ROBIN,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name="paraphrases-multi",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset_dict,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-paraphrases-multi")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-paraphrases-multi')`."
    )
