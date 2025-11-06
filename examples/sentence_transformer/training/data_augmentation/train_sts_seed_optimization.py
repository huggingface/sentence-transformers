"""
This script is identical to examples/sentence_transformer/training/sts/training_stsbenchmark.py with seed optimization.
We apply early stopping and evaluate the models over the dev set, to find out the best performing seeds.

For more details refer to -
Fine-Tuning Pretrained Language Models:
Weight Initializations, Data Orders, and Early Stopping by Dodge et al. 2020
https://huggingface.co/papers/2002.06305

Why Seed Optimization?
Dodge et al. (2020) show a high dependence on the random seed for transformer based models like BERT,
as it converges to different minima that generalize differently to unseen data. This is especially the
case for small training datasets.

Citation: https://huggingface.co/papers/2010.08240

Usage:
python train_sts_seed_optimization.py

OR
python train_sts_seed_optimization.py pretrained_transformer_model_name seed_count stop_after

python train_sts_seed_optimization.py bert-base-uncased 10 0.3
"""

import logging
import math
import pprint
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import TrainerCallback, TrainerControl, TrainerState

from sentence_transformers import LoggingHandler, SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "bert-base-uncased"
seed_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
stop_after = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

logging.info(f"Train and Evaluate: {seed_count} Random Seeds")

scores_per_seed = {}

for seed in range(seed_count):
    # Setting seed for all random initializations
    logging.info(f"##### Seed {seed} #####")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read the dataset
    train_batch_size = 16
    model_save_path = "output/bi-encoder/training_stsbenchmark_" + model_name + "/seed-" + str(seed)

    # Use Hugging Face/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
    train_dataset = load_dataset("sentence-transformers/stsb", split="train")
    eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
    test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    logging.info(train_dataset)

    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Define an evaluator for use during training.
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
        show_progress_bar=True,
    )

    # Stopping and Evaluating after 30% of training data (less than 1 epoch)
    # We find from (Dodge et al.) that 20-30% is often ideal for convergence of random seed
    num_steps_until_stop = math.ceil(len(train_dataset) / train_batch_size * stop_after)

    logging.info(f"Early-stopping: {stop_after:.2%} ({num_steps_until_stop} steps) of the training-data")

    # 5. Create a Training Callback that stops training after a certain number of steps
    class SeedTestingEarlyStoppingCallback(TrainerCallback):
        def __init__(self, num_steps_until_stop: int):
            self.num_steps_until_stop = num_steps_until_stop

        def on_step_end(
            self, args: SentenceTransformerTrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
        ):
            if state.global_step >= self.num_steps_until_stop:
                control.should_training_stop = True

    seed_testing_early_stopping_callback = SeedTestingEarlyStoppingCallback(num_steps_until_stop)

    # 6. Define the training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=model_save_path,
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        logging_steps=num_steps_until_stop // 10,  # Log every 10% of the steps
        seed=seed,
        run_name=f"sts-{seed}",  # Will be used in W&B if `wandb` is installed
    )

    # 7. Create the trainer & start training
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        callbacks=[seed_testing_early_stopping_callback],
    )
    trainer.train()

    # 8. With the partial train, evaluate this seed on the dev set
    dev_score = dev_evaluator(model)
    logging.info(f"Evaluator Scores for Seed {seed} after early stopping: {dev_score}")
    primary_dev_score = dev_score[dev_evaluator.primary_metric]
    scores_per_seed[seed] = primary_dev_score
    scores_per_seed = dict(sorted(scores_per_seed.items(), key=lambda item: item[1], reverse=True))
    logging.info(
        f"Current {dev_evaluator.primary_metric} Scores per Seed:\n{pprint.pformat(scores_per_seed, sort_dicts=False)}"
    )

    # 9. Save the model for this seed
    model.save_pretrained(model_save_path)
