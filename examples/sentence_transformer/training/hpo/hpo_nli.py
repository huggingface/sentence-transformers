from datasets import load_dataset

from sentence_transformers import (
    BaseTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.base.training_args import BatchSamplers
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

# 1. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli, 10k samples
train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# 2. Create an evaluator to perform useful HPO
stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=stsb_eval_dataset["sentence1"],
    sentences2=stsb_eval_dataset["sentence2"],
    scores=stsb_eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)


# 3. Define the Hyperparameter Search Space
def hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 2),
        "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 32, 128),
        "warmup_steps": trial.suggest_float("warmup_steps", 0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
    }


# 4. Define the Model Initialization
def hpo_model_init(trial):
    return SentenceTransformer("distilbert-base-uncased")


# 5. Define the Loss Initialization
def hpo_loss_init(model):
    return losses.MultipleNegativesRankingLoss(model)


# 6. Define the Objective Function
def hpo_compute_objective(metrics):
    """
    Valid keys are: 'eval_loss', 'eval_sts-dev_pearson_cosine', 'eval_sts-dev_spearman_cosine',
    'eval_sts-dev_pearson_manhattan', 'eval_sts-dev_spearman_manhattan', 'eval_sts-dev_pearson_euclidean',
    'eval_sts-dev_spearman_euclidean', 'eval_sts-dev_pearson_dot', 'eval_sts-dev_spearman_dot',
    'eval_sts-dev_pearson_max', 'eval_sts-dev_spearman_max', 'eval_runtime', 'eval_samples_per_second',
    'eval_steps_per_second', 'epoch'

    due to the evaluator that we're using.
    """
    return metrics["eval_sts-dev_spearman_cosine"]


# 7. Define the training arguments
args = BaseTrainingArguments(
    # Required parameter:
    output_dir="checkpoints",
    # Optional training parameters:
    # max_steps=10000, # We might want to limit the number of steps for HPO
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="no",  # We don't need to evaluate/save during HPO
    save_strategy="no",
    logging_steps=10,
    run_name="hpo",  # Will be used in W&B if `wandb` is installed
)

# 8. Create the trainer with model_init rather than model
trainer = SentenceTransformerTrainer(
    model=None,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    evaluator=dev_evaluator,
    model_init=hpo_model_init,
    loss=hpo_loss_init,
)

# 9. Perform the HPO
best_trial = trainer.hyperparameter_search(
    hp_space=hpo_search_space,
    compute_objective=hpo_compute_objective,
    n_trials=20,
    direction="maximize",
    backend="optuna",
)
print(best_trial)

# Alternatively, to just train normally:
# trainer.train()
# print(dev_evaluator(trainer.model))
