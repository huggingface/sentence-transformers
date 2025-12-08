
Hyperparameter Optimization
===========================

The :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` supports hyperparameter optimization using ``transformers``, which in turn supports four hyperparameter search backends: `optuna <https://optuna.org/>`_, `sigopt <https://sigopt.org/>`_, `raytune <https://docs.ray.io/en/latest/tune/index.html>`_, and `wandb <https://wandb.ai/site/sweeps>`_. You should install your backend of choice before using it::

    pip install optuna/sigopt/wandb/ray[tune] 

On this page, we'll show you how to use the hyperparameter optimization feature with the `optuna` backend. The other backends are similar to use, but you should refer to their respective documentation or the `transformers HPO documentation <https://huggingface.co/docs/transformers/en/hpo_train>`_ for more information.

HPO Components
--------------

The hyperparameter optimization process consists of the following components:

.. raw:: html

    <div class="components">
        <a href="#hyperparameter-search-space" class="box">
            <div class="header">Hyperparameter Search Space</div>
            Specify ranges for hyperparameter values.
        </a>
        <a href="#model-initialization" class="box">
            <div class="header">Model Initialization</div>
            Initialize a SentenceTransformer model for a trial.
        </a>
        <a href="#loss-initialization" class="box">
            <div class="header">Loss Initialization</div>
            Initialize a loss function given a model.
        </a>
        <a href="#compute-objective" class="box">
            <div class="header">Compute Objective</div>
            Determines the value to be minimized or maximized.
        </a>
    </div>
    <br>

Hyperparameter Search Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hyperparameter search space is defined by a function that returns a dictionary of hyperparameters and their respective search spaces. Here's an example using ``optuna`` of a search space function that defines the hyperparameters for a `SentenceTransformer` model::

    def hpo_search_space(trial):
        return {
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 2),
            "per_device_train_batch_size": trial.suggest_int("per_device_train_batch_size", 32, 128),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        }

Model Initialization
~~~~~~~~~~~~~~~~~~~~

The model initialization function is a function that takes the hyperparameters of the current "trial" as input and returns a `SentenceTransformer` model. Generally, this function is quite simple. Here's an example of a model initialization function::

    def hpo_model_init(trial):
        return SentenceTransformer("distilbert-base-uncased")

Loss Initialization
~~~~~~~~~~~~~~~~~~~

The loss initialization function is a function that takes the model initialized for the current trial and returns a loss function. Here's an example of a loss initialization function::

    def hpo_loss_init(model):
        return losses.CosineSimilarityLoss(model)

Compute Objective
~~~~~~~~~~~~~~~~~

The compute objective function is a function that takes the evaluation ``metrics`` and returns the float value to be minimized or maximized. Here's an example of a compute objective function::

    def hpo_compute_objective(metrics):
        return metrics["eval_sts-dev_spearman_cosine"]

.. note:

    The dictionary keys of ``metrics`` are all prepended with ``eval_``. Additionally, if you're interested in maximizing the performance of an evaluator, note that the ``name`` of the evaluator is also prepended with a ``-``. So, to optimize on ``spearman_cosine`` from :class:`~sentence_transformers.evaluation.EmbeddingSimilarityEvaluator` which was initialized with ``name="stsb_dev"``, then you would use the key ``eval_sts-dev_spearman_cosine`` in your ``hpo_compute_objective``.

    Another common option is to use ``eval_loss``.

Putting It All Together
------------------------

You can perform HPO on any regular training loop, the only difference being that you don't call :meth:`SentenceTransformerTrainer.train <sentence_transformers.trainer.SentenceTransformerTrainer.train>`, but :meth:`SentenceTransformerTrainer.hyperparameter_search <sentence_transformers.trainer.SentenceTransformerTrainer.hyperparameter_search>` instead. Here's an example of how to put it all together:

.. sidebar:: Documentation

    #. `sentence-transformers/all-nli <https://huggingface.co/datasets/sentence-transformers/all-nli>`_
    #. :class:`~sentence_transformers.evaluation.EmbeddingSimilarityEvaluator`
    #. `Hyperparameter Search Space <#hyperparameter-search-space>`_
    #. `Model Initialization <#model-initialization>`_
    #. `Loss Initialization <#loss-initialization>`_
    #. `Compute Objective <#compute-objective>`_
    #. :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`
    #. :class:`~sentence_transformers.trainer.SentenceTransformerTrainer`
    #. :meth:`~sentence_transformers.trainer.SentenceTransformerTrainer.hyperparameter_search`

::

    from sentence_transformers import losses
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
    from sentence_transformers.training_args import BatchSamplers
    from datasets import load_dataset

    # 1. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli, only 10k train and 1k dev
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
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
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
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="checkpoints",
        # Optional training parameters:
        # max_steps=10000, # We might want to limit the number of steps for HPO
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="no", # We don't need to evaluate/save during HPO
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

::

    [I 2024-05-17 15:10:47,844] Trial 0 finished with value: 0.7889856589698055 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 123, 'warmup_ratio': 0.07380948785410107, 'learning_rate': 2.686331417509812e-06}. Best is trial 0 with value: 0.7889856589698055.
    [I 2024-05-17 15:12:13,283] Trial 1 finished with value: 0.7927780672090986 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 69, 'warmup_ratio': 0.2927897848007451, 'learning_rate': 5.885372118095137e-06}. Best is trial 1 with value: 0.7927780672090986.
    [I 2024-05-17 15:12:43,896] Trial 2 finished with value: 0.7684829743509601 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 114, 'warmup_ratio': 0.0739429232666916, 'learning_rate': 7.344415188959276e-05}. Best is trial 1 with value: 0.7927780672090986.
    [I 2024-05-17 15:14:49,730] Trial 3 finished with value: 0.7873032743147989 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 43, 'warmup_ratio': 0.15184370143796674, 'learning_rate': 9.703232080395476e-06}. Best is trial 1 with value: 0.7927780672090986.
    [I 2024-05-17 15:15:39,597] Trial 4 finished with value: 0.7759251781929949 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 127, 'warmup_ratio': 0.263946220093495, 'learning_rate': 1.231454337152625e-06}. Best is trial 1 with value: 0.7927780672090986.
    [I 2024-05-17 15:17:02,191] Trial 5 finished with value: 0.7964580509886684 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 34, 'warmup_ratio': 0.2276865359631089, 'learning_rate': 7.889007438884571e-06}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:18:55,559] Trial 6 finished with value: 0.7901878917859169 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 48, 'warmup_ratio': 0.23228838664572948, 'learning_rate': 2.883013292682523e-06}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:20:27,027] Trial 7 finished with value: 0.7935671067660925 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 62, 'warmup_ratio': 0.22061123927198237, 'learning_rate': 2.95413457610349e-06}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:22:23,147] Trial 8 finished with value: 0.7848123114933252 and parameters: {'num_train_epochs': 2, 'per_device_train_batch_size': 45, 'warmup_ratio': 0.23071701022961139, 'learning_rate': 9.793681667449783e-06}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:22:52,826] Trial 9 finished with value: 0.7909708416168918 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 121, 'warmup_ratio': 0.22440506724181647, 'learning_rate': 4.0744671365843346e-05}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:23:30,395] Trial 10 finished with value: 0.7928991732385567 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 89, 'warmup_ratio': 0.14607293301068847, 'learning_rate': 2.5557492055039498e-05}. Best is trial 5 with value: 0.7964580509886684.
    [I 2024-05-17 15:24:18,024] Trial 11 finished with value: 0.7991870087507459 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 66, 'warmup_ratio': 0.16886154348739527, 'learning_rate': 3.705926066938032e-06}. Best is trial 11 with value: 0.7991870087507459.
    [I 2024-05-17 15:25:44,198] Trial 12 finished with value: 0.7923304174306207 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 33, 'warmup_ratio': 0.15953772535423974, 'learning_rate': 1.8076298025704224e-05}. Best is trial 11 with value: 0.7991870087507459.
    [I 2024-05-17 15:26:20,739] Trial 13 finished with value: 0.8020260244040395 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 90, 'warmup_ratio': 0.18105202625281253, 'learning_rate': 5.513908793512551e-06}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:26:57,783] Trial 14 finished with value: 0.7571110256860063 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 95, 'warmup_ratio': 0.00122391151793258, 'learning_rate': 1.0432486633629492e-06}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:27:32,581] Trial 15 finished with value: 0.8009013936824717 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 101, 'warmup_ratio': 0.1761274711346081, 'learning_rate': 4.5918293464430035e-06}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:28:05,850] Trial 16 finished with value: 0.8017668050806169 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 103, 'warmup_ratio': 0.10766501647726355, 'learning_rate': 5.0309795522333e-06}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:28:37,393] Trial 17 finished with value: 0.7769412380909586 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 108, 'warmup_ratio': 0.1036610178950246, 'learning_rate': 1.7747598626081271e-06}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:29:19,340] Trial 18 finished with value: 0.8011921300048339 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 80, 'warmup_ratio': 0.117014165550441, 'learning_rate': 1.238558867958792e-05}. Best is trial 13 with value: 0.8020260244040395.
    [I 2024-05-17 15:29:59,508] Trial 19 finished with value: 0.8027501854704168 and parameters: {'num_train_epochs': 1, 'per_device_train_batch_size': 84, 'warmup_ratio': 0.014601112207929548, 'learning_rate': 5.627813947769514e-06}. Best is trial 19 with value: 0.8027501854704168.

    BestRun(run_id='19', objective=0.8027501854704168, hyperparameters={'num_train_epochs': 1, 'per_device_train_batch_size': 84, 'warmup_ratio': 0.014601112207929548, 'learning_rate': 5.627813947769514e-06}, run_summary=None)

As you can see, the strongest hyperparameters reached **0.802** Spearman correlation on the STS (dev) benchmark. For context, training with the default training arguments (``per_device_train_batch_size=8``, ``learning_rate=5e-5``) results in **0.736**, and hyperparameters chosen based on experience (``per_device_train_batch_size=64``, ``learning_rate=2e-5``) results in **0.783** Spearman correlation. Consequently, HPO proved quite effective here in improving the model performance.

Example Scripts
---------------

- `hpo_nli.py <https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/hpo/hpo_nli.py>`_ - An example script that performs hyperparameter optimization on the AllNLI dataset.
