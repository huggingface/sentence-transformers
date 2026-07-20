# Training Overview

```{eval-rst}
.. tip::

   Using an AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, ...)?

   .. code-block:: bash
      :class: tight-code

      hf skills add train-sentence-transformers [--claude] [--global]

   And ask your agent to fine-tune a ColBERT-style multi-vector retrieval model for whatever task you have in mind.
```

## Why Finetune?

Finetuning multi-vector (a.k.a. late-interaction or ColBERT-style) models heavily improves their retrieval performance on your specific domain: the vocabulary, the query style, and the notion of relevance all differ between e.g. web search, legal discovery, code search, and scientific literature review. Because queries and documents are matched *token by token* with MaxSim, multi-vector models pick up fine-grained domain signals that single-vector models tend to average away, and they typically respond very well to even modest amounts of in-domain finetuning data.

Also see [**Training Examples**](training/examples) for training scripts for common real-world recipes that you can adopt.

## Training Components

Training MultiVectorEncoder models involves between 4 to 6 components:

<div class="components">
    <a href="#model" class="box">
        <div class="header">Model</div>
        Learn how to initialize the <b>model</b> for training.
    </a>
    <a href="#dataset" class="box">
        <div class="header">Dataset</div>
        Learn how to prepare the <b>data</b> for training.
    </a>
    <a href="#loss-function" class="box">
        <div class="header">Loss Function</div>
        Learn how to prepare and choose a <b>loss</b> function.
    </a>
    <a href="#training-arguments" class="box optional">
        <div class="header">Training Arguments</div>
        Learn which <b>training arguments</b> are useful.
    </a>
    <a href="#evaluator" class="box optional">
        <div class="header">Evaluator</div>
        Learn how to <b>evaluate</b> during and after training.
    </a>
    <a href="#trainer" class="box">
        <div class="header">Trainer</div>
        Learn how to start the <b>training</b> process.
    </a>
</div>
<p></p>

## Model

```{eval-rst}
Multi-vector models consist of a sequence of `Modules <../package_reference/sentence_transformer/modules.html>`_, `Multi-Vector Encoder specific Modules <../package_reference/multi_vector_encoder/modules.html>`_ or `Custom Modules <usage/custom_models.html>`_, allowing for a lot of flexibility. If you want to further finetune an existing multi-vector model (e.g. it has a `modules.json file <https://huggingface.co/lightonai/GTE-ModernColBERT-v1/tree/main/modules.json>`_), then you don't have to worry about which modules are used::

    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")

But if instead you want to train from a base transformer model, the classic ColBERT architecture is the default: a :class:`~sentence_transformers.base.modules.Transformer` producing contextualized token embeddings, a token-level :class:`~sentence_transformers.base.modules.Dense` projecting each token down to the multi-vector dimension (classically 128), a :class:`~sentence_transformers.multi_vector_encoder.modules.MultiVectorMask` computing the per-token scoring mask, and a token-level :class:`~sentence_transformers.sentence_transformer.modules.Normalize`:

.. sidebar:: Documentation

    - :class:`sentence_transformers.base.modules.Transformer`
    - :class:`sentence_transformers.base.modules.Dense`
    - :class:`sentence_transformers.multi_vector_encoder.modules.MultiVectorMask`
    - :class:`sentence_transformers.sentence_transformer.modules.Normalize`

::

    from sentence_transformers import MultiVectorEncoder

    # Loading in fp32 is preferred for training if your memory can handle it
    model = MultiVectorEncoder("answerdotai/ModernBERT-base", model_kwargs={"torch_dtype": "float32"})
    # MultiVectorEncoder(
    #   (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'ModernBertModel'})
    #   (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, 'activation_function': 'torch.nn.modules.linear.Identity', 'module_input_name': 'token_embeddings', 'module_output_name': 'token_embeddings'})
    #   (2): MultiVectorMask({'skiplist_words': [], 'keep_only_token_ids': None})
    #   (3): Normalize({'module_input_name': 'token_embeddings', 'module_output_name': 'token_embeddings'})
    # )

The fresh projection is randomly initialized, so training is required before this model is useful.

Note that this default leaves the classic ColBERT tokenization tricks off: no ``[Q]`` / ``[D]`` prefix tokens, no query expansion, no per-task lengths, and no punctuation skiplist. To reproduce the full classic ColBERT recipe, configure them explicitly::

    from torch import nn

    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.base.modules import Dense, Transformer
    from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask
    from sentence_transformers.sentence_transformer.modules import Normalize
    import string

    transformer = Transformer(
        "answerdotai/ModernBERT-base",
        query_expansion={"strategy": "pad_skip", "length": 32},  # pad queries to 32 tokens with [MASK], truncate longer ones
        document_length=300,  # also truncate (not pad) documents to 300 tokens
        model_kwargs={"torch_dtype": "float32"},
    )
    dense = Dense(
        in_features=transformer.get_embedding_dimension(),
        out_features=128,
        bias=False,
        activation_function=nn.Identity(),
        module_input_name="token_embeddings",
    )
    mask = MultiVectorMask(skiplist_words=list(string.punctuation))  # exclude punctuation from document scoring
    normalize = Normalize(module_input_name="token_embeddings")

    model = MultiVectorEncoder(
        modules=[transformer, dense, mask, normalize],
        prompts={"query": "[Q] ", "document": "[D] "},
    )

See `Creating Custom Models <usage/custom_models.html>`_ for more details on the module pipeline, including how to add extra per-token feature channels.
```

## Dataset

```{eval-rst}
The :class:`~sentence_transformers.multi_vector_encoder.trainer.MultiVectorEncoderTrainer` trains and evaluates using :class:`datasets.Dataset` (one dataset) or :class:`datasets.DatasetDict` instances (multiple datasets, see also `Multi-dataset training <#multi-dataset-training>`_).

.. tab:: Data on 🤗 Hugging Face Hub

    If you want to load data from the `Hugging Face Datasets <https://huggingface.co/datasets>`_, then you should use :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#hugging-face-hub">Datasets, Loading from the Hugging Face Hub</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/msmarco-bm25">sentence-transformers/msmarco-bm25</a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset

        train_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")

        print(train_dataset)
        """
        Dataset({
            features: ['query', 'positive', 'negative'],
            num_rows: 502931
        })
        """

    .. note::

        Many Hugging Face datasets that work out of the box with Sentence Transformers have been tagged with ``sentence-transformers``, allowing you to easily find them by browsing to `https://huggingface.co/datasets?other=sentence-transformers <https://huggingface.co/datasets?other=sentence-transformers>`_. We strongly recommend that you browse these datasets to find training datasets that might be useful for your tasks.

.. tab:: Local Data (CSV, JSON, Parquet, Arrow, SQL)

    If you have local data in common file-formats, then you can load this data easily using :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#local-and-remote-files">Datasets, Loading local files</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset

        dataset = load_dataset("csv", data_files="my_file.csv")

    or::

        from datasets import load_dataset

        dataset = load_dataset("json", data_files="my_file.json")

.. tab:: Local Data that requires pre-processing

    If you have local data that requires some extra pre-processing, my recommendation is to initialize your dataset using :meth:`datasets.Dataset.from_dict` and a dictionary of lists, like so:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict" title="(in datasets vmain)"><code class="xref py py-meth docutils literal notranslate"><span class="pre">datasets.Dataset.from_dict()</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import Dataset

        queries = []
        positives = []
        # Open a file, do preprocessing, filtering, cleaning, etc.
        # and append to the lists

        dataset = Dataset.from_dict({
            "query": queries,
            "positive": positives,
        })

    Each key from the dictionary will become a column in the resulting dataset.

```

### Dataset Format

```{eval-rst}
It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format). Verifying whether a dataset format works with a loss function involves two steps:

1. If your loss function requires a *Label* according to the `Loss Overview <loss_overview.html>`_ table, then your dataset must have a **column named "label" or "score"**. This column is automatically taken as the label.
2. All columns not named "label" or "score" are considered *Inputs* according to the `Loss Overview <loss_overview.html>`_ table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are **irrelevant**, only the **order matters**.

Be sure to re-order your dataset columns with :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` if your columns are not ordered correctly. For example, if your dataset has ``["good_answer", "bad_answer", "question"]`` as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the ``good_answer`` column will be taken as the query, ``bad_answer`` as the positive document, and ``question`` as the negative document.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with :meth:`Dataset.remove_columns <datasets.Dataset.remove_columns>` as they will be used as inputs otherwise. You can also use :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` to keep only the desired columns.

There are two multi-vector specific conventions on top of this:

- **Knowledge distillation format**: one column per candidate document, i.e. ``(query, document_1, ..., document_N, scores)`` where ``scores`` is a list of N teacher scores per row. This is the same multi-column convention as ``(query, positive, negative_1, ...)``, read positionally by :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss`. For KD datasets that store query / document *IDs* alongside separate text datasets (e.g. `lightonai/ms-marco-en-bge <https://huggingface.co/datasets/lightonai/ms-marco-en-bge>`_), you can use :func:`~sentence_transformers.util.dataset.resolve_ids` to resolve the IDs on the fly: it expands the stored ID list into the numbered document columns.
- **Positional query / document assignment**: the first column is embedded as the *query* and all following columns as *documents*. This default can be overridden per column via the standard ``router_mapping`` training argument, mapping column names to ``"query"`` or ``"document"``.
```

## Loss Function

Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See [Dataset Format](#dataset-format) to learn what datasets are valid for which loss functions. Additionally, the [Loss Overview](loss_overview) will be your best friend to learn about the options.

```{eval-rst}
Most loss functions can be initialized with just the :class:`~sentence_transformers.MultiVectorEncoder` that you're training, alongside some optional parameters, e.g.:

.. sidebar:: Documentation

    - :class:`sentence_transformers.multi_vector_encoder.losses.MultiVectorMultipleNegativesRankingLoss`
    - `Losses API Reference <../package_reference/multi_vector_encoder/losses.html>`_
    - `Loss Overview <loss_overview.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss

    # Load a model to train/finetune
    # Loading in fp32 is preferred for training if your memory can handle it
    model = MultiVectorEncoder("answerdotai/ModernBERT-base", model_kwargs={"torch_dtype": "float32"})

    # Initialize the loss: in-batch negatives, scored with MaxSim
    loss = MultiVectorMultipleNegativesRankingLoss(model=model)

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train")
    print(train_dataset)
    """
    Dataset({
        features: ['query', 'positive', 'negative'],
        num_rows: 502931
    })
    """
```

## Training Arguments

```{eval-rst}
The :class:`~sentence_transformers.MultiVectorEncoderTrainingArguments` class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.
```

<div class="training-arguments">
    <div class="header">Key Training Arguments for improving training performance</div>
    <div class="table">
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.learning_rate"><code>learning_rate</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.lr_scheduler_type"><code>lr_scheduler_type</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.warmup_steps"><code>warmup_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.num_train_epochs"><code>num_train_epochs</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.max_steps"><code>max_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size"><code>per_device_train_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.per_device_eval_batch_size"><code>per_device_eval_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.auto_find_batch_size "><code>auto_find_batch_size</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.fp16"><code>fp16</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.bf16"><code>bf16</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.load_best_model_at_end"><code>load_best_model_at_end</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.metric_for_best_model"><code>metric_for_best_model</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps"><code>gradient_accumulation_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.gradient_checkpointing"><code>gradient_checkpointing</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_accumulation_steps"><code>eval_accumulation_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.optim"><code>optim</code></a>
        <a href="../package_reference/multi_vector_encoder/training_args.html"><code>batch_sampler</code></a>
        <a href="../package_reference/multi_vector_encoder/training_args.html"><code>multi_dataset_batch_sampler</code></a>
        <a href="../package_reference/multi_vector_encoder/training_args.html"><code>router_mapping</code></a>
        <a href="../package_reference/multi_vector_encoder/training_args.html"><code>learning_rate_mapping</code></a>
    </div>
</div>
<br>
<div class="training-arguments">
    <div class="header">Key Training Arguments for observing training performance</div>
    <div class="table">
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_strategy"><code>eval_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.eval_steps"><code>eval_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_strategy"><code>save_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_steps"><code>save_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_total_limit"><code>save_total_limit</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.report_to"><code>report_to</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name"><code>run_name</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.log_level"><code>log_level</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.logging_steps"><code>logging_steps</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.push_to_hub"><code>push_to_hub</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_model_id"><code>hub_model_id</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_strategy"><code>hub_strategy</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.hub_private_repo"><code>hub_private_repo</code></a>
    </div>
</div>
<br>

```{eval-rst}
Here is an example of how :class:`~sentence_transformers.MultiVectorEncoderTrainingArguments` can be initialized:
```

```python
args = MultiVectorEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/multivector-ModernBERT-base-msmarco",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    warmup_steps=0.05,
    fp16=False,  # Set to True if your GPU doesn't support BF16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=0.01,
    run_name="multivector-ModernBERT-base-msmarco",  # Will be used in W&B if `wandb` is installed
)
```

## Evaluator

```{eval-rst}
You can provide the :class:`~sentence_transformers.multi_vector_encoder.trainer.MultiVectorEncoderTrainer` with an ``eval_dataset`` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an ``eval_dataset`` and an evaluator, one or the other, or neither. They evaluate based on the ``eval_strategy`` and ``eval_steps`` `Training Arguments <#training-arguments>`_.

Here are the implemented Evaluators that come with Sentence Transformers for Multi-Vector Encoder models:

========================================================================================================  ===========================================================================================================================
Evaluator                                                                                                 Required Data
========================================================================================================  ===========================================================================================================================
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorInformationRetrievalEvaluator`  Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]).
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorNanoBEIREvaluator`              No data required.
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorTripletEvaluator`               (anchor, positive, negative) triplets.
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorRerankingEvaluator`             List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries.
:class:`~sentence_transformers.multi_vector_encoder.evaluation.MultiVectorDistillationEvaluator`          Queries with candidate documents and teacher scores.
========================================================================================================  ===========================================================================================================================

Additionally, :class:`~sentence_transformers.sentence_transformer.evaluation.SequentialEvaluator` should be used to combine multiple evaluators into one Evaluator that can be passed to the :class:`~sentence_transformers.multi_vector_encoder.trainer.MultiVectorEncoderTrainer`.

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

.. tab:: MultiVectorNanoBEIREvaluator

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/multi_vector_encoder/evaluation.html"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.multi_vector_encoder.evaluation.MultiVectorNanoBEIREvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator

        # Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
        # directly from Hugging Face, so there's no mandatory arguments
        dev_evaluator = MultiVectorNanoBEIREvaluator()
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: MultiVectorTripletEvaluator with MS MARCO

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/msmarco-bm25">sentence-transformers/msmarco-bm25</a></li>
                <li><a class="reference internal" href="../package_reference/multi_vector_encoder/evaluation.html"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.multi_vector_encoder.evaluation.MultiVectorTripletEvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorTripletEvaluator

        # Load triplets from the MS MARCO dataset (https://huggingface.co/datasets/sentence-transformers/msmarco-bm25)
        max_samples = 1000
        eval_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split=f"train[:{max_samples}]")

        # Initialize the evaluator
        dev_evaluator = MultiVectorTripletEvaluator(
            anchors=eval_dataset["query"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="msmarco-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tip::

    When evaluating frequently during training with a small ``eval_steps``, consider using a tiny ``eval_dataset`` to minimize evaluation overhead. If you're concerned about the evaluation set size, a 90-1-9 train-eval-test split can provide a balance, reserving a reasonably sized test set for final evaluations. After training, you can assess your model's performance using ``trainer.evaluate(test_dataset)`` for test loss or initialize a testing evaluator with ``test_evaluator(model)`` for detailed test metrics.

    If you evaluate after training, but before saving the model, your automatically generated model card will still include the test results.

.. warning::

    When using `Distributed Training <../sentence_transformer/training/distributed.html>`_, the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices.
```

## Trainer

```{eval-rst}
The :class:`~sentence_transformers.MultiVectorEncoderTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let's have a look at a script where all of these components come together:

.. tab:: Contrastive (in-batch negatives)

    ::

        import logging

        from datasets import load_dataset

        from sentence_transformers import (
            MultiVectorEncoder,
            MultiVectorEncoderModelCardData,
            MultiVectorEncoderTrainer,
            MultiVectorEncoderTrainingArguments,
        )
        from sentence_transformers.base.sampler import BatchSamplers
        from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
        from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss

        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

        # 1. Load a model to finetune with 2. (Optional) model card data
        # Loading in fp32 is preferred for training if your memory can handle it
        model = MultiVectorEncoder(
            "answerdotai/ModernBERT-base",
            model_card_data=MultiVectorEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name="ColBERT ModernBERT-base trained on MS MARCO triplets",
            ),
            model_kwargs={"torch_dtype": "float32"},
        )

        # 3. Load a dataset to finetune on
        full_dataset = load_dataset("sentence-transformers/msmarco-bm25", "triplet", split="train").select(range(51_000))
        dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

        # 4. Define a loss function: in-batch negatives, scored with MaxSim
        loss = MultiVectorMultipleNegativesRankingLoss(model=model)

        # 5. (Optional) Specify training arguments
        run_name = "multivector-ModernBERT-base-msmarco"
        args = MultiVectorEncoderTrainingArguments(
            # Required parameter:
            output_dir=f"models/{run_name}",
            # Optional training parameters:
            num_train_epochs=1,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=3e-5,
            warmup_steps=0.05,
            fp16=False,  # Set to True if your GPU doesn't support BF16
            bf16=True,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            load_best_model_at_end=True,
            metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10",
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=0.1,
            save_strategy="steps",
            save_steps=0.1,
            save_total_limit=2,
            logging_steps=0.01,
            run_name=run_name,  # Will be used in W&B if `wandb` is installed
        )

        # 6. (Optional) Create an evaluator & evaluate the base model
        dev_evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=32)
        dev_evaluator(model)

        # 7. Create a trainer & train
        trainer = MultiVectorEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        trainer.train()

        # 8. Evaluate the model performance again after training
        dev_evaluator(model)

        # 9. Save the trained model
        model.save_pretrained(f"models/{run_name}/final")

        # 10. (Optional) Push it to the Hugging Face Hub
        model.push_to_hub(run_name)

.. tab:: Knowledge Distillation

    The strongest late-interaction models are trained by distilling the rankings of a strong teacher (e.g. a `Cross Encoder <../cross_encoder/usage/usage.html>`_) over N candidate documents per query, rather than from raw pairs or triplets. `lightonai/ms-marco-en-bge <https://huggingface.co/datasets/lightonai/ms-marco-en-bge>`_ provides exactly that: per-query candidate document IDs with teacher scores, resolved to texts on the fly by :func:`~sentence_transformers.util.dataset.resolve_ids`.

    ::

        import logging

        from datasets import load_dataset

        from sentence_transformers import (
            MultiVectorEncoder,
            MultiVectorEncoderModelCardData,
            MultiVectorEncoderTrainer,
            MultiVectorEncoderTrainingArguments,
        )
        from sentence_transformers.util import resolve_ids
        from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoBEIREvaluator
        from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss

        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

        max_list_length = 32  # Number of candidate documents (1 positive + negatives) scored per query
        train_batch_size = 4  # Each batch holds train_batch_size * max_list_length documents, so keep it small

        # 1. Load a model to finetune with 2. (Optional) model card data
        # Loading in fp32 is preferred for training if your memory can handle it
        model = MultiVectorEncoder(
            "answerdotai/ModernBERT-base",
            model_card_data=MultiVectorEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name="ColBERT ModernBERT-base distilled from BGE on MS MARCO",
            ),
            model_kwargs={"torch_dtype": "float32"},
        )

        # 3. Load the KD dataset: per-query candidate ids + teacher scores, with separate text datasets
        train_dataset = load_dataset("lightonai/ms-marco-en-bge", "train", split="train").select(range(20_000))
        queries = load_dataset("lightonai/ms-marco-en-bge", "queries", split="train")
        documents = load_dataset("lightonai/ms-marco-en-bge", "documents", split="train")

        # resolve_ids resolves query_id -> query text and document_ids -> document texts on the fly.
        train_dataset.set_transform(
            resolve_ids({"query_id": queries, "document_ids": documents}, max_list_length=max_list_length)
        )

        # 4. Define a loss function: KL divergence between the teacher and student score distributions
        loss = MultiVectorDistillKLDivLoss(model=model)

        # 5. (Optional) Specify training arguments
        run_name = "multivector-ModernBERT-base-msmarco-kd"
        args = MultiVectorEncoderTrainingArguments(
            # Required parameter:
            output_dir=f"models/{run_name}",
            # Optional training parameters:
            num_train_epochs=1,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=train_batch_size,
            learning_rate=3e-5,
            warmup_steps=0.05,
            fp16=False,  # Set to True if your GPU doesn't support BF16
            bf16=True,  # Set to True if you have a GPU that supports BF16
            load_best_model_at_end=True,
            metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10",
            # Optional tracking/debugging parameters:
            eval_strategy="steps",  # The NanoBEIR evaluator runs on its own datasets, so no eval_dataset is needed
            eval_steps=0.1,
            save_strategy="steps",
            save_steps=0.1,
            save_total_limit=2,
            logging_steps=0.01,
            run_name=run_name,  # Will be used in W&B if `wandb` is installed
        )

        # 6. (Optional) Create an evaluator & evaluate the base model
        dev_evaluator = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco", "nq", "fiqa2018"], batch_size=train_batch_size)
        dev_evaluator(model)

        # 7. Create a trainer & train
        trainer = MultiVectorEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        trainer.train()

        # 8. Evaluate the model performance again after training
        dev_evaluator(model)

        # 9. Save the trained model
        model.save_pretrained(f"models/{run_name}/final")

        # 10. (Optional) Push it to the Hugging Face Hub
        model.push_to_hub(run_name)
```

### Callbacks

```{eval-rst}
This Multi-Vector Encoder trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

- :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if ``wandb`` is installed
- :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if ``tensorboard`` is accessible.
- :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if ``codecarbon`` is installed.

    - Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
documentation for more information on the integrated callbacks and how to write your own callbacks.
```

## Multi-Dataset Training

```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`~sentence_transformers.MultiVectorEncoderTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

- Use a dictionary of :class:`~datasets.Dataset` instances (or a :class:`~datasets.DatasetDict`) as the ``train_dataset`` (and optionally also ``eval_dataset``).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the :class:`~sentence_transformers.sentence_transformer.training_args.MultiDatasetBatchSamplers` enum, which can be passed to the :class:`~sentence_transformers.MultiVectorEncoderTrainingArguments` via ``multi_dataset_batch_sampler``. Valid options are:

- ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted. With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- ``MultiDatasetBatchSamplers.PROPORTIONAL`` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.
```

## Training Tips

```{eval-rst}
Multi-Vector Encoder models have a few quirks that you should be aware of when training them:

1. The contrastive losses default to ``scale=1.0`` (i.e. ``temperature=1.0``), matching PyLate, and unlike the dense :class:`~sentence_transformers.sentence_transformer.losses.MultipleNegativesRankingLoss` default of ``scale=20.0``. That 20.0 exists to amplify *bounded* cosine similarity (``[-1, 1]``), but MaxSim is an *unbounded* sum over query-token similarities (range ``~[0, num_query_tokens]``), so it needs no amplification, exactly as the dense loss recommends ``scale=1`` for dot-product similarity. A large ``scale`` here would saturate the softmax and kill gradients.
2. The strongest late-interaction models are trained almost exclusively with n-way knowledge distillation from a stronger teacher model using :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss`, instead of training directly from text pairs or triplets. See the Knowledge Distillation tab under `Trainer <#trainer>`_.
3. In-batch negatives losses benefit heavily from larger batch sizes. If GPU memory is the bottleneck, :class:`~sentence_transformers.multi_vector_encoder.losses.CachedMultiVectorMultipleNegativesRankingLoss` reaches much larger effective batch sizes at a small speed cost via `GradCache <https://huggingface.co/papers/2101.06983>`_.
4. A fresh model from a base transformer starts without the classic ColBERT tokenization tricks (``[Q]`` / ``[D]`` prefixes, query expansion, per-task lengths, punctuation skiplist). They are worth configuring explicitly (see `Model <#model>`_): in particular query expansion and the prefix tokens are part of the classic recipe that most released checkpoints use.
5. Multi-vector models are evaluated (and scored during training) with MaxSim, and the per-query-token score contributions are inspectable: see the `interpretability utilities <../package_reference/multi_vector_encoder/interpretability.html>`_ for similarity maps and heatmaps on image documents.
```
