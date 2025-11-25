# Training Overview

## Why Finetune?

Cross Encoder models are very often used as 2nd stage rerankers in a [Retrieve and Rerank](../../examples/sentence_transformer/applications/retrieve_rerank/README.md) search stack. In such a situation, the Cross Encoder reranks the top X candidates from the retriever (which can be a [Sentence Transformer model](../sentence_transformer/usage/usage.rst)). To avoid the reranker model reducing the performance on your use case, finetuning it can be crucial. Rerankers always have just 1 output label.

Beyond that, Cross Encoder models can also be used as pair classifiers. For example, a model trained on Natural Language Inference data can be used to classify pairs of texts as "contradiction", "entailment", and "neutral". Pair Classifiers generally have more than 1 output label.

See [**Training Examples**](training/examples) for numerous training scripts for common real-world applications that you can adopt.

## Training Components

Training Cross Encoder models involves between 4 to 6 components, just like [training Sentence Transformer models](../sentence_transformer/training_overview.md):

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

Cross Encoder models are initialized by loading a pretrained `transformers <https://huggingface.co/docs/transformers>`_ model using a sequence classification head. If the model itself does not have such a head, then it will be added automatically. Consequently, initializing a Cross Encoder model is rather simple:

.. sidebar:: Documentation

    - :class:`sentence_transformers.cross_encoder.CrossEncoder`

::

    from sentence_transformers import CrossEncoder

    # This model already has a sequence classification head
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    # And this model does not, so it will be added automatically
    model = CrossEncoder("google-bert/bert-base-uncased")

.. tip::

    You can find pretrained reranker models in the `Cross Encoder > Pretrained Models <pretrained_models.html>`_ documentation.

    For other models, the strongest pretrained models are often "encoder models", i.e. models that are trained to produce a meaningful token embedding for inputs. You can find strong candidates here:

    - `fill-mask models <https://huggingface.co/models?pipeline_tag=fill-mask>`_ - trained for token embeddings
    - `sentence similarity models <https://huggingface.co/models?pipeline_tag=sentence-similarity>`_ - trained for text embeddings
    - `feature-extraction models <https://huggingface.co/models?pipeline_tag=feature-extraction>`_ - trained for text embeddings

    Consider looking for base models that are designed on your language and/or domain of interest. For example, `klue/bert-base <https://huggingface.co/klue/bert-base>`_ will work much better than `google-bert/bert-base-uncased <https://huggingface.co/google-bert/bert-base-uncased>`_ for Korean.

```

## Dataset

```{eval-rst}
The :class:`CrossEncoderTrainer` trains and evaluates using :class:`datasets.Dataset` (one dataset) or :class:`datasets.DatasetDict` instances (multiple datasets, see also `Multi-dataset training <#multi-dataset-training>`_). 

.. tab:: Data on 🤗 Hugging Face Hub

    If you want to load data from the `Hugging Face Datasets <https://huggingface.co/datasets>`_, then you should use :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#hugging-face-hub">Datasets, Loading from the Hugging Face Hub</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset

        train_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
        eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev")

        print(train_dataset)
        """
        Dataset({
            features: ['premise', 'hypothesis', 'label'],
            num_rows: 942069
        })
        """

    Some datasets (including `sentence-transformers/all-nli <https://huggingface.co/datasets/sentence-transformers/all-nli>`_) require you to provide a "subset" alongside the dataset name. ``sentence-transformers/all-nli`` has 4 subsets, each with different data formats: `pair <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair>`_, `pair-class <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair-class>`_, `pair-score <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/pair-score>`_, `triplet <https://huggingface.co/datasets/sentence-transformers/all-nli/viewer/triplet>`_.

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

        anchors = []
        positives = []
        # Open a file, do preprocessing, filtering, cleaning, etc.
        # and append to the lists

        dataset = Dataset.from_dict({
            "anchor": anchors,
            "positive": positives,
        })

    Each key from the dictionary will become a column in the resulting dataset.

```

### Dataset Format

```{eval-rst}
It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format and model). Verifying whether a dataset format and model work with a loss function involves three steps:

1. All columns not named "label", "labels", "score", or "scores" are considered *Inputs* according to the `Loss Overview <loss_overview.html>`_ table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are **irrelevant**, only the **order matters**. 
2. If your loss function requires a *Label* according to the `Loss Overview <loss_overview.html>`_ table, then your dataset must have a **column named "label", "labels", "score", or "scores"**. This column is automatically taken as the label.
3. The number of model output labels matches what is required for the loss according to `Loss Overview <loss_overview.html>`_ table.

For example, given a dataset with columns ``["text1", "text2", "label"]`` where the "label" column has float similarity score ranging from 0 to 1 and a model outputting 1 label, we can use it with :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss` because:

1. the dataset has a "label" column as is required for this loss function.
2. the dataset has 2 non-label columns, exactly the amount required by this loss functions.
3. the model has 1 output label, exactly as required by this loss function.

Be sure to re-order your dataset columns with :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` if your columns are not ordered correctly. For example, if your dataset has ``["good_answer", "bad_answer", "question"]`` as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the ``good_answer`` column will be taken as the anchor, ``bad_answer`` as the positive, and ``question`` as the negative.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with :meth:`Dataset.remove_columns <datasets.Dataset.remove_columns>` as they will be used as inputs otherwise. You can also use :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` to keep only the desired columns.
```

### Hard Negatives Mining

The success of training CrossEncoder models often depends on the quality of the *negatives*, i.e. the passages for which the query-negative score should be low. Negatives can be divided into two types:

- **Soft negatives**: passages that are completely unrelated.
- **Hard negatives**: passages that seem like they might be relevant for the query, but are not.

A concise example is:

- **Query**: Where was Apple founded?
- **Soft Negative**: The Cache River Bridge is a Parker pony truss that spans the Cache River between Walnut Ridge and Paragould, Arkansas.
- **Hard Negative**: The Fuji apple is an apple cultivar developed in the late 1930s, and brought to market in 1962.

```{eval-rst}
The strongest CrossEncoder models are generally trained to recognize hard negatives, and so it's valuable to be able to "mine" hard negatives. Sentence Transformers supports a strong :func:`~sentence_transformers.util.mine_hard_negatives` function that can assist, given a dataset of query-answer pairs:

.. sidebar:: Documentation

    * `sentence-transformers/gooaq <https://huggingface.co/datasets/sentence-transformers/gooaq>`_
    * `sentence-transformers/static-retrieval-mrl-en-v1 <https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1>`_
    * :class:`~sentence_transformers.SentenceTransformer`
    * :func:`~sentence_transformers.util.mine_hard_negatives`

::

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives

    # Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    train_dataset = load_dataset("sentence-transformers/gooaq", split=f"train").select(range(100_000))
    print(train_dataset)

    # Mine hard negatives using a very efficient embedding model
    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=5,  # How many negatives per question-answer pair
        range_min=10,  # Skip the x most similar samples
        range_max=100,  # Consider only the x most similar samples
        max_score=0.8,  # Only consider samples with a similarity score of at most x
        absolute_margin=0.1,  # Anchor-negative similarity is at least x lower than anchor-positive similarity
        relative_margin=0.1,  # Anchor-negative similarity is at most 1-x times the anchor-positive similarity, e.g. 90%
        sampling_strategy="top",  # Sample the top negatives from the range
        batch_size=4096,  # Use a batch size of 4096 for the embedding model
        output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
        use_faiss=True,  # Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)
    )
    print(hard_train_dataset)
    print(hard_train_dataset[1])

```

<details><summary>Click to see the outputs of this script.</summary>

```
Dataset({
    features: ['question', 'answer'],
    num_rows: 100000
})

Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:01<00:00, 12.74it/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:00<00:00, 37.50it/s]
Querying FAISS index: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:18<00:00,  2.66s/it]
Metric       Positive       Negative     Difference
Count         100,000        436,925
Mean           0.5882         0.4040         0.2157
Median         0.5989         0.4024         0.1836
Std            0.1425         0.0905         0.1013
Min           -0.0514         0.1405         0.1014
25%            0.4993         0.3377         0.1352
50%            0.5989         0.4024         0.1836
75%            0.6888         0.4681         0.2699
Max            0.9748         0.7486         0.7545
Skipped 2,420,871 potential negatives (23.97%) due to the absolute_margin of 0.1.
Skipped 43 potential negatives (0.00%) due to the max_score of 0.8.
Could not find enough negatives for 63075 samples (12.62%). Consider adjusting the range_max, range_min, absolute_margin, relative_margin and max_score parameters if you'd like to find more valid negatives.
Dataset({
    features: ['question', 'answer', 'label'],
    num_rows: 536925
})

{
    'question': 'how to transfer bookmarks from one laptop to another?',
    'answer': 'Using an External Drive Just about any external drive, including a USB thumb drive, or an SD card can be used to transfer your files from one laptop to another. Connect the drive to your old laptop; drag your files to the drive, then disconnect it and transfer the drive contents onto your new laptop.',
    'label': 0
}
```

</details>
<br>

## Loss Function

Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See [Dataset Format](#dataset-format) to learn what datasets are valid for which loss functions. Additionally, the [Loss Overview](loss_overview) will be your best friend to learn about the options.

```{eval-rst}
Most loss functions can be initialized with just the :class:`~sentence_transformers.cross_encoder.CrossEncoder` that you're training, alongside some optional parameters, e.g.:

.. sidebar:: Documentation

    - :class:`sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss`
    - `Losses API Reference <../package_reference/cross_encoder/losses.html>`_
    - `Loss Overview <loss_overview.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.losses import MultipleNegativesRankingLoss

    # Load a model to train/finetune
    model = CrossEncoder("xlm-roberta-base", num_labels=1) # num_labels=1 is for rerankers

    # Initialize the MultipleNegativesRankingLoss
    # This loss requires pairs of related texts or triplets
    loss = MultipleNegativesRankingLoss(model)

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset("sentence-transformers/gooaq", split="train")
```

## Training Arguments

```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments` class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.
```

<div class="training-arguments">
    <div class="header">Key Training Arguments for improving training performance</div>
    <div class="table">
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.learning_rate"><code>learning_rate</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.lr_scheduler_type"><code>lr_scheduler_type</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.warmup_ratio"><code>warmup_ratio</code></a>
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
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.dataloader_num_workers"><code>dataloader_num_workers</code></a>
        <a href="https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.dataloader_prefetch_factor"><code>dataloader_prefetch_factor</code></a>
        <a href="../package_reference/cross_encoder/training_args.html#sentence_transformers.cross_encoder.training_args.SentenceTransformerTrainingArguments"><code>batch_sampler</code></a>
        <a href="../package_reference/cross_encoder/training_args.html#sentence_transformers.cross_encoder.training_args.SentenceTransformerTrainingArguments"><code>multi_dataset_batch_sampler</code></a>
        <a href="../package_reference/cross_encoder/training_args.html#sentence_transformers.cross_encoder.training_args.SentenceTransformerTrainingArguments"><code>learning_rate_mapping</code></a>
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
Here is an example of how :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments` can be initialized:
```

```python
from sentence_transformers.cross_encoder import CrossEncoderTrainingArguments

args = CrossEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/reranker-MiniLM-msmarco-v1",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="reranker-MiniLM-msmarco-v1",  # Will be used in W&B if `wandb` is installed
)
```

## Evaluator

```{eval-rst}
You can provide the :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` with an ``eval_dataset`` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an ``eval_dataset`` and an evaluator, one or the other, or neither. They evaluate based on the ``eval_strategy`` and ``eval_steps`` `Training Arguments <#training-arguments>`_.

Here are the implemented Evaluators that come with Sentence Transformers for Cross Encoder models:

=============================================================================================  ========================================================================================================================================================================
Evaluator                                                                                      Required Data
=============================================================================================  ========================================================================================================================================================================
:class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`   Pairs with class labels (binary or multiclass).
:class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator`      Pairs with similarity scores.
:class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`         No data required.
:class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`        List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries. Negatives can be mined with :func:`~sentence_transformers.util.mine_hard_negatives`.
=============================================================================================  ========================================================================================================================================================================

Additionally, :class:`~sentence_transformers.evaluation.SequentialEvaluator` should be used to combine multiple evaluators into one Evaluator that can be passed to the :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer`.

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

.. tab:: CrossEncoderNanoBEIREvaluator

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2">cross-encoder/ms-marco-MiniLM-L6-v2</a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.CrossEncoderNanoBEIREvaluator" title="sentence_transformers.evaluation.CrossEncoderNanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.evaluation.CrossEncoderNanoBEIREvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

        # Load a model
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

        # Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
        # directly from Hugging Face, so there's no mandatory arguments
        dev_evaluator = CrossEncoderNanoBEIREvaluator()
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: CrossEncoderRerankingEvaluator with GooAQ mined negatives

    Preparing data for :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator` can be difficult as you need negatives in addition to your query-positive data.

    The :func:`~sentence_transformers.util.mine_hard_negatives` function has a convenient ``include_positives`` parameter, which can be set to ``True`` to also mine for the positive texts. When supplied as ``documents`` (which have to be 1. ranked and 2. contain positives) to :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`, the evaluator will not just evaluate the reranking performance of the CrossEncoder, but also the original rankings by the embedding model used for mining.

    For example::

        CrossEncoderRerankingEvaluator: Evaluating the model on the gooaq-dev dataset:
        Queries:  1000     Positives: Min 1.0, Mean 1.0, Max 1.0   Negatives: Min 49.0, Mean 49.1, Max 50.0
                  Base  -> Reranked
        MAP:      53.28 -> 67.28
        MRR@10:   52.40 -> 66.65
        NDCG@10:  59.12 -> 71.35

    Note that by default, if you are using :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator` with ``documents``, the evaluator will rerank with *all* positives, even if they are not in the documents. This is useful for getting a stronger signal out of your evaluator, but does give a slightly unrealistic performance. After all, the maximum performance is now 100, whereas normally its bounded by whether the first-stage retriever actually retrieved the positives.

    You can enable the realistic behaviour by setting ``always_rerank_positives=False`` when initializing :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`. Repeating the same script with this realistic two-stage performance results in::

        CrossEncoderRerankingEvaluator: Evaluating the model on the gooaq-dev dataset:
        Queries:  1000     Positives: Min 1.0, Mean 1.0, Max 1.0   Negatives: Min 49.0, Mean 49.1, Max 50.0
                  Base  -> Reranked
        MAP:      53.28 -> 66.12
        MRR@10:   52.40 -> 65.61
        NDCG@10:  59.12 -> 70.10

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2">cross-encoder/ms-marco-MiniLM-L6-v2</a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/gooaq">sentence-transformers/gooaq</a></li>
                <li><a class="reference internal" href="../package_reference/util.html#sentence_transformers.util.mine_hard_negatives" title="sentence_transformers.util.mine_hard_negatives"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.util.mine_hard_negatives</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator" title="sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
        from sentence_transformers.util import mine_hard_negatives

        # Load a model
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

        # Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
        full_dataset = load_dataset("sentence-transformers/gooaq", split=f"train").select(range(100_000))
        dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
        print(eval_dataset)
        """
        Dataset({
            features: ['question', 'answer'],
            num_rows: 1000
        })
        """

        # Mine hard negatives using a very efficient embedding model
        embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
        hard_eval_dataset = mine_hard_negatives(
            eval_dataset,
            embedding_model,
            corpus=full_dataset["answer"],  # Use the full dataset as the corpus
            num_negatives=50,  # How many negatives per question-answer pair
            batch_size=4096,  # Use a batch size of 4096 for the embedding model
            output_format="n-tuple",  # The output format is (query, positive, negative1, negative2, ...) for the evaluator
            include_positives=True,  # Key: Include the positive answer in the list of negatives
            use_faiss=True,  # Using FAISS is recommended to keep memory usage low (pip install faiss-gpu or pip install faiss-cpu)
        )
        print(hard_eval_dataset)
        """
        Dataset({
            features: ['question', 'answer', 'negative_1', 'negative_2', 'negative_3', 'negative_4', 'negative_5', 'negative_6', 'negative_7', 'negative_8', 'negative_9', 'negative_10', 'negative_11', 'negative_12', 'negative_13', 'negative_14', 'negative_15', 'negative_16', 'negative_17', 'negative_18', 'negative_19', 'negative_20', 'negative_21', 'negative_22', 'negative_23', 'negative_24', 'negative_25', 'negative_26', 'negative_27', 'negative_28', 'negative_29', 'negative_30', 'negative_31', 'negative_32', 'negative_33', 'negative_34', 'negative_35', 'negative_36', 'negative_37', 'negative_38', 'negative_39', 'negative_40', 'negative_41', 'negative_42', 'negative_43', 'negative_44', 'negative_45', 'negative_46', 'negative_47', 'negative_48', 'negative_49', 'negative_50'],
            num_rows: 1000
        })
        """

        reranking_evaluator = CrossEncoderRerankingEvaluator(
            samples=[
                {
                    "query": sample["question"],
                    "positive": [sample["answer"]],
                    "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
                }
                for sample in hard_eval_dataset
            ],
            batch_size=32,
            name="gooaq-dev",
        )
        # You can run evaluation like so
        results = reranking_evaluator(model)
        """
        CrossEncoderRerankingEvaluator: Evaluating the model on the gooaq-dev dataset:
        Queries:  1000     Positives: Min 1.0, Mean 1.0, Max 1.0   Negatives: Min 49.0, Mean 49.1, Max 50.0
                  Base  -> Reranked
        MAP:      53.28 -> 67.28
        MRR@10:   52.40 -> 66.65
        NDCG@10:  59.12 -> 71.35
        """
        # {'gooaq-dev_map': 0.6728370126462222, 'gooaq-dev_mrr@10': 0.6665190476190477, 'gooaq-dev_ndcg@10': 0.7135068904582963, 'gooaq-dev_base_map': 0.5327714512001362, 'gooaq-dev_base_mrr@10': 0.5239674603174603, 'gooaq-dev_base_ndcg@10': 0.5912299141913905}


.. tab:: CrossEncoderCorrelationEvaluator with STSb

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/cross-encoder/stsb-TinyBERT-L4">cross-encoder/stsb-TinyBERT-L4</a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/stsb">sentence-transformers/stsb</a></li>
                <li><a class="reference internal" href="../package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator" title="sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator

        # Load a model
        model = CrossEncoder("cross-encoder/stsb-TinyBERT-L4")

        # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
        pairs = list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"]))

        # Initialize the evaluator
        dev_evaluator = CrossEncoderCorrelationEvaluator(
            sentence_pairs=pairs,
            scores=eval_dataset["score"],
            name="sts_dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: CrossEncoderClassificationEvaluator with AllNLI

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/cross-encoder/nli-deberta-v3-base">cross-encoder/nli-deberta-v3-base</a></li>
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator" title="sentence_transformers.evaluation.TripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.evaluation.TripletEvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers import CrossEncoder
        from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction

        # Load a model
        model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

        # Load triplets from the AllNLI dataset (https://huggingface.co/datasets/sentence-transformers/all-nli)
        max_samples = 1000
        eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split=f"dev[:{max_samples}]")

        # Create a list of pairs, and map the labels to the labels that the model knows
        pairs = list(zip(eval_dataset["premise"], eval_dataset["hypothesis"]))
        label_mapping = {0: 1, 1: 2, 2: 0}
        labels = [label_mapping[label] for label in eval_dataset["label"]]

        # Initialize the evaluator
        cls_evaluator = CrossEncoderClassificationEvaluator(
            sentence_pairs=pairs,
            labels=labels,
            name="all-nli-dev",
        )
        # You can run evaluation like so:
        # results = cls_evaluator(model)

.. warning::

    When using `Distributed Training <training/distributed.html>`_, the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices. 
```

## Trainer

```{eval-rst}
The :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let's have a look at a script where all of these components come together:

.. tab:: Simple Example

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ol class="arabic simple">
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder" title="sentence_transformers.cross_encoder.CrossEncoder"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoder</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.model_card.CrossEncoderModelCardData" title="sentence_transformers.cross_encoder.model_card.CrossEncoderModelCardData"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderModelCardData</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">load_dataset()</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/gooaq">sentence-transformers/gooaq</a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/losses.html#sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss" title="sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss"><code class="xref py py-class docutils literal notranslate"><span class="pre">CachedMultipleNegativesRankingLoss</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator" title="sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderNanoBEIREvaluator</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/training_args.html#sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments" title="sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderTrainingArguments</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/trainer.html#sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer" title="sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderTrainer</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/trainer.html#sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer.train" title="sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer.train"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoderTrainer.train()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.save_pretrained" title="sentence_transformers.cross_encoder.CrossEncoder.save_pretrained"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoder.save_pretrained()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.push_to_hub" title="sentence_transformers.cross_encoder.CrossEncoder.push_to_hub"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoder.push_to_hub()</span></code></a></p></li>
            </ol>
        </div>

    ::

        import logging
        import traceback

        from datasets import load_dataset

        from sentence_transformers.cross_encoder import (
            CrossEncoder,
            CrossEncoderModelCardData,
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
        )
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
        from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

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
            mini_batch_size=32,  # Informs the memory usage
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
            warmup_ratio=0.1,
            fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=True,  # Set to True if you have a GPU that supports BF16
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=50,
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


.. tab:: Extensive Example

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ol class="arabic simple">
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder" title="sentence_transformers.cross_encoder.CrossEncoder"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoder</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.model_card.CrossEncoderModelCardData" title="sentence_transformers.cross_encoder.model_card.CrossEncoderModelCardData"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderModelCardData</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">load_dataset()</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/gooaq">sentence-transformers/gooaq</a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer" title="sentence_transformers.SentenceTransformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">SentenceTransformer</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/util.html#sentence_transformers.util.mine_hard_negatives" title="sentence_transformers.util.mine_hard_negatives"><code class="xref py py-class docutils literal notranslate"><span class="pre">mine_hard_negatives</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/losses.html#sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss" title="sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss"><code class="xref py py-class docutils literal notranslate"><span class="pre">BinaryCrossEntropyLoss</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/evaluation.html#sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator" title="sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderNanoBEIREvaluator</span></code></a></p></li>
                <li><p><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderRerankingEvaluators</span></code></p></li>
                <li><p><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.SequentialEvaluator" title="sentence_transformers.evaluation.SequentialEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">SequentialEvaluator</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/training_args.html#sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments" title="sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderTrainingArguments</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/trainer.html#sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer" title="sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer"><code class="xref py py-class docutils literal notranslate"><span class="pre">CrossEncoderTrainer</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/trainer.html#sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer.train" title="sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer.train"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoderTrainer.train()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.save_pretrained" title="sentence_transformers.cross_encoder.CrossEncoder.save_pretrained"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoder.save_pretrained()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/cross_encoder/cross_encoder.html#sentence_transformers.cross_encoder.CrossEncoder.push_to_hub" title="sentence_transformers.cross_encoder.CrossEncoder.push_to_hub"><code class="xref py py-meth docutils literal notranslate"><span class="pre">CrossEncoder.push_to_hub()</span></code></a></p></li>
            </ol>
        </div>

    ::

        import logging
        import traceback

        import torch
        from datasets import load_dataset

        from sentence_transformers import SentenceTransformer
        from sentence_transformers.cross_encoder import (
            CrossEncoder,
            CrossEncoderModelCardData,
            CrossEncoderTrainer,
            CrossEncoderTrainingArguments,
        )
        from sentence_transformers.cross_encoder.evaluation import (
            CrossEncoderNanoBEIREvaluator,
            CrossEncoderRerankingEvaluator,
        )
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
        from sentence_transformers.evaluation import SequentialEvaluator
        from sentence_transformers.util import mine_hard_negatives

        # Set the log level to INFO to get more information
        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


        def main():
            model_name = "answerdotai/ModernBERT-base"

            train_batch_size = 64
            num_epochs = 1
            num_hard_negatives = 5  # How many hard negatives should be mined for each question-answer pair

            # 1a. Load a model to finetune with 1b. (Optional) model card data
            model = CrossEncoder(
                model_name,
                model_card_data=CrossEncoderModelCardData(
                    language="en",
                    license="apache-2.0",
                    model_name="ModernBERT-base trained on GooAQ",
                ),
            )
            print("Model max length:", model.max_length)
            print("Model num labels:", model.num_labels)

            # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
            logging.info("Read the gooaq training dataset")
            full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(100_000))
            dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]
            logging.info(train_dataset)
            logging.info(eval_dataset)

            # 2b. Modify our training dataset to include hard negatives using a very efficient embedding model
            embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
            hard_train_dataset = mine_hard_negatives(
                train_dataset,
                embedding_model,
                num_negatives=num_hard_negatives,  # How many negatives per question-answer pair
                margin=0,  # Similarity between query and negative samples should be x lower than query-positive similarity
                range_min=0,  # Skip the x most similar samples
                range_max=100,  # Consider only the x most similar samples
                sampling_strategy="top",  # Sample the top negatives from the range
                batch_size=4096,  # Use a batch size of 4096 for the embedding model
                output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
                use_faiss=True,
            )
            logging.info(hard_train_dataset)

            # 2c. (Optionally) Save the hard training dataset to disk
            # hard_train_dataset.save_to_disk("gooaq-hard-train")
            # Load again with:
            # hard_train_dataset = load_from_disk("gooaq-hard-train")

            # 3. Define our training loss.
            # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`
            loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

            # 4a. Define evaluators. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
            nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
                dataset_names=["msmarco", "nfcorpus", "nq"],
                batch_size=train_batch_size,
            )

            # 4b. Define a reranking evaluator by mining hard negatives given query-answer pairs
            # We include the positive answer in the list of negatives, so the evaluator can use the performance of the
            # embedding model as a baseline.
            hard_eval_dataset = mine_hard_negatives(
                eval_dataset,
                embedding_model,
                corpus=full_dataset["answer"],  # Use the full dataset as the corpus
                num_negatives=30,  # How many documents to rerank
                batch_size=4096,
                include_positives=True,
                output_format="n-tuple",
                use_faiss=True,
            )
            logging.info(hard_eval_dataset)
            reranking_evaluator = CrossEncoderRerankingEvaluator(
                samples=[
                    {
                        "query": sample["question"],
                        "positive": [sample["answer"]],
                        "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
                    }
                    for sample in hard_eval_dataset
                ],
                batch_size=train_batch_size,
                name="gooaq-dev",
                # Realistic setting: only rerank the positives that the retriever found
                # Set to True to rerank *all* positives
                always_rerank_positives=False,
            )

            # 4c. Combine the evaluators & run the base model on them
            evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
            evaluator(model)

            # 5. Define the training arguments
            short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
            run_name = f"reranker-{short_model_name}-gooaq-bce"
            args = CrossEncoderTrainingArguments(
                # Required parameter:
                output_dir=f"models/{run_name}",
                # Optional training parameters:
                num_train_epochs=num_epochs,
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=train_batch_size,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
                bf16=True,  # Set to True if you have a GPU that supports BF16
                dataloader_num_workers=4,
                load_best_model_at_end=True,
                metric_for_best_model="eval_gooaq-dev_ndcg@10",
                # Optional tracking/debugging parameters:
                eval_strategy="steps",
                eval_steps=1000,
                save_strategy="steps",
                save_steps=1000,
                save_total_limit=2,
                logging_steps=200,
                logging_first_step=True,
                run_name=run_name,  # Will be used in W&B if `wandb` is installed
                seed=12,
            )

            # 6. Create the trainer & start training
            trainer = CrossEncoderTrainer(
                model=model,
                args=args,
                train_dataset=hard_train_dataset,
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


        if __name__ == "__main__":
            main()


```

### Callbacks

```{eval-rst}
This CrossEncoder trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

- :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if ``wandb`` is installed
- :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if ``tensorboard`` is accessible.
- :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if ``codecarbon`` is installed.

    - Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
documentation for more information on the integrated callbacks and how to write your own callbacks.
```

## Multi-Dataset Training

```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

- Use a dictionary of :class:`~datasets.Dataset` instances (or a :class:`~datasets.DatasetDict`) as the ``train_dataset`` (and optionally also ``eval_dataset``).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers` enum, which can be passed to the :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments` via ``multi_dataset_batch_sampler``. Valid options are:

- ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted. With this strategy, it’s likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- ``MultiDatasetBatchSamplers.PROPORTIONAL`` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.
```

## Training Tips

```{eval-rst}
Cross Encoder models have their own unique quirks, so here's some tips to help you out:

#. :class:`~sentence_transformers.cross_encoder.CrossEncoder` models overfit rather quickly, so it's recommended to use an evaluator like :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator` or :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator` together with the ``load_best_model_at_end`` and ``metric_for_best_model`` training arguments to load the model with the best evaluation performance after training.
#. :class:`~sentence_transformers.cross_encoder.CrossEncoder` are particularly receptive to strong hard negatives (:func:`~sentence_transformers.util.mine_hard_negatives`). They teach the model to be very strict, useful e.g. when distinguishing between passages that answer a question or passages that relate to a question. 

    a. Note that if you only use hard negatives, `your model may unexpectedly perform worse for easier tasks <https://huggingface.co/papers/2411.11767>`_. This can mean that reranking the top 200 results from a first-stage retrieval system (e.g. with a :class:`~sentence_transformers.SentenceTransformer` model) can actually give worse top-10 results than reranking the top 100. Training using random negatives alongside hard negatives can mitigate this.
#. Don't underestimate :class:`~sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss`, it remains a very strong option despite being simpler than learning-to-rank (:class:`~sentence_transformers.cross_encoder.losses.LambdaLoss`, :class:`~sentence_transformers.cross_encoder.losses.ListNetLoss`) or in-batch negatives (:class:`~sentence_transformers.cross_encoder.losses.CachedMultipleNegativesRankingLoss`, :class:`~sentence_transformers.cross_encoder.losses.MultipleNegativesRankingLoss`) losses, and its data is easy to prepare, especially using :func:`~sentence_transformers.util.mine_hard_negatives`.
```

## Deprecated Training

```{eval-rst}
Prior to the Sentence Transformers v4.0 release, models would be trained with the :meth:`CrossEncoder.fit() <sentence_transformers.cross_encoder.CrossEncoder.fit>` method and a :class:`~torch.utils.data.DataLoader` of :class:`~sentence_transformers.readers.InputExample`, which looked something like this::

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    # Define the model. Either from scratch of by loading a pre-trained model
    model = CrossEncoder("distilbert/distilbert-base-uncased")

    # Define your train examples. You need more than just two examples...
    train_examples = [
        InputExample(texts=["What are pandas?", "The giant panda ..."], label=1),
        InputExample(texts=["What's a panda?", "Mount Vesuvius is a ..."], label=0),
    ]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Tune the model
    model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)

Since the v4.0 release, using :meth:`CrossEncoder.fit() <sentence_transformers.cross_encoder.CrossEncoder.fit>` is still possible, but it will initialize a :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` behind the scenes. It is recommended to use the Trainer directly, as you will have more control via the :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments`, but existing training scripts relying on :meth:`CrossEncoder.fit() <sentence_transformers.cross_encoder.CrossEncoder.fit>` should still work.

In case there are issues with the updated :meth:`CrossEncoder.fit() <sentence_transformers.cross_encoder.CrossEncoder.fit>`, you can also get exactly the old behaviour by calling :meth:`CrossEncoder.old_fit() <sentence_transformers.cross_encoder.CrossEncoder.old_fit>` instead, but this method is planned to be deprecated fully in the future.

```

## Comparisons with SentenceTransformer Training

```{eval-rst}
Training :class:`~sentence_transformers.cross_encoder.CrossEncoder` models is very similar as training :class:`~sentence_transformers.SentenceTransformer` models, with some key differences:

- Instead of just ``score`` and ``label``, columns named ``scores`` and ``labels`` will also be considered "label columns" for :class:`~sentence_transformers.cross_encoder.CrossEncoder` training. As you can see in the `Loss Overview <loss_overview.html>`_ documentation, some losses require specific labels/scores in a column with one of these names.
- In :class:`~sentence_transformers.SentenceTransformer` training, you cannot use lists of inputs (e.g. texts) in a column of the training/evaluation dataset(s). For :class:`~sentence_transformers.cross_encoder.CrossEncoder` training, you **can** use (variably sized) lists of texts in a column. This is required for the :class:`~sentence_transformers.cross_encoder.losses.ListNetLoss` class, for example.

See the `Sentence Transformer > Training Overview <../sentence_transformer/training_overview.html>`_ documentation for more details on training :class:`~sentence_transformers.SentenceTransformer` models.

```
