# Training Overview

## Why Finetune?

Finetuning Sentence Transformer models often heavily improves the performance of the model on your use case, because each task requires a different notion of similarity. For example, given news articles:

- "Apple launches the new iPad"
- "NVIDIA is gearing up for the next GPU generation"

Then the following use cases, we may have different notions of similarity:

- a model for **classification** of news articles as Economy, Sports, Technology, Politics, etc., should produce **similar embeddings** for these texts.
- a model for **semantic textual similarity** should produce **dissimilar embeddings** for these texts, as they have different meanings.
- a model for **semantic search** would **not need a notion for similarity** between two documents, as it should only compare queries and documents.

Also see [**Training Examples**](training/examples) for numerous training scripts for common real-world applications that you can adopt.

## Training Components

Training Sentence Transformer models involves between 4 to 6 components:

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

Sentence Transformer models consist of a sequence of `Modules <../package_reference/sentence_transformer/models.html>`_ or `Custom Modules <usage/custom_models.html#advanced-custom-modules>`_, allowing for a lot of flexibility. If you want to further finetune a SentenceTransformer model (e.g. it has a `modules.json file <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/modules.json>`_), then you don't have to worry about which modules are used::

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

But if instead you want to train from another checkpoint, or from scratch, then these are the most common architectures you can use:

.. tab:: Transformers

    Most Sentence Transformer models use the :class:`~sentence_transformers.models.Transformer` and :class:`~sentence_transformers.models.Pooling` modules. The former loads a pretrained transformer model (e.g. `BERT <https://huggingface.co/google-bert/bert-base-uncased>`_, `RoBERTa <https://huggingface.co/FacebookAI/roberta-base>`_, `DistilBERT <https://huggingface.co/distilbert/distilbert-base-uncased>`_, `ModernBERT <https://huggingface.co/answerdotai/ModernBERT-base>`_, etc.) and the latter pools the output of the transformer to produce a single vector representation for each input sentence.

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Transformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.Transformer</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Pooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.Pooling</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import models, SentenceTransformer

        transformer = models.Transformer("google-bert/bert-base-uncased")
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

        model = SentenceTransformer(modules=[transformer, pooling])
    
    This is the default option in Sentence Transformers, so it's easier to use the shortcut:

    ::

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("google-bert/bert-base-uncased")

    .. tip::

        The strongest base models are often "encoder models", i.e. models that are trained to produce a meaningful token embedding for inputs. You can find strong candidates here:

        - `fill-mask models <https://huggingface.co/models?pipeline_tag=fill-mask>`_ - trained for token embeddings
        - `sentence similarity models <https://huggingface.co/models?pipeline_tag=sentence-similarity>`_ - trained for text embeddings
        - `feature-extraction models <https://huggingface.co/models?pipeline_tag=feature-extraction>`_ - trained for text embeddings

        Consider looking for base models that are designed on your language and/or domain of interest. For example, `FacebookAI/xlm-roberta-base <https://huggingface.co/FacebookAI/xlm-roberta-base>`_ will work better than `google-bert/bert-base-uncased <https://huggingface.co/google-bert/bert-base-uncased>`_ for Turkish.

.. tab:: Static

    Static Embedding models (`blogpost <https://huggingface.co/blog/static-embeddings>`_) use the :class:`~sentence_transformers.models.StaticEmbedding` module, and are encoder models that don't use slow transformers or attention mechanisms. For these models, computing embeddings is simply: given the input token, return the pre-computed token embedding. These models are orders of magnitude faster, but cannot capture complex semantics as token embeddings are computed separate from the context.

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/blog/static-embeddings">Static Embedding Models</a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.StaticEmbedding"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.StaticEmbedding</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.StaticEmbedding.from_model2vec"><code class="xref py py-meth docutils literal notranslate"><span class="pre">sentence_transformers.models.StaticEmbedding.from_model2vec</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.StaticEmbedding.from_distillation"><code class="xref py py-meth docutils literal notranslate"><span class="pre">sentence_transformers.models.StaticEmbedding.from_distillation</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import models, SentenceTransformer
        from tokenizers import Tokenizer

        # Load any Tokenizer from Hugging Face
        tokenizer = Tokenizer.from_pretrained("google-bert/bert-base-uncased")
        # The `embedding_dim` is the dimensionality (size) of the token embeddings
        static_embedding = StaticEmbedding(tokenizer, embedding_dim=512)

        model = SentenceTransformer(modules=[static_embedding])

```

## Dataset

```{eval-rst}
The :class:`SentenceTransformerTrainer` trains and evaluates using :class:`datasets.Dataset` (one dataset) or :class:`datasets.DatasetDict` instances (multiple datasets, see also `Multi-dataset training <#multi-dataset-training>`_). 

.. tab:: Data on 🤗 Hugging Face Hub

    If you want to load data from the `Hugging Face Datasets <https://huggingface.co/datasets>`_, then you should use :func:`datasets.load_dataset`:

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/loading#hugging-face-hub">Datasets, Loading from the Hugging Face Hub</a></li>
                <li><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset"><code class="xref py py-func docutils literal notranslate"><span class="pre">datasets.load_dataset()</span></code></a></li>
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
It is important that your dataset format matches your loss function (or that you choose a loss function that matches your dataset format). Verifying whether a dataset format works with a loss function involves two steps:

1. If your loss function requires a *Label* according to the `Loss Overview <loss_overview.html>`_ table, then your dataset must have a **column named "label" or "score"**. This column is automatically taken as the label.
2. All columns not named "label" or "score" are considered *Inputs* according to the `Loss Overview <loss_overview.html>`_ table. The number of remaining columns must match the number of valid inputs for your chosen loss. The names of these columns are **irrelevant**, only the **order matters**. 

For example, given a dataset with columns ``["text1", "text2", "label"]`` where the "label" column has float similarity score between 0 and 1, we can use it with :class:`~sentence_transformers.losses.CoSENTLoss`, :class:`~sentence_transformers.losses.AnglELoss`, and :class:`~sentence_transformers.losses.CosineSimilarityLoss` because it:

1. has a "label" column as is required for these loss functions.
2. has 2 non-label columns, exactly the amount required by these loss functions.

Be sure to re-order your dataset columns with :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` if your columns are not ordered correctly. For example, if your dataset has ``["good_answer", "bad_answer", "question"]`` as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the ``good_answer`` column will be taken as the anchor, ``bad_answer`` as the positive, and ``question`` as the negative.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with :meth:`Dataset.remove_columns <datasets.Dataset.remove_columns>` as they will be used as inputs otherwise. You can also use :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` to keep only the desired columns.
```

## Loss Function

Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See [Dataset Format](#dataset-format) to learn what datasets are valid for which loss functions. Additionally, the [Loss Overview](loss_overview) will be your best friend to learn about the options.

```{eval-rst}
Most loss functions can be initialized with just the :class:`~sentence_transformers.SentenceTransformer` that you're training, alongside some optional parameters, e.g.:

.. sidebar:: Documentation

    - :class:`sentence_transformers.losses.CoSENTLoss`
    - `Losses API Reference <../package_reference/sentence_transformer/losses.html>`_
    - `Loss Overview <loss_overview.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import CoSENTLoss

    # Load a model to train/finetune
    model = SentenceTransformer("xlm-roberta-base")

    # Initialize the CoSENTLoss
    # This loss requires pairs of text and a float similarity score as a label
    loss = CoSENTLoss(model)

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
    """
    Dataset({
        features: ['sentence1', 'sentence2', 'label'],
        num_rows: 942069
    })
    """
```

## Training Arguments

```{eval-rst}
The :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments` class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.
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
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>batch_sampler</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>multi_dataset_batch_sampler</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>prompts</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>router_mapping</code></a>
        <a href="../package_reference/sentence_transformer/training_args.html#sentence_transformers.training_args.SentenceTransformerTrainingArguments"><code>learning_rate_mapping</code></a>
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
Here is an example of how :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments` can be initialized:
```

```python
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
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
    run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
)
```

## Evaluator

```{eval-rst}
You can provide the :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` with an ``eval_dataset`` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an ``eval_dataset`` and an evaluator, one or the other, or neither. They evaluate based on the ``eval_strategy`` and ``eval_steps`` `Training Arguments <#training-arguments>`_.

Here are the implemented Evaluators that come with Sentence Transformers:

========================================================================  ===========================================================================================================================
Evaluator                                                                 Required Data
========================================================================  ===========================================================================================================================
:class:`~sentence_transformers.evaluation.BinaryClassificationEvaluator`  Pairs with class labels.
:class:`~sentence_transformers.evaluation.EmbeddingSimilarityEvaluator`   Pairs with similarity scores.
:class:`~sentence_transformers.evaluation.InformationRetrievalEvaluator`  Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]).
:class:`~sentence_transformers.evaluation.NanoBEIREvaluator`              No data required.
:class:`~sentence_transformers.evaluation.MSEEvaluator`                   Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts.
:class:`~sentence_transformers.evaluation.ParaphraseMiningEvaluator`      Mapping of IDs to sentences & pairs with IDs of duplicate sentences.
:class:`~sentence_transformers.evaluation.RerankingEvaluator`             List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries.
:class:`~sentence_transformers.evaluation.TranslationEvaluator`           Pairs of sentences in two separate languages.
:class:`~sentence_transformers.evaluation.TripletEvaluator`               (anchor, positive, negative) pairs.
========================================================================  ===========================================================================================================================

Additionally, :class:`~sentence_transformers.evaluation.SequentialEvaluator` should be used to combine multiple evaluators into one Evaluator that can be passed to the :class:`~sentence_transformers.trainer.SentenceTransformerTrainer`.

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

.. tab:: EmbeddingSimilarityEvaluator with STSb

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/stsb">sentence-transformers/stsb</a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator" title="sentence_transformers.evaluation.EmbeddingSimilarityEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.evaluation.EmbeddingSimilarityEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

        # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

        # Initialize the evaluator
        dev_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: TripletEvaluator with AllNLI

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator" title="sentence_transformers.evaluation.TripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.evaluation.TripletEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import TripletEvaluator, SimilarityFunction

        # Load triplets from the AllNLI dataset (https://huggingface.co/datasets/sentence-transformers/all-nli)
        max_samples = 1000
        eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=f"dev[:{max_samples}]")

        # Initialize the evaluator
        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_distance_function=SimilarityFunction.COSINE,
            name="all-nli-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: NanoBEIREvaluator

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.NanoBEIREvaluator" title="sentence_transformers.evaluation.NanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.evaluation.NanoBEIREvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers.evaluation import NanoBEIREvaluator

        # Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
        # directly from Hugging Face, so there's no mandatory arguments
        dev_evaluator = NanoBEIREvaluator()
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tip::

    When evaluating frequently during training with a small ``eval_steps``, consider using a tiny ``eval_dataset`` to minimize evaluation overhead. If you're concerned about the evaluation set size, a 90-1-9 train-eval-test split can provide a balance, reserving a reasonably sized test set for final evaluations. After training, you can assess your model's performance using ``trainer.evaluate(test_dataset)`` for test loss or initialize a testing evaluator with ``test_evaluator(model)`` for detailed test metrics.

    If you evaluate after training, but before saving the model, your automatically generated model card will still include the test results.

.. warning::

    When using `Distributed Training <training/distributed.html>`_, the evaluator only runs on the first device, unlike the training and evaluation datasets, which are shared across all devices. 
```

## Trainer

```{eval-rst}
The :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let's have a look at a script where all of these components come together:

.. sidebar:: Documentation

    #. :class:`~sentence_transformers.SentenceTransformer`
    #. :class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`
    #. :func:`~datasets.load_dataset`
    #. :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
    #. :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`
    #. :class:`~sentence_transformers.evaluation.TripletEvaluator`
    #. :class:`~sentence_transformers.trainer.SentenceTransformerTrainer`
    #. :class:`SentenceTransformer.save_pretrained <sentence_transformers.SentenceTransformer.save_pretrained>`
    #. :class:`SentenceTransformer.push_to_hub <sentence_transformers.SentenceTransformer.push_to_hub>`

    - `Training Examples <training/examples.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        SentenceTransformerModelCardData,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import TripletEvaluator

    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "microsoft/mpnet-base",
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="MPNet base trained on AllNLI triplets",
        )
    )

    # 3. Load a dataset to finetune on
    dataset = load_dataset("sentence-transformers/all-nli", "triplet")
    train_dataset = dataset["train"].select(range(100_000))
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/mpnet-base-all-nli-triplet",
        # Optional training parameters:
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name="all-nli-dev",
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # (Optional) Evaluate the trained model on the test set
    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained("models/mpnet-base-all-nli-triplet/final")
    
    # 9. (Optional) Push it to the Hugging Face Hub
    model.push_to_hub("mpnet-base-all-nli-triplet")

```

### Callbacks

```{eval-rst}
This Sentence Transformers trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

- :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if ``wandb`` is installed
- :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if ``tensorboard`` is accessible.
- :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if ``codecarbon`` is installed.

    - Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
documentation for more information on the integrated callbacks and how to write your own callbacks.
```

## Multi-Dataset Training

```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`sentence_transformers.trainer.SentenceTransformerTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

- Use a dictionary of :class:`~datasets.Dataset` instances (or a :class:`~datasets.DatasetDict`) as the ``train_dataset``  (and optionally also ``eval_dataset``).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers` enum, which can be passed to the :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments` via ``multi_dataset_batch_sampler``. Valid options are:

- ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted. With this strategy, it’s likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- ``MultiDatasetBatchSamplers.PROPORTIONAL`` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.

This multi-task training has been shown to be very effective, e.g. `Huang et al. <https://huggingface.co/papers/2405.06932>`_ employed :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`, :class:`~sentence_transformers.losses.CoSENTLoss`, and a variation on :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` without in-batch negatives and only hard negatives to reach state-of-the-art performance on Chinese. They even applied :class:`~sentence_transformers.losses.MatryoshkaLoss` to allow the model to produce `Matryoshka Embeddings <../../examples/sentence_transformer/training/matryoshka/README.html>`_.

Training on multiple datasets looks like this:

.. sidebar:: Documentation

    - :func:`datasets.load_dataset`
    - :class:`~sentence_transformers.SentenceTransformer`
    - :class:`~sentence_transformers.trainer.SentenceTransformerTrainer`
    - :class:`~sentence_transformers.losses.CoSENTLoss`
    - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
    - :class:`~sentence_transformers.losses.SoftmaxLoss`
    - `sentence-transformers/all-nli <https://huggingface.co/datasets/sentence-transformers/all-nli>`_
    - `sentence-transformers/stsb <https://huggingface.co/datasets/sentence-transformers/stsb>`_
    - `sentence-transformers/quora-duplicates <https://huggingface.co/datasets/sentence-transformers/quora-duplicates>`_
    - `sentence-transformers/natural-questions <https://huggingface.co/datasets/sentence-transformers/natural-questions>`_

    **Training Examples:**

    - `Quora Duplicate Questions > Multi-task learning <https://github.com/huggingface/sentence-transformers/blob/master/examples/sentence_transformer/training/quora_duplicate_questions/training_multi-task-learning.py>`_
    - `AllNLI + STSb > Multi-task learning <https://github.com/huggingface/sentence-transformers/blob/master/examples/sentence_transformer/training/other/training_multi-task.py>`_

::

    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss, SoftmaxLoss

    # 1. Load a model to finetune
    model = SentenceTransformer("bert-base-uncased")

    # 2. Load several Datasets to train with
    # (anchor, positive)
    all_nli_pair_train = load_dataset("sentence-transformers/all-nli", "pair", split="train[:10000]")
    # (premise, hypothesis) + label
    all_nli_pair_class_train = load_dataset("sentence-transformers/all-nli", "pair-class", split="train[:10000]")
    # (sentence1, sentence2) + score
    all_nli_pair_score_train = load_dataset("sentence-transformers/all-nli", "pair-score", split="train[:10000]")
    # (anchor, positive, negative)
    all_nli_triplet_train = load_dataset("sentence-transformers/all-nli", "triplet", split="train[:10000]")
    # (sentence1, sentence2) + score
    stsb_pair_score_train = load_dataset("sentence-transformers/stsb", split="train[:10000]")
    # (anchor, positive)
    quora_pair_train = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[:10000]")
    # (query, answer)
    natural_questions_train = load_dataset("sentence-transformers/natural-questions", split="train[:10000]")

    # We can combine all datasets into a dictionary with dataset names to datasets
    train_dataset = {
        "all-nli-pair": all_nli_pair_train,
        "all-nli-pair-class": all_nli_pair_class_train,
        "all-nli-pair-score": all_nli_pair_score_train,
        "all-nli-triplet": all_nli_triplet_train,
        "stsb": stsb_pair_score_train,
        "quora": quora_pair_train,
        "natural-questions": natural_questions_train,
    }

    # 3. Load several Datasets to evaluate with
    # (anchor, positive, negative)
    all_nli_triplet_dev = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")
    # (sentence1, sentence2, score)
    stsb_pair_score_dev = load_dataset("sentence-transformers/stsb", split="validation")
    # (anchor, positive)
    quora_pair_dev = load_dataset("sentence-transformers/quora-duplicates", "pair", split="train[10000:11000]")
    # (query, answer)
    natural_questions_dev = load_dataset("sentence-transformers/natural-questions", split="train[10000:11000]")

    # We can use a dictionary for the evaluation dataset too, but we don't have to. We could also just use
    # no evaluation dataset, or one dataset.
    eval_dataset = {
        "all-nli-triplet": all_nli_triplet_dev,
        "stsb": stsb_pair_score_dev,
        "quora": quora_pair_dev,
        "natural-questions": natural_questions_dev,
    }

    # 4. Load several loss functions to train with
    # (anchor, positive), (anchor, positive, negative)
    mnrl_loss = MultipleNegativesRankingLoss(model)
    # (sentence_A, sentence_B) + class
    softmax_loss = SoftmaxLoss(model, model.get_sentence_embedding_dimension(), 3)
    # (sentence_A, sentence_B) + score
    cosent_loss = CoSENTLoss(model)

    # Create a mapping with dataset names to loss functions, so the trainer knows which loss to apply where.
    # Note that you can also just use one loss if all of your training/evaluation datasets use the same loss
    losses = {
        "all-nli-pair": mnrl_loss,
        "all-nli-pair-class": softmax_loss,
        "all-nli-pair-score": cosent_loss,
        "all-nli-triplet": mnrl_loss,
        "stsb": cosent_loss,
        "quora": mnrl_loss,
        "natural-questions": mnrl_loss,
    }

    # 5. Define a simple trainer, although it's recommended to use one with args & evaluators
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=losses,
    )
    trainer.train()

    # 6. save the trained model and optionally push it to the Hugging Face Hub
    model.save_pretrained("bert-base-all-nli-stsb-quora-nq")
    model.push_to_hub("bert-base-all-nli-stsb-quora-nq")
```

## Deprecated Training

```{eval-rst}
Prior to the Sentence Transformers v3.0 release, models would be trained with the :meth:`SentenceTransformer.fit() <sentence_transformers.SentenceTransformer.fit>` method and a :class:`~torch.utils.data.DataLoader` of :class:`~sentence_transformers.readers.InputExample`, which looked something like this::

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer("distilbert/distilbert-base-uncased")

    # Define your train examples. You need more than just two examples...
    train_examples = [
        InputExample(texts=["My first sentence", "My second sentence"], label=0.8),
        InputExample(texts=["Another pair", "Unrelated sentence"], label=0.3),
    ]

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

Since the v3.0 release, using :meth:`SentenceTransformer.fit() <sentence_transformers.SentenceTransformer.fit>` is still possible, but it will initialize a :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` behind the scenes. It is recommended to use the Trainer directly, as you will have more control via the :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`, but existing training scripts relying on :meth:`SentenceTransformer.fit() <sentence_transformers.SentenceTransformer.fit>` should still work.

In case there are issues with the updated :meth:`SentenceTransformer.fit() <sentence_transformers.SentenceTransformer.fit>`, you can also get exactly the old behaviour by calling :meth:`SentenceTransformer.old_fit() <sentence_transformers.SentenceTransformer.old_fit>` instead, but this method is planned to be deprecated fully in the future.

```

## Best Base Embedding Models

The quality of your text embedding model depends on which transformer model you choose. Sadly we cannot infer from a better performance on e.g. the GLUE or SuperGLUE benchmark that this model will also yield better representations.

To test the suitability of transformer models, I use the [training_nli_v2.py](https://github.com/huggingface/sentence-transformers/blob/master/examples/sentence_transformer/training/nli/training_nli_v2.py) script and train on 560k (anchor, positive, negative)-triplets for 1 epoch with batch size 64. I then evaluate on 14 diverse text similarity tasks (clustering, semantic search, duplicate detection etc.) from various domains.

In the following table you find the performance for different models and their performance on this benchmark:

| Model                                                                                                                             | Performance (14 sentence similarity tasks) |
|:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|
| [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base)                                                               | 60.99                                      |
| [nghuyong/ernie-2.0-en](https://huggingface.co/nghuyong/ernie-2.0-en)                                                             | 60.73                                      |
| [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base)                                                           | 60.21                                      |
| [roberta-base](https://huggingface.co/roberta-base)                                                                               | 59.63                                      |
| [t5-base](https://huggingface.co/t5-base)                                                                                         | 59.21                                      |
| [bert-base-uncased](https://huggingface.co/bert-base-uncased)                                                                     | 59.17                                      |
| [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)                                                         | 59.03                                      |
| [nreimers/TinyBERT_L-6_H-768_v2](https://huggingface.co/nreimers/TinyBERT_L-6_H-768_v2)                                           | 58.27                                      |
| [google/t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base)                                                                 | 57.63                                      |
| [nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large](https://huggingface.co/nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Large) | 57.31                                      |
| [albert-base-v2](https://huggingface.co/albert-base-v2)                                                                           | 57.14                                      |
| [microsoft/MiniLM-L12-H384-uncased](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)                                     | 56.79                                      |
| [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base)                                                     | 54.46                                      |

## Comparisons with CrossEncoder Training

```{eval-rst}
Training :class:`~sentence_transformers.SentenceTransformer` models is very similar as training :class:`~sentence_transformers.cross_encoder.CrossEncoder` models, with some key differences:

- Instead of ``score``, ``scores``, ``label`` and ``labels`` columns being considered "label columns", only ``score`` and ``label`` are. As you can see in the `Loss Overview <loss_overview.html>`_ documentation, some losses require specific labels/scores in a column with one of these names.
- For :class:`~sentence_transformers.cross_encoder.CrossEncoder` training, you can use (variably sized) lists of texts in a column. In :class:`~sentence_transformers.SentenceTransformer` training, you **cannot** use lists of inputs (e.g. texts) in a column of the training/evaluation dataset(s). In short: training with a variable number of negatives is not supported.

See the `Cross Encoder > Training Overview <../cross_encoder/training_overview.html>`_ documentation for more details on training :class:`~sentence_transformers.cross_encoder.CrossEncoder` models.

```
