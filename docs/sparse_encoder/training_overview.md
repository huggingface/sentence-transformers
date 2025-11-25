# Training Overview

## Why Finetune?
Finetuning Sparse Encoder models often heavily improves the performance of the model on your use case, because each task requires a different notion of similarity. For example, given news articles: 
- "Apple launches the new iPad"
- "NVIDIA is gearing up for the next GPU generation"

Then the following use cases, we may have different notions of similarity:
- a model for **classification** of news articles as Economy, Sports, Technology, Politics, etc., should produce **similar embeddings** for these texts.
- a model for **semantic textual similarity** should produce **dissimilar embeddings** for these texts, as they have different meanings.
- a model for **semantic search** would **not need a notion for similarity** between two documents, as it should only compare queries and documents.


Also see [**Training Examples**](training/examples) for numerous training scripts for common real-world applications that you can adopt.


## Training Components
Training Sparse Encoder models involves between 4 to 6 components:

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

Sparse Encoder models consist of a sequence of `Modules <../package_reference/sentence_transformer/models.html>`_,  `Sparse Encoder specific Modules <../package_reference/sparse_encoder/models.html>`_ or `Custom Modules <../sentence_transformer/usage/custom_models.html#advanced-custom-modules>`_, allowing for a lot of flexibility. If you want to further finetune a SparseEncoder model (e.g. it has a `modules.json file <https://huggingface.co/naver/splade-cocondenser-ensembledistil/tree/main/modules.json>`_), then you don't have to worry about which modules are used::

    from sentence_transformers import SparseEncoder

    model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

But if instead you want to train from another checkpoint, or from scratch, then these are the most common architectures you can use:

.. tab:: Splade

    Splade models use the :class:`~sentence_transformers.sparse_encoder.models.MLMTransformer` followed by a :class:`~sentence_transformers.sparse_encoder.models.SpladePooling` modules. The former loads a pretrained `Masked Language Modeling transformer model <https://huggingface.co/models?pipeline_tag=fill-mask>`_ (e.g. `BERT <https://huggingface.co/google-bert/bert-base-uncased>`_, `RoBERTa <https://huggingface.co/FacebookAI/roberta-base>`_, `DistilBERT <https://huggingface.co/distilbert/distilbert-base-uncased>`_, `ModernBERT <https://huggingface.co/answerdotai/ModernBERT-base>`_, etc.) and the latter pools the output of the MLMHead to produce a single sparse embedding of the size of the vocabulary.
    
    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.MLMTransformer</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.SpladePooling</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import models, SparseEncoder
        from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling

        # Initialize MLM Transformer (use a fill-mask model)
        mlm_transformer = MLMTransformer("google-bert/bert-base-uncased")
        
        # Initialize SpladePooling module
        splade_pooling = SpladePooling(pooling_strategy="max")

        # Create the Splade model
        model = SparseEncoder(modules=[mlm_transformer, splade_pooling])
    
    This architecture is the default if you provide a fill-mask model architecture to SparseEncoder, so it's easier to use the shortcut:

    ::

        from sentence_transformers import SparseEncoder

        model = SparseEncoder("google-bert/bert-base-uncased")
        # SparseEncoder(
        #   (0): MLMTransformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertForMaskedLM'})
        #   (1): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': None})
        # )

.. tab:: Inference-free Splade

    Inference-free Splade uses a :class:`~sentence_transformers.models.Router` module with different modules for queries and documents. Usually for this type of architecture, the documents part is a traditional Splade architecture (a :class:`~sentence_transformers.sparse_encoder.models.MLMTransformer` followed by a :class:`~sentence_transformers.sparse_encoder.models.SpladePooling` module) and the query part is an :class:`~sentence_transformers.sparse_encoder.models.SparseStaticEmbedding` module, which just returns a pre-computed score for every token in the query.

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Router"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.Router</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SparseStaticEmbedding"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.SparseStaticEmbedding</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.MLMTransformer</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.SpladePooling</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import SparseEncoder
        from sentence_transformers.models import Router
        from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling

        # Initialize MLM Transformer for document encoding
        doc_encoder = MLMTransformer("google-bert/bert-base-uncased")

        # Create a router model with different paths for queries and documents
        router = Router.for_query_document(
            query_modules=[SparseStaticEmbedding(tokenizer=doc_encoder.tokenizer, frozen=False)],
            # Document path: full MLM transformer + pooling
            document_modules=[doc_encoder, SpladePooling("max")],
        )

        # Create the inference-free model
        model = SparseEncoder(modules=[router], similarity_fn_name="dot")
        # SparseEncoder(
        #   (0): Router(
        #     (query_0_SparseStaticEmbedding): SparseStaticEmbedding({'frozen': False}, dim:30522, tokenizer: BertTokenizerFast)
        #     (document_0_MLMTransformer): MLMTransformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertForMaskedLM'})
        #     (document_1_SpladePooling): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': None})
        #   )
        # )
    
    This architecture allows for fast query-time processing using the lightweight SparseStaticEmbedding approach, that can be trained and seen as a linear weights, while documents are processed with the full MLM transformer and SpladePooling.

    .. tip::

        Inference-free Splade is particularly useful for search applications where query latency is critical, as it shifts the computational complexity to the document indexing phase which can be done offline.
    
    .. note::

        When training models with the :class:`~sentence_transformers.models.Router` module, you must use the ``router_mapping`` argument in the :class:`~sentence_transformers.sparse_encoder.SparseEncoderTrainingArguments` to map the training dataset columns to the correct route ("query" or "document"). For example, if your dataset(s) have ``["question", "answer"]`` columns, then you can use the following mapping::

            args = SparseEncoderTrainingArguments(
                ...,
                router_mapping={
                    "question": "query",
                    "answer": "document",
                }
            )
        
        Additionally, it is recommended to use a much higher learning rate for the SparseStaticEmbedding module than for the rest of the model. For this, you should use the ``learning_rate_mapping`` argument in the :class:`~sentence_transformers.sparse_encoder.SparseEncoderTrainingArguments` to map parameter patterns to their learning rates. For example, if you want to use a learning rate of ``1e-3`` for the SparseStaticEmbedding module and ``2e-5`` for the rest of the model, you can do this::

            args = SparseEncoderTrainingArguments(
                ...,
                learning_rate=2e-5,
                learning_rate_mapping={
                    r"SparseStaticEmbedding\.*": 1e-3,
                }
            )
            
.. tab:: Contrastive Sparse Representation (CSR) 

    .. 
        Contrastive Sparse Representation (CSR) models usually use a sequence of :class:`~sentence_transformers.models.Transformer`, :class:`~sentence_transformers.models.Pooling` and :class:`~sentence_transformers.sparse_encoder.models.SparseAutoEncoder` modules to create sparse representations on top of an already trained dense Sentence Transformer model.
    Contrastive Sparse Representation (CSR) models apply a :class:`~sentence_transformers.sparse_encoder.models.SparseAutoEncoder` module on top of a dense Sentence Transformer model, which usually consist of a :class:`~sentence_transformers.models.Transformer` followed by a :class:`~sentence_transformers.models.Pooling` module. You can initialize one from scratch like so:
    
    .. 
        usually use a sequence of :class:`~sentence_transformers.models.Transformer`, :class:`~sentence_transformers.models.Pooling` and :class:`~sentence_transformers.sparse_encoder.models.SparseAutoEncoder` modules to create sparse representations on top of an already trained dense Sentence Transformer model.

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Transformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.Transformer</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Pooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.models.Pooling</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SparseAutoEncoder"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.models.SparseAutoEncoder</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers import models, SparseEncoder
        from sentence_transformers.sparse_encoder.models import SparseAutoEncoder

        # Initialize transformer (can be any dense encoder model)
        transformer = models.Transformer("google-bert/bert-base-uncased")
        
        # Initialize pooling
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        
        # Initialize SparseAutoEncoder module
        sae = SparseAutoEncoder(
            input_dim=transformer.get_word_embedding_dimension(),
            hidden_dim=4 * transformer.get_word_embedding_dimension(),
            k=256,  # Number of top values to keep
            k_aux=512,  # Number of top values for auxiliary loss
        )
        # Create the CSR model
        model = SparseEncoder(modules=[transformer, pooling, sae])
    
    Or if your base model is 1) a dense Sentence Transformer model or 2) a non-MLM Transformer model (those are loaded as Splade models by default), then this shortcut will automatically initialize the CSR model for you:

    ::

        from sentence_transformers import SparseEncoder

        model = SparseEncoder("mixedbread-ai/mxbai-embed-large-v1")
        # SparseEncoder(
        #   (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
        #   (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
        #   (2): SparseAutoEncoder({'input_dim': 1024, 'hidden_dim': 4096, 'k': 256, 'k_aux': 512, 'normalize': False, 'dead_threshold': 30})
        # )

    .. warning::

        Unlike (Inference-free) Splade models, sparse embeddings by CSR models don't have the same size as the vocabulary of the base model. This means you can't directly interpret which words are activated in your embedding like you can with Splade models, where each dimension corresponds to a specific token in the vocabulary.

        Beyond that, CSR models are most effective on dense encoder models that use high-dimensional representations (e.g. 1024-4096 dimensions).

```

## Dataset
```{eval-rst}
The :class:`SparseEncoderTrainer` trains and evaluates using :class:`datasets.Dataset` (one dataset) or :class:`datasets.DatasetDict` instances (multiple datasets, see also `Multi-dataset training <#multi-dataset-training>`_). 

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

        train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
        eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

        print(train_dataset)
        """
        Dataset({
            features: ['anchor', 'positive', 'negative'],
            num_rows: 557850
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

For example, given a dataset with columns ``["text1", "text2", "label"]`` where the "label" column has float similarity score between 0 and 1, we can use it with :class:`~sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss`, :class:`~sentence_transformers.sparse_encoder.losses.SparseAnglELoss`, and :class:`~sentence_transformers.sparse_encoder.losses.SparseCosineSimilarityLoss` because it:

1. has a "label" column as is required for these loss functions.
2. has 2 non-label columns, exactly the amount required by these loss functions.

Be sure to re-order your dataset columns with :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` if your columns are not ordered correctly. For example, if your dataset has ``["good_answer", "bad_answer", "question"]`` as columns, then this dataset can technically be used with a loss that requires (anchor, positive, negative) triplets, but the ``good_answer`` column will be taken as the anchor, ``bad_answer`` as the positive, and ``question`` as the negative.

Additionally, if your dataset has extraneous columns (e.g. sample_id, metadata, source, type), you should remove these with :meth:`Dataset.remove_columns <datasets.Dataset.remove_columns>` as they will be used as inputs otherwise. You can also use :meth:`Dataset.select_columns <datasets.Dataset.select_columns>` to keep only the desired columns.
```

## Loss Function
Loss functions quantify how well a model performs for a given batch of data, allowing an optimizer to update the model weights to produce more favourable (i.e., lower) loss values. This is the core of the training process.

Sadly, there is no single loss function that works best for all use-cases. Instead, which loss function to use greatly depends on your available data and on your target task. See [Dataset Format](#dataset-format) to learn what datasets are valid for which loss functions. Additionally, the [Loss Overview](loss_overview) will be your best friend to learn about the options.

```{eval-rst}
.. warning:: 
    To train a :class:`~sentence_transformers.sparse_encoder.SparseEncoder`, you need either :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` or :class:`~sentence_transformers.sparse_encoder.losses.CSRLoss`, depending on the architecture. These are wrapper losses that add sparsity regularization on top of a main loss function, which must be provided as a parameter. The only loss that can be used independently is :class:`~sentence_transformers.sparse_encoder.losses.SparseMSELoss`, as it performs embedding-level distillation, ensuring sparsity by directly copying the teacher's sparse embedding.
    
Most loss functions can be initialized with just the :class:`~sentence_transformers.sparse_encoder.SparseEncoder` that you're training, alongside some optional parameters, e.g.:

.. sidebar:: Documentation

    - :class:`sentence_transformers.sparse_encoder.losses.SpladeLoss`
    - :class:`sentence_transformers.sparse_encoder.losses.CSRLoss`
    - `Losses API Reference <../package_reference/sparse_encoder/losses.html>`_
    - `Loss Overview <loss_overview.html>`_

::

    from datasets import load_dataset
    from sentence_transformers import SparseEncoder
    from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss

    # Load a model to train/finetune
    model = SparseEncoder("distilbert/distilbert-base-uncased")

    # Initialize the SpladeLoss with a SparseMultipleNegativesRankingLoss
    # This loss requires pairs of related texts or triplets
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=5e-5,  # Weight for query loss
        document_regularizer_weight=3e-5,
    ) 

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    print(train_dataset)
    """
    Dataset({
        features: ['query', 'answer'],
        num_rows: 100231
    })
    """

```

## Training Arguments

```{eval-rst}
The :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` class can be used to specify parameters for influencing training performance as well as defining the tracking/debugging parameters. Although it is optional, it is heavily recommended to experiment with the various useful arguments.
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
        <a href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code>batch_sampler</code></a>
        <a href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code>multi_dataset_batch_sampler</code></a>
        <a href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code>prompts</code></a>
        <a href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code>router_mapping</code></a>
        <a href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code>learning_rate_mapping</code></a>
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
Here is an example of how :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` can be initialized:
```

```python
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir="models/splade-distilbert-base-uncased-nq",
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
    run_name="splade-distilbert-base-uncased-nq",  # Will be used in W&B if `wandb` is installed
)
```

## Evaluator

```{eval-rst}
You can provide the :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` with an ``eval_dataset`` to get the evaluation loss during training, but it may be useful to get more concrete metrics during training, too. For this, you can use evaluators to assess the model's performance with useful metrics before, during, or after training. You can use both an ``eval_dataset`` and an evaluator, one or the other, or neither. They evaluate based on the ``eval_strategy`` and ``eval_steps`` `Training Arguments <#training-arguments>`_.

Here are the implemented Evaluators that come with Sentence Transformers for Sparse Encoder models:

=============================================================================================  ===========================================================================================================================
Evaluator                                                                                      Required Data
=============================================================================================  ===========================================================================================================================
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseBinaryClassificationEvaluator`  Pairs with class labels.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator`   Pairs with similarity scores.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator`  Queries (qid => question), Corpus (cid => document), and relevant documents (qid => set[cid]).
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`              No data required.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseMSEEvaluator`                   Source sentences to embed with a teacher model and target sentences to embed with the student model. Can be the same texts.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseRerankingEvaluator`             List of ``{'query': '...', 'positive': [...], 'negative': [...]}`` dictionaries.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTranslationEvaluator`           Pairs of sentences in two separate languages.
:class:`~sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator`               (anchor, positive, negative) pairs.
=============================================================================================  ===========================================================================================================================

Additionally, :class:`~sentence_transformers.evaluation.SequentialEvaluator` should be used to combine multiple evaluators into one Evaluator that can be passed to the :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer`.

Sometimes you don't have the required evaluation data to prepare one of these evaluators on your own, but you still want to track how well the model performs on some common benchmarks. In that case, you can use these evaluators with data from Hugging Face.

.. tab:: SparseNanoBEIREvaluator

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator</span></code></a></li>
            </ul>
        </div>

    ::

        from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

        # Initialize the evaluator. Unlike most other evaluators, this one loads the relevant datasets
        # directly from Hugging Face, so there's no mandatory arguments
        dev_evaluator = SparseNanoBEIREvaluator()
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: SparseEmbeddingSimilarityEvaluator with STSb

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/stsb">sentence-transformers/stsb</a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseEmbeddingSimilarityEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import SimilarityFunction
        from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

        # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
        eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

        # Initialize the evaluator
        dev_evaluator = SparseEmbeddingSimilarityEvaluator(
            sentences1=eval_dataset["sentence1"],
            sentences2=eval_dataset["sentence2"],
            scores=eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name="sts-dev",
        )
        # You can run evaluation like so:
        # results = dev_evaluator(model)

.. tab:: SparseTripletEvaluator with AllNLI

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ul class="simple">
                <li><a class="reference external" href="https://huggingface.co/datasets/sentence-transformers/all-nli">sentence-transformers/all-nli</a></li>
                <li><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator</span></code></a></li>
                <li><a class="reference internal" href="../package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SimilarityFunction" title="sentence_transformers.SimilarityFunction"><code class="xref py py-class docutils literal notranslate"><span class="pre">sentence_transformers.SimilarityFunction</span></code></a></li>
            </ul>
        </div>

    ::

        from datasets import load_dataset
        from sentence_transformers.evaluation import SimilarityFunction
        from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

        # Load triplets from the AllNLI dataset (https://huggingface.co/datasets/sentence-transformers/all-nli)
        max_samples = 1000
        eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=f"dev[:{max_samples}]")

        # Initialize the evaluator
        dev_evaluator = SparseTripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            main_distance_function=SimilarityFunction.DOT,
            name="all-nli-dev",
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
The :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` is where all previous components come together. We only have to specify the trainer with the model, training arguments (optional), training dataset, evaluation dataset (optional), loss function, evaluator (optional) and we can start training. Let's have a look at a script where all of these components come together:

.. tab:: SPLADE

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ol class="arabic">
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder" title="sentence_transformers.sparse_encoder.SparseEncoder"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder</span></code></a></p>
                <ol class="loweralpha simple">
                    <li><p><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer" title="sentence_transformers.sparse_encoder.models.MLMTransformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">MLMTransformer</span></code></a></p></li>
                    <li><p><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling" title="sentence_transformers.sparse_encoder.models.SpladePooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">SpladePooling</span></code></a></p></li>
                </ol>
                </li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData" title="sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderModelCardData</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">load_dataset()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss" title="sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseMultipleNegativesRankingLoss</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments" title="sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderTrainingArguments</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseTripletEvaluator</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer" title="sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderTrainer</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder.save_pretrained" title="sentence_transformers.sparse_encoder.SparseEncoder.save_pretrained"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder.save_pretrained</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub" title="sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder.push_to_hub</span></code></a></p></li>
            </ol>
            <ul class="simple">
            <li><p><a class="reference external" href="training/examples.html">Training Examples</a></p></li>
            </ul>
        </div>

    ::
    
        import logging

        from datasets import load_dataset

        from sentence_transformers import (
            SparseEncoder,
            SparseEncoderModelCardData,
            SparseEncoderTrainer,
            SparseEncoderTrainingArguments,
        )
        from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
        from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
        from sentence_transformers.training_args import BatchSamplers

        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

        # 1. Load a model to finetune with 2. (Optional) model card data
        model = SparseEncoder(
            "distilbert/distilbert-base-uncased",
            model_card_data=SparseEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name="DistilBERT base trained on Natural-Questions tuples",
            )
        )
    
        # 3. Load a dataset to finetune on
        full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
        dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    
        # 4. Define a loss function
        loss = SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=5e-5,
            document_regularizer_weight=3e-5,
        )
    
        # 5. (Optional) Specify training arguments
        run_name = "splade-distilbert-base-uncased-nq"
        args = SparseEncoderTrainingArguments(
            # Required parameter:
            output_dir=f"models/{run_name}",
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
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            logging_steps=200,
            run_name=run_name,  # Will be used in W&B if `wandb` is installed
        )
    
        # 6. (Optional) Create an evaluator & evaluate the base model
        dev_evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)
    
        # 7. Create a trainer & train
        trainer = SparseEncoderTrainer(
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

.. tab:: Inference-free SPLADE

    .. raw:: html

        <div class="sidebar">
            <p class="sidebar-title">Documentation</p>
            <ol class="arabic">
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder" title="sentence_transformers.sparse_encoder.SparseEncoder"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder</span></code></a></p>
                <ol class="loweralpha simple">
                    <li><p><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SparseStaticEmbedding" title="sentence_transformers.sparse_encoder.models.SparseStaticEmbedding"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseStaticEmbedding</span></code></a></p></li>
                    <li><p><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.MLMTransformer" title="sentence_transformers.sparse_encoder.models.MLMTransformer"><code class="xref py py-class docutils literal notranslate"><span class="pre">MLMTransformer</span></code></a></p></li>
                    <li><p><a class="reference internal" href="../package_reference/sparse_encoder/models.html#sentence_transformers.sparse_encoder.models.SpladePooling" title="sentence_transformers.sparse_encoder.models.SpladePooling"><code class="xref py py-class docutils literal notranslate"><span class="pre">SpladePooling</span></code></a></p></li>
                    <li><p><a class="reference internal" href="../package_reference/sentence_transformer/models.html#sentence_transformers.models.Router" title="sentence_transformers.models.Router"><code class="xref py py-class docutils literal notranslate"><span class="pre">Router</span></code></a></p></li>
                </ol>
                </li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData" title="sentence_transformers.sparse_encoder.model_card.SparseEncoderModelCardData"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderModelCardData</span></code></a></p></li>
                <li><p><a class="reference external" href="https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset" title="(in datasets vmain)"><code class="xref py py-func docutils literal notranslate"><span class="pre">load_dataset()</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/losses.html#sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss" title="sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseMultipleNegativesRankingLoss</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/training_args.html#sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments" title="sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderTrainingArguments</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/evaluation.html#sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator" title="sentence_transformers.sparse_encoder.evaluation.SparseTripletEvaluator"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseTripletEvaluator</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/trainer.html#sentence_transformers.sparse_encoder.SparseEncoderTrainer" title="sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoderTrainer</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder.save_pretrained" title="sentence_transformers.sparse_encoder.SparseEncoder.save_pretrained"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder.save_pretrained</span></code></a></p></li>
                <li><p><a class="reference internal" href="../package_reference/sparse_encoder/SparseEncoder.html#sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub" title="sentence_transformers.sparse_encoder.SparseEncoder.push_to_hub"><code class="xref py py-class docutils literal notranslate"><span class="pre">SparseEncoder.push_to_hub</span></code></a></p></li>
            </ol>
            <ul class="simple">
            <li><p><a class="reference external" href="training/examples.html">Training Examples</a></p></li>
            </ul>
        </div>

    :: 

        import logging

        from datasets import load_dataset

        from sentence_transformers import (
            SparseEncoder,
            SparseEncoderModelCardData,
            SparseEncoderTrainer,
            SparseEncoderTrainingArguments,
        )
        from sentence_transformers.models import Router
        from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
        from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
        from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling
        from sentence_transformers.training_args import BatchSamplers

        logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

        # 1. Load a model to finetune with 2. (Optional) model card data
        mlm_transformer = MLMTransformer("distilbert/distilbert-base-uncased", tokenizer_args={"model_max_length": 512})
        splade_pooling = SpladePooling(
            pooling_strategy="max", word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension()
        )
        router = Router.for_query_document(
            query_modules=[SparseStaticEmbedding(tokenizer=mlm_transformer.tokenizer, frozen=False)],
            document_modules=[mlm_transformer, splade_pooling],
        )

        model = SparseEncoder(
            modules=[router],
            model_card_data=SparseEncoderModelCardData(
                language="en",
                license="apache-2.0",
                model_name="Inference-free SPLADE distilbert-base-uncased trained on Natural-Questions tuples",
            ),
        )

        # 3. Load a dataset to finetune on
        full_dataset = load_dataset("sentence-transformers/natural-questions", split="train").select(range(100_000))
        dataset_dict = full_dataset.train_test_split(test_size=1_000, seed=12)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
        print(train_dataset)
        print(train_dataset[0])

        # 4. Define a loss function
        loss = SpladeLoss(
            model=model,
            loss=SparseMultipleNegativesRankingLoss(model=model),
            query_regularizer_weight=0,
            document_regularizer_weight=3e-4,
        )

        # 5. (Optional) Specify training arguments
        run_name = "inference-free-splade-distilbert-base-uncased-nq"
        args = SparseEncoderTrainingArguments(
            # Required parameter:
            output_dir=f"models/{run_name}",
            # Optional training parameters:
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            learning_rate_mapping={r"SparseStaticEmbedding\.weight": 1e-3},  # Set a higher learning rate for the SparseStaticEmbedding module
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            router_mapping={"query": "query", "answer": "document"},  # Map the column names to the routes
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            logging_steps=200,
            run_name=run_name,  # Will be used in W&B if `wandb` is installed
        )

        # 6. (Optional) Create an evaluator & evaluate the base model
        dev_evaluator = SparseNanoBEIREvaluator(dataset_names=["msmarco", "nfcorpus", "nq"], batch_size=16)

        # 7. Create a trainer & train
        trainer = SparseEncoderTrainer(
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
```

### Callbacks

```{eval-rst}
This Sparse Encoder trainer integrates support for various :class:`transformers.TrainerCallback` subclasses, such as:

- :class:`~sentence_transformers.sparse_encoder.callbacks.splade_callbacks.SpladeRegularizerWeightSchedulerCallback` to schedule
  the lambda parameters of the :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` loss during training.
- :class:`~transformers.integrations.WandbCallback` to automatically log training metrics to W&B if ``wandb`` is installed
- :class:`~transformers.integrations.TensorBoardCallback` to log training metrics to TensorBoard if ``tensorboard`` is accessible.
- :class:`~transformers.integrations.CodeCarbonCallback` to track the carbon emissions of your model during training if ``codecarbon`` is installed.

    - Note: These carbon emissions will be included in your automatically generated model card.

See the Transformers `Callbacks <https://huggingface.co/docs/transformers/main/en/main_classes/callback>`_
documentation for more information on the integrated callbacks and how to write your own callbacks. 

```

## Multi-Dataset Training

```{eval-rst}
The top performing models are trained using many datasets at once. Normally, this is rather tricky, as each dataset has a different format. However, :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` can train with multiple datasets without having to convert each dataset to the same format. It can even apply different loss functions to each of the datasets. The steps to train with multiple datasets are:

- Use a dictionary of :class:`~datasets.Dataset` instances (or a :class:`~datasets.DatasetDict`) as the ``train_dataset`` (and optionally also ``eval_dataset``).
- (Optional) Use a dictionary of loss functions mapping dataset names to losses. Only required if you wish to use different loss function for different datasets.

Each training/evaluation batch will only contain samples from one of the datasets. The order in which batches are samples from the multiple datasets is defined by the :class:`~sentence_transformers.training_args.MultiDatasetBatchSamplers` enum, which can be passed to the :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments` via ``multi_dataset_batch_sampler``. Valid options are:

- ``MultiDatasetBatchSamplers.ROUND_ROBIN``: Round-robin sampling from each dataset until one is exhausted. With this strategy, it's likely that not all samples from each dataset are used, but each dataset is sampled from equally.
- ``MultiDatasetBatchSamplers.PROPORTIONAL`` (default): Sample from each dataset in proportion to its size. With this strategy, all samples from each dataset are used and larger datasets are sampled from more frequently.
```

## Training Tips

```{eval-rst}
Sparse Encoder models have a few quirks that you should be aware of when training them:

1. Sparse Encoder models should not be evaluated solely using the evaluation scores, but also with the sparsity of the embeddings. After all, a low sparsity means that the model embeddings are expensive to store and slow to retrieve. This also means that the parameters that determine sparsity (e.g. ``query_regularizer_weight``, ``document_regularizer_weight`` in :class:`~sentence_transformers.sparse_encoder.losses.SpladeLoss` and ``beta`` and ``gamma`` in the :class:`~sentence_transformers.sparse_encoder.losses.CSRLoss`) should be tuned to achieve a good balance between performance and sparsity. Each `Evaluator <../package_reference/sparse_encoder/evaluation.html>`_ outputs the ``active_dims`` and ``sparsity_ratio`` metrics that can be used to assess the sparsity of the embeddings. 
2. It is not recommended to use an `Evaluator <../package_reference/sparse_encoder/evaluation.html>`_ on an untrained model prior to training, as the sparsity will be very low, and so the memory usage might be unexpectedly high.
3. The stronger Sparse Encoder models are trained almost exclusively with distillation from a stronger teacher model (e.g. a `CrossEncoder model <../cross_encoder/usage/usage.html>`_), instead of training directly from text pairs or triplets. See for example the `SPLADE-v3 paper <https://huggingface.co/papers/2403.06789>`_, which uses :class:`~sentence_transformers.sparse_encoder.losses.SparseDistillKLDivLoss` and :class:`~sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss` for distillation.
4. Whereas the majority of dense embedding models are trained to be used with cosine similarity, :class:`~sentence_transformers.sparse_encoder.SparseEncoder` models are commonly trained to be used with dot product to compute similarity. Some losses require you to provide a similarity function, and you might be better off using dot product there. Note that you can often provide the loss with :meth:`model.similarity <sentence_transformers.sparse_encoder.SparseEncoder.similarity>` or :meth:`model.similarity_pairwise <sentence_transformers.sparse_encoder.SparseEncoder.similarity_pairwise>`.

```

<!--
## Comparisons with SentenceTransformer Training

```{eval-rst}
Training :class:`~sentence_transformers.sparse_encoder.SparseEncoder` models is very similar as training :class:`~sentence_transformers.SentenceTransformer` models. 

See the `Sentence Transformer > Training Overview <../sentence_transformer/training_overview.html>`_ documentation for more details on training :class:`~sentence_transformers.SentenceTransformer` models.

```
-->