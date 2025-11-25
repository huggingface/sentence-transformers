# Migration Guide

## Migrating from v4.x to v5.x

```{eval-rst}
The v5 Sentence Transformers release introduced :class:`~sentence_transformers.sparse_encoder.SparseEncoder` embedding models (see the `Sparse Encoder Usage <sparse_encoder/usage/usage.html>`_ for more details on them) alongside an extensive training suite for them, including :class:`~sentence_transformers.sparse_encoder.trainer.SparseEncoderTrainer` and :class:`~sentence_transformers.sparse_encoder.training_args.SparseEncoderTrainingArguments`. Unlike with v3 (updated :class:`~sentence_transformers.SentenceTransformer`) and v4 (updated :class:`~sentence_transformers.cross_encoder.CrossEncoder`), this update does not deprecate any training methods.
```

### Migration for model.encode

```{eval-rst}

We introduce two new methods, :meth:`~sentence_transformers.SentenceTransformer.encode_query` and :meth:`~sentence_transformers.SentenceTransformer.encode_document`, which are recommended to use instead of the :meth:`~sentence_transformers.SentenceTransformer.encode` method when working with information retrieval tasks. These methods are specialized version of :meth:`~sentence_transformers.SentenceTransformer.encode` that differs in exactly two ways:

1. If no ``prompt_name`` or ``prompt`` is provided, it uses a predefined "query" prompt,
   if available in the model's ``prompts`` dictionary.
2. It sets the ``task`` to "query". If the model has a :class:`~sentence_transformers.models.Router`
   module, it will use the "query" task type to route the input through the appropriate submodules.

The same methods apply to the :class:`~sentence_transformers.sparse_encoder.SparseEncoder` models.

.. list-table:: encode_query and encode_document
   :widths: 50 50
   :header-rows: 1

   * - v4.x
     - v5.x (recommended)
   * - .. code-block:: python
         :emphasize-lines: 7-9

         from sentence_transformers import SentenceTransformer

         model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
         query = "What is the capital of France?"
         document = "Paris is the capital of France."

         # Use the prompt with the name "query" for the query
         query_embedding = model.encode(query, prompt_name="query")
         document_embedding = model.encode(document)

         print(query_embedding.shape, document_embedding.shape)
         # => (1, 768) (1, 768)

     - .. code-block:: python
         :emphasize-lines: 7-12

         from sentence_transformers import SentenceTransformer

         model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
         query = "What is the capital of France?"
         document = "Paris is the capital of France."

         # The new encode_query and encode_document methods call encode,
         # but with the prompt name set to "query" or "document" if the
         # model has prompts saved, and the task set to "query" or "document",
         # if the model has a Router module.
         query_embedding = model.encode_query(query)
         document_embedding = model.encode_document(document)

         print(query_embedding.shape, document_embedding.shape)
         # => (1, 768) (1, 768)

We also deprecated the :meth:`~sentence_transformers.SentenceTransformer.encode_multi_process` method, which was used to encode large datasets in parallel using multiple processes. This method has now been subsumed by the :meth:`~sentence_transformers.SentenceTransformer.encode` method with the ``device``, ``pool``, and ``chunk_size`` arguments. Provide a list of devices to the ``device`` argument to use multiple processes, or a single device to use a single process. The ``pool`` argument can be used to pass a multiprocessing pool that gets reused across calls, and the ``chunk_size`` argument can be used to control the size of the chunks that are sent to each process in parallel.

.. list-table:: encode_multi_process deprecation -> encode
   :widths: 50 50
   :header-rows: 1

   * - v4.x
     - v5.x (recommended)
   * - .. code-block:: python
         :emphasize-lines: 7-9

         from sentence_transformers import SentenceTransformer

         def main():
             model = SentenceTransformer("all-mpnet-base-v2")
             texts = ["The weather is so nice!", "It's so sunny outside.", ...]

             pool = model.start_multi_process_pool(["cpu", "cpu", "cpu", "cpu"])
             embeddings = model.encode_multi_process(texts, pool, chunk_size=512)
             model.stop_multi_process_pool(pool)

             print(embeddings.shape)
             # => (4000, 768)

         if __name__ == "__main__":
             main()

     - .. code-block:: python
         :emphasize-lines: 7

         from sentence_transformers import SentenceTransformer

         def main():
             model = SentenceTransformer("all-mpnet-base-v2")
             texts = ["The weather is so nice!", "It's so sunny outside.", ...]

             embeddings = model.encode(texts, device=["cpu", "cpu", "cpu", "cpu"], chunk_size=512)

             print(embeddings.shape)
             # => (4000, 768)

         if __name__ == "__main__":
             main()


The ``truncate_dim`` parameter allows you to reduce the dimensionality of embeddings by truncating them. This is useful for optimizing storage and retrieval while maintaining most of the semantic information. Research has shown that the first dimensions often contain most of the important information in transformer embeddings.

.. list-table:: Add truncate_dim to encode
   :widths: 50 50
   :header-rows: 1

   * - v4.x
     - v5.x (recommended)
   * - .. code-block:: python
         :emphasize-lines: 3-8

         from sentence_transformers import SentenceTransformer

         # To truncate embeddings to a specific dimension,
         # you had to specify the dimension when loading
         model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1",
            truncate_dim=384,
         )
         sentences = ["This is an example sentence", "Each sentence is converted"]

         embeddings = model.encode(sentences)
         print(embeddings.shape)
         # => (2, 384)
     - .. code-block:: python
         :emphasize-lines: 3-7, 10-18

         from sentence_transformers import SentenceTransformer

         # Now you can either specify the dimension when loading the model...
         model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1",
            truncate_dim=384,
         )
         sentences = ["This is an example sentence", "Each sentence is converted"]

         # ... or you can specify it when encoding
         embeddings = model.encode(sentences, truncate_dim=256)
         print(embeddings.shape)
         # => (2, 256)

         # The encode parameter has priority, but otherwise the model truncate_dim is used
         embeddings = model.encode(sentences)
         print(embeddings.shape)
         # => (2, 384)

```

### Migration for Asym to Router

```{eval-rst}

The ``Asym`` module has been renamed and updated to the new :class:`~sentence_transformers.models.Router` module, which provides the same functionality but with a more consistent API and additional features. The new :class:`~sentence_transformers.models.Router` module allows for more flexible routing of different tasks, such as query and document embeddings, and is recommended when working with asymmetric models that require different processing for different tasks, notably queries and documents.

The :meth:`~sentence_transformers.SentenceTransformer.encode_query` and :meth:`~sentence_transformers.SentenceTransformer.encode_document` methods automatically set the ``task`` parameter that is used by the :class:`~sentence_transformers.models.Router` module to route the input to the query or document submodules, respectively.

.. collapse:: Asym -> Router

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 7-10

           from sentence_transformers import SentenceTransformer, models

           # Load a Sentence Transformer model and add an asymmetric router
           # for different query and document post-processing
           model = SentenceTransformer("microsoft/mpnet-base")
           dim = model.get_sentence_embedding_dimension()
           asym_model = models.Asym({
               'sts': [models.Dense(dim, dim)],
               'classification': [models.Dense(dim, dim)]
           })
           model.add_module("asym", asym_model)

        - .. code-block:: python
           :emphasize-lines: 7-10

           from sentence_transformers import SentenceTransformer, models

           # Load a Sentence Transformer model and add a router
           # for different query and document post-processing
           model = SentenceTransformer("microsoft/mpnet-base")
           dim = model.get_sentence_embedding_dimension()
           router_model = models.Router({
               'sts': [models.Dense(dim, dim)],
               'classification': [models.Dense(dim, dim)]
           })
           model.add_module("router", router_model)

.. collapse:: Asym -> Router for queries and documents

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 8-11, 22-23

           from sentence_transformers import SentenceTransformer
           from sentence_transformers.models import Router, Normalize

           # Use a regular SentenceTransformer for the document embeddings,
           # and a static embedding model for the query embeddings
           document_embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
           query_embedder = SentenceTransformer("static-retrieval-mrl-en-v1")
           asym = Asym({
               "query": list(query_embedder.children()),
               "document": list(document_embedder.children()),
           })
           normalize = Normalize()

           # Create an asymmetric model with different encoders for queries and documents
           model = SentenceTransformer(
               modules=[asym, normalize],
           )

           # ... requires more training to align the vector spaces

           # Use the query & document routes
           query_embedding = model.encode({"query": "What is the capital of France?"})
           document_embedding = model.encode({"document": "Paris is the capital of France."})

        - .. code-block:: python
           :emphasize-lines: 8-11, 22-23

           from sentence_transformers import SentenceTransformer
           from sentence_transformers.models import Router, Normalize

           # Use a regular SentenceTransformer for the document embeddings,
           # and a static embedding model for the query embeddings
           document_embedder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
           query_embedder = SentenceTransformer("static-retrieval-mrl-en-v1")
           router = Router.for_query_document(
               query_modules=list(query_embedder.children()),
               document_modules=list(document_embedder.children()),
           )
           normalize = Normalize()

           # Create an asymmetric model with different encoders for queries and documents
           model = SentenceTransformer(
               modules=[router, normalize],
           )

           # ... requires more training to align the vector spaces

           # Use the query & document routes
           query_embedding = model.encode_query("What is the capital of France?")
           document_embedding = model.encode_document("Paris is the capital of France.")

.. collapse:: Asym inference -> Router inference

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python

           ...

           # Use the query & document routes as keys in dictionaries
           query_embedding = model.encode([{"query": "What is the capital of France?"}])
           document_embedding = model.encode([
               {"document": "Paris is the capital of France."},
               {"document": "Berlin is the capital of Germany."},
           ])
           class_embedding = model.encode(
               [{"classification": "S&P500 is down 2.1% today."}],
           )

        - .. code-block:: python

           ...

           # Use the query & document routes with encode_query/encode_document
           query_embedding = model.encode_query(["What is the capital of France?"])
           document_embedding = model.encode_document([
               "Paris is the capital of France.",
               "Berlin is the capital of Germany.",
           ])

           # When using routes other than "query" and "document", you can use the `task` parameter
           # on model.encode
           class_embedding = model.encode(
               ["S&P500 is down 2.1% today."],
               task="classification"  # or any other task defined in the model Router
           )

.. collapse:: Asym training -> Router training

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 16-22

           ...

           # Prepare a training dataset for an Asym model with "query" and "document" keys
           train_dataset = Dataset.from_dict({
               "query": [
                   "is toprol xl the same as metoprolol?",
                   "are eyes always the same size?",
               ],
               "answer": [
                   "Metoprolol succinate is also known by the brand name Toprol XL.",
                   "The eyes are always the same size from birth to death.",
               ],
           })

           # This mapper turns normal texts into a dictionary mapping Asym keys to the text
           def mapper(sample):
               return {
                   "question": {"query": sample["question"]},
                   "answer": {"document": sample["answer"]},
               }

           train_dataset = train_dataset.map(mapper)
           print(train_dataset[0])
           """
           {
               "question": {"query": "is toprol xl the same as metoprolol?"},
               "answer": {"document": "Metoprolol succinate is also known by the ..."}
           }
           """

           trainer = SentenceTransformerTrainer(  # Or SparseEncoderTrainer
               model=model,
               args=training_args,
               train_dataset=train_dataset,
               ...
           )

        - .. code-block:: python
           :emphasize-lines: 25-28

           ...

           # Prepare a training dataset for a Router model with "query" and "document" keys
           train_dataset = Dataset.from_dict({
               "query": [
                   "is toprol xl the same as metoprolol?",
                   "are eyes always the same size?",
               ],
               "answer": [
                   "Metoprolol succinate is also known by the brand name Toprol XL.",
                   "The eyes are always the same size from birth to death.",
               ],
           })
           train_dataset = train_dataset.map(mapper)
           print(train_dataset[0])
           """
           {
               "question": "is toprol xl the same as metoprolol?",
               "answer": "Metoprolol succinate is also known by the brand name Toprol XL."
           }
           """

           args = SentenceTransformerTrainingArguments(  # Or SparseEncoderTrainingArguments
               # Map dataset columns to the Router keys
               router_mapping={
                   "question": "query",
                   "answer": "document",
               }
           )

           trainer = SentenceTransformerTrainer(  # Or SparseEncoderTrainer
               model=model,
               args=training_args,
               train_dataset=train_dataset,
               ...
           )

```

<br>

### Migration of advanced usage

```{eval-rst}

.. collapse:: Module and InputModule convenience superclasses

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 4

           from sentence_transformers import SentenceTransformer
           import torch

           class MyModule(torch.nn.Module):
               def __init__(self):
                   super().__init__()
                   # Custom code here

           model = SentenceTransformer(modules=[MyModule()])
        - .. code-block:: python
           :emphasize-lines: 4-9

           from sentence_transformers import SentenceTransformer
           from sentence_transformers.models import Module, InputModule

           # The new Module and InputModule superclasses provide convenience methods
           # like 'load', 'load_file_path', 'load_dir_path', 'load_torch_weights',
           # 'save_config', 'save_torch_weights', 'get_config_dict'
           # InputModule is meant to be used as the first module, is requires the
           # 'tokenize' method to be implemented
           class MyModule(Module):
               def __init__(self):
                   super().__init__()
                   # Custom initialization code here

           model = SentenceTransformer(modules=[MyModule()])

.. collapse:: Custom batch samplers via class or function

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python

           from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

           class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
               # Custom batch samplers require subclassing the Trainer
               def get_batch_sampler(
                   self,
                   dataset,
                   batch_size,
                   drop_last,
                   valid_label_columns=None,
                   generator=None,
                   seed=0,
               ):
                   # Custom batch sampler logic here
                   return ...

           ...

           trainer = CustomSentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               ...
           )
           trainer.train()
        - .. code-block:: python

             from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
             from sentence_transformers.sampler import DefaultBatchSampler
             import torch

             class CustomBatchSampler(DefaultBatchSampler):
                 def __init__(
                     self,
                     dataset: Dataset,
                     batch_size: int,
                     drop_last: bool,
                     valid_label_columns: list[str] | None = None,
                     generator: torch.Generator | None = None,
                     seed: int = 0,
                 ):
                     super().__init__(dataset, batch_size, drop_last, valid_label_columns, generator, seed)
                     # Custom batch sampler logic here

             args = SentenceTransformerTrainingArguments(
                 # Other training arguments
                 batch_sampler=CustomBatchSampler,  # Use the custom batch sampler class
             )
             trainer = SentenceTransformerTrainer(
                 model=model,
                 args=args,
                 train_dataset=train_dataset,
                 ...
             )
             trainer.train()

             # Or, use a function to initialize the batch sampler
             def custom_batch_sampler(
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool,
                 valid_label_columns: list[str] | None = None,
                 generator: torch.Generator | None = None,
                 seed: int = 0,
             ):
                 # Custom batch sampler logic here
                 return ...

             args = SentenceTransformerTrainingArguments(
                 # Other training arguments
                 batch_sampler=custom_batch_sampler,  # Use the custom batch sampler function
             )
             trainer = SentenceTransformerTrainer(
                 model=model,
                 args=args,
                 train_dataset=train_dataset,
                 ...
             )
             trainer.train()

.. collapse:: Custom multi-dataset batch samplers via class or function

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python

           from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

           class CustomSentenceTransformerTrainer(SentenceTransformerTrainer):
               def get_multi_dataset_batch_sampler(
                   self,
                   dataset: ConcatDataset,
                   batch_samplers: list[BatchSampler],
                   generator: torch.Generator | None = None,
                   seed: int | None = 0,
               ):
                   # Custom multi-dataset batch sampler logic here
                   return ...

           ...

           trainer = CustomSentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               ...
           )
           trainer.train()
        - .. code-block:: python

             from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
             from sentence_transformers.sampler import MultiDatasetDefaultBatchSampler
             import torch

             class CustomMultiDatasetBatchSampler(MultiDatasetDefaultBatchSampler):
                 def __init__(
                     self,
                     dataset: ConcatDataset,
                     batch_samplers: list[BatchSampler],
                     generator: torch.Generator | None = None,
                     seed: int = 0,
                 ):
                     super().__init__(dataset, batch_samplers=batch_samplers, generator=generator, seed=seed)
                     # Custom multi-dataset batch sampler logic here

             args = SentenceTransformerTrainingArguments(
                 # Other training arguments
                 multi_dataset_batch_sampler=CustomMultiDatasetBatchSampler,
             )
             trainer = SentenceTransformerTrainer(
                 model=model,
                 args=args,
                 train_dataset=train_dataset,
                 ...
             )
             trainer.train()

             # Or, use a function to initialize the batch sampler
             def custom_batch_sampler(
                 dataset: ConcatDataset,
                 batch_samplers: list[BatchSampler],
                 generator: torch.Generator | None = None,
                 seed: int = 0,
             ):
                 # Custom multi-dataset batch sampler logic here
                 return ...

             args = SentenceTransformerTrainingArguments(
                 # Other training arguments
                 multi_dataset_batch_sampler=custom_batch_sampler,  # Use the custom batch sampler function
             )
             trainer = SentenceTransformerTrainer(
                 model=model,
                 args=args,
                 train_dataset=train_dataset,
                 ...
             )
             trainer.train()

.. collapse:: Custom learning rate for sections

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python

           # A bunch of hacky code to set different learning rates
           # for different sections of the model

        - .. code-block:: python
           :emphasize-lines: 3-9, 14

           from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

           # Custom learning rate for each section of the model,
           # mapping regular expressions of parameter names to learning rates
           # Matching is done with 'search', not just 'match' or 'fullmatch'
           learning_rate_mapping = {
               "SparseStaticEmbedding": 1e-4,
               "linear_.*": 1e-5,
           }

           args = SentenceTransformerTrainingArguments(
               ...,
               learning_rate=1e-5,  # Default learning rate
               learning_rate_mapping=learning_rate_mapping,
           )

           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               ...
           )
           trainer.train()

.. collapse:: Training with composite losses

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 10-11

           class CustomLoss(torch.nn.Module):
               def __init__(self, model, ...):
                   super().__init__()
                   # Custom loss initialization code here

               def forward(self, features, labels):
                   loss_component_one = self.compute_loss_one(features, labels)
                   loss_component_two = self.compute_loss_two(features, labels)

                   loss = loss_component_one * alpha + loss_component_two * beta
                   return loss

            loss = CustomLoss(model, ...)

        - .. code-block:: python
            :emphasize-lines: 10-16

            class CustomLoss(torch.nn.Module):
                def __init__(self, model, ...):
                    super().__init__()
                    # Custom loss initialization code here

                def forward(self, features, labels):
                    loss_component_one = self.compute_loss_one(features, labels)
                    loss_component_two = self.compute_loss_two(features, labels)

                    # You can now return a dictionary of loss components.
                    # The trainer considers the full loss as the sum of all
                    # components, but each component will also be logged separately.
                    return {
                        "loss_one": loss_component_one,
                        "loss_two": loss_component_two,
                    }

            loss = CustomLoss(model, ...)

.. collapse:: Accessing the underlying Transformer model

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v4.x
        - v5.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 8

           from sentence_transformers import SentenceTransformer
           
           # Sometimes, for one reason or another, you need to access the underlying
           # Transformer directly. This was previously commonly done by accessing
           # the first module, often 'Transformer', and then accessing the
           # `auto_model` attribute.
           model = SentenceTransformer("all-MiniLM-L6-v2")
           print(model[0].auto_model)
           # BertModel(
           #   (embeddings): BertEmbeddings(
           # ...

        - .. code-block:: python
           :emphasize-lines: 6

           from sentence_transformers import SentenceTransformer
           
           # Now, you can just use the `transformers_model` attribute on the model itself
           # even if your model has non-standard modules.
           model = SentenceTransformer("all-MiniLM-L6-v2")
           print(model.transformers_model)
           # BertModel(
           #   (embeddings): BertEmbeddings(
           # ...

```

<br>

## Migrating from v3.x to v4.x

```{eval-rst}
The v4 Sentence Transformers release refactored the training of :class:`~sentence_transformers.cross_encoder.CrossEncoder` reranker/pair classification models, replacing :meth:`CrossEncoder.fit <sentence_transformers.SentenceTransformer.fit>` with a :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` and :class:`~sentence_transformers.cross_encoder..training_args.CrossEncoderTrainingArguments`. Like with v3 and :class:`~sentence_transformers.SentenceTransformer` models, this update softly deprecated :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`, meaning that it still works, but it's recommended to switch to the new v4.x training format. Behind the scenes, this method now uses the new trainer.

.. warning::
    If you don't have code that uses :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>`, then you will not have to make any changes to your code to update from v3.x to v4.x.

    If you do, your code still works, but it is recommended to switch to the new v4.x training format, as it allows more training arguments and functionality. See the `Training Overview <cross_encoder/training_overview.html>`_ for more details.

.. list-table:: Old and new training flow
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ::

        from sentence_transformers import CrossEncoder, InputExample
        from torch.utils.data import DataLoader

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = CrossEncoder("microsoft/mpnet-base")

        # 2. Define your train examples. You need more than just two examples...
        train_examples = [
            InputExample(texts=["What are pandas?", "The giant panda ..."], label=1),
            InputExample(texts=["What's a panda?", "Mount Vesuvius is a ..."], label=0),
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # 3. Finetune the model
        model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)
     - ::

        from datasets import load_dataset
        from sentence_transformers import CrossEncoder, CrossEncoderTrainer
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = CrossEncoder("microsoft/mpnet-base")

        # 2. Load a dataset to finetune on, convert to required format
        dataset = load_dataset("sentence-transformers/hotpotqa", "triplet", split="train")

        def triplet_to_labeled_pair(batch):
            anchors = batch["anchor"]
            positives = batch["positive"]
            negatives = batch["negative"]
            return {
                "sentence_A": anchors * 2,
                "sentence_B": positives + negatives,
                "labels": [1] * len(positives) + [0] * len(negatives),
            }

        dataset = dataset.map(triplet_to_labeled_pair, batched=True, remove_columns=dataset.column_names)
        train_dataset = dataset.select(range(10_000))
        eval_dataset = dataset.select(range(10_000, 11_000))

        # 3. Define a loss function
        loss = BinaryCrossEntropyLoss(model)

        # 4. Create a trainer & train
        trainer = CrossEncoderTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()

        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-hotpotqa")
        # model.push_to_hub("mpnet-base-hotpotqa")

```

### Migration for parameters on `CrossEncoder` initialization and methods

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ``CrossEncoder(model_name=...)``
     - Renamed to ``CrossEncoder(model_name_or_path=...)``
   * - ``CrossEncoder(automodel_args=...)``
     - Renamed to ``CrossEncoder(model_kwargs=...)``
   * - ``CrossEncoder(tokenizer_args=...)``
     - Renamed to ``CrossEncoder(tokenizer_kwargs=...)``
   * - ``CrossEncoder(config_args=...)``
     - Renamed to ``CrossEncoder(config_kwargs=...)``
   * - ``CrossEncoder(cache_dir=...)``
     - Renamed to ``CrossEncoder(cache_folder=...)``
   * - ``CrossEncoder(default_activation_function=...)``
     - Renamed to ``CrossEncoder(activation_fn=...)``
   * - ``CrossEncoder(classifier_dropout=...)``
     - Use ``CrossEncoder(config_kwargs={"classifier_dropout": ...})`` instead.
   * - ``CrossEncoder.predict(activation_fct=...)``
     - Renamed to ``CrossEncoder.predict(activation_fn=...)``
   * - ``CrossEncoder.rank(activation_fct=...)``
     - Renamed to ``CrossEncoder.rank(activation_fn=...)``
   * - ``CrossEncoder.predict(num_workers=...)``
     - Fully deprecated, no longer has any effect.
   * - ``CrossEncoder.rank(num_workers=...)``
     - Fully deprecated, no longer has any effect.

.. note::

   The old keyword arguments still work, but they will emit a warning recommending you to use the new names instead.
```

### Migration for specific parameters from `CrossEncoder.fit`

```{eval-rst}
.. collapse:: CrossEncoder.fit(train_dataloader)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 8-12, 15

           from sentence_transformers import CrossEncoder, InputExample
           from torch.utils.data import DataLoader

           # 1. Define the model. Either from scratch of by loading a pre-trained model
           model = CrossEncoder("microsoft/mpnet-base")

           # 2. Define your train examples. You need more than just two examples...
           train_examples = [
               InputExample(texts=["What are pandas?", "The giant panda ..."], label=1),
               InputExample(texts=["What's a panda?", "Mount Vesuvius is a ..."], label=0),
           ]
           train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

           # 3. Finetune the model
           model.fit(train_dataloader=train_dataloader)
        - .. code-block:: python
           :emphasize-lines: 6-18, 26

           from datasets import Dataset
           from sentence_transformers import CrossEncoder, CrossEncoderTrainer
           from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

           # Define a training dataset
           train_examples = [
               {
                   "sentence_1": "A person on a horse jumps over a broken down airplane.",
                   "sentence_2": "A person is outdoors, on a horse.",
                   "label": 1,
               },
               {
                   "sentence_1": "Children smiling and waving at camera",
                   "sentence_2": "The kids are frowning",
                   "label": 0,
               },
           ]
           train_dataset = Dataset.from_list(train_examples)

           # Define a loss function
           loss = BinaryCrossEntropyLoss(model)

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(loss_fct)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               loss_fct=torch.nn.MSELoss(),
           )
        - .. code-block:: python
           :emphasize-lines: 1, 6, 7, 14

           from sentence_transformers.cross_encoder.losses import MSELoss

           ...

           # Prepare the loss function
           # See all valid losses in https://sbert.net/docs/cross_encoder/loss_overview.html
           loss = MSELoss(model)

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(evaluator)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 9

           ...

           # Load an evaluator
           evaluator = CrossEncoderNanoBEIREvaluator()

           # Finetune with an evaluator
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
           )
        - .. code-block:: python
           :emphasize-lines: 10

           # Load an evaluator
           evaluator = CrossEncoderNanoBEIREvaluator()

           # Finetune with an evaluator
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(epochs)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               epochs=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               num_train_epochs=1,
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(activation_fct)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               activation_fct=torch.nn.Sigmoid(),
           )
        - .. code-block:: python
           :emphasize-lines: 4

           ...

           # Prepare the loss function
           loss = MSELoss(model, activation_fn=torch.nn.Sigmoid())

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(scheduler)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               scheduler="WarmupLinear",
           )
        - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               # See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
               lr_scheduler_type="linear"
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(warmup_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               warmup_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               warmup_steps=1000,
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(optimizer_class, optimizer_params)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               optimizer_class=torch.optim.AdamW,
               optimizer_params={"eps": 1e-7},
           )
        - .. code-block:: python
           :emphasize-lines: 6-7

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               # See https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
               optim="adamw_torch",
               optim_args={"eps": 1e-7},
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(weight_decay)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               weight_decay=0.02,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               weight_decay=0.02,
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(evaluation_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6, 7

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
               evaluation_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 10, 15, 17

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               eval_strategy="steps",
               eval_steps=1000,
           )

           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to evaluate
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(output_path, save_best_model)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 7, 8

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               evaluator=evaluator,
               output_path="my/path",
               save_best_model=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 19

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               load_best_model_at_end=True,
               metric_for_best_model="hotpotqa_ndcg@10", # E.g. `evaluator.primary_metric`
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

           # Save the best model at my output path
           model.save_pretrained("my/path")

.. collapse:: CrossEncoder.fit(max_grad_norm)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               max_grad_norm=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               max_grad_norm=1,
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(use_amp)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               use_amp=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               fp16=True,
               bf16=False, # If your GPU supports it, you can also use bf16 instead
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(callback)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 3, 4, 9

           ...

           def printer_callback(score, epoch, steps):
               print(f"Score: {score:.4f} at epoch {epoch:d}, step {steps:d}")

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               callback=printer_callback,
           )
        - .. code-block:: python
           :emphasize-lines: 1, 5-10, 17

           from transformers import TrainerCallback

           ...

           class PrinterCallback(TrainerCallback):
               # Subclass any method from https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
               def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                   print(f"Metrics: {metrics} at epoch {state.epoch:d}, step {state.global_step:d}")

           printer_callback = PrinterCallback()

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
               callbacks=[printer_callback],
           )
           trainer.train()

.. collapse:: CrossEncoder.fit(show_progress_bar)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v3.x
        - v4.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_dataloader=train_dataloader,
               show_progress_bar=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = CrossEncoderTrainingArguments(
               disable_tqdm=False,
           )

           # Finetune the model
           trainer = CrossEncoderTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. raw:: html

   <br>

.. note::

   The old :meth:`CrossEncoder.fit <sentence_transformers.cross_encoder.CrossEncoder.fit>` method still works, it was only softly deprecated. It now uses the new :class:`~sentence_transformers.cross_encoder.trainer.CrossEncoderTrainer` behind the scenes.
```

### Migration for CrossEncoder evaluators

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v3.x
     - v4.x (recommended)
   * - ``CEBinaryAccuracyEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CEBinaryClassificationEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CECorrelationEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderCorrelationEvaluator`, this evaluator was renamed.
   * - ``CEF1Evaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CESoftmaxAccuracyEvaluator``
     - Use :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderClassificationEvaluator`, an encompassed evaluator which uses the same inputs & outputs.
   * - ``CERerankingEvaluator``
     - Renamed to :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`, this evaluator was renamed

.. note::

   The old evaluators still work, they will simply warn you to update to the new evaluators.
```

## Migrating from v2.x to v3.x

```{eval-rst}
The v3 Sentence Transformers release refactored the training of :class:`~sentence_transformers.SentenceTransformer` embedding models, replacing :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>` with a :class:`~sentence_transformers.trainer.SentenceTransformerTrainer` and :class:`~sentence_transformers.training_args.SentenceTransformerTrainingArguments`. This update softly deprecated :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>`, meaning that it still works, but it's recommended to switch to the new v3.x training format. Behind the scenes, this method now uses the new trainer.

.. warning::
    If you don't have code that uses :meth:`SentenceTransformer.fit <sentence_transformers.SentenceTransformer.fit>`, then you will not have to make any changes to your code to update from v2.x to v3.x.

    If you do, your code still works, but it is recommended to switch to the new v3.x training format, as it allows more training arguments and functionality. See the `Training Overview <sentence_transformer/training_overview.html>`_ for more details.

.. list-table:: Old and new training flow
   :widths: 50 50
   :header-rows: 1

   * - v2.x
     - v3.x (recommended)
   * - ::

        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer("microsoft/mpnet-base")

        # 2. Define your train examples. You need more than just two examples...
        train_examples = [
            InputExample(texts=[
                "A person on a horse jumps over a broken down airplane.",
                "A person is outdoors, on a horse.",
                "A person is at a diner, ordering an omelette.",
            ]),
            InputExample(texts=[
                "Children smiling and waving at camera",
                "There are children present",
                "The kids are frowning",
            ]),
        ]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

        # 3. Define a loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 4. Finetune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=100,
        )

        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-all-nli")
     - ::

        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
        from sentence_transformers.losses import MultipleNegativesRankingLoss

        # 1. Define the model. Either from scratch of by loading a pre-trained model
        model = SentenceTransformer("microsoft/mpnet-base")

        # 2. Load a dataset to finetune on
        dataset = load_dataset("sentence-transformers/all-nli", "triplet")
        train_dataset = dataset["train"].select(range(10_000))
        eval_dataset = dataset["dev"].select(range(1_000))

        # 3. Define a loss function
        loss = MultipleNegativesRankingLoss(model)

        # 4. Create a trainer & train
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()

        # 5. Save the trained model
        model.save_pretrained("models/mpnet-base-all-nli")
        # model.push_to_hub("mpnet-base-all-nli")
```

### Migration for specific parameters from `SentenceTransformer.fit`

```{eval-rst}
.. collapse:: SentenceTransformer.fit(train_objectives)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 5-17, 20, 23

           from sentence_transformers import SentenceTransformer, InputExample, losses
           from torch.utils.data import DataLoader

           # Define a training dataloader
           train_examples = [
               InputExample(texts=[
                   "A person on a horse jumps over a broken down airplane.",
                   "A person is outdoors, on a horse.",
                   "A person is at a diner, ordering an omelette.",
               ]),
               InputExample(texts=[
                   "Children smiling and waving at camera",
                   "There are children present",
                   "The kids are frowning",
               ]),
           ]
           train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

           # Define a loss function
           train_loss = losses.MultipleNegativesRankingLoss(model)

           # Finetune the model
           model.fit(train_objectives=[(train_dataloader, train_loss)])
        - .. code-block:: python
           :emphasize-lines: 6-18, 21, 26, 27

           from datasets import Dataset
           from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
           from sentence_transformers.losses import MultipleNegativesRankingLoss

           # Define a training dataset
           train_examples = [
               {
                   "anchor": "A person on a horse jumps over a broken down airplane.",
                   "positive": "A person is outdoors, on a horse.",
                   "negative": "A person is at a diner, ordering an omelette.",
               },
               {
                   "anchor": "Children smiling and waving at camera",
                   "positive": "There are children present",
                   "negative": "The kids are frowning",
               },
           ]
           train_dataset = Dataset.from_list(train_examples)

           # Define a loss function
           loss = MultipleNegativesRankingLoss(model)

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(evaluator)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 9

           ...

           # Load an evaluator
           evaluator = NanoBEIREvaluator()

           # Finetune with an evaluator
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
           )
        - .. code-block:: python
           :emphasize-lines: 10

           # Load an evaluator
           evaluator = NanoBEIREvaluator()

           # Finetune with an evaluator
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(epochs)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               epochs=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               num_train_epochs=1,
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(steps_per_epoch)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               steps_per_epoch=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               max_steps=1000, # Note: max_steps is across all epochs, not per epoch
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(scheduler)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               scheduler="WarmupLinear",
           )
        - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               # See https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
               lr_scheduler_type="linear"
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(warmup_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               warmup_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               warmup_steps=1000,
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(optimizer_class, optimizer_params)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               optimizer_class=torch.optim.AdamW,
               optimizer_params={"eps": 1e-7},
           )
        - .. code-block:: python
           :emphasize-lines: 6-7

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               # See https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
               optim="adamw_torch",
               optim_args={"eps": 1e-7},
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(weight_decay)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               weight_decay=0.02,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               weight_decay=0.02,
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(evaluation_steps)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6, 7

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
               evaluation_steps=1000,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 10, 15, 17

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               eval_strategy="steps",
               eval_steps=1000,
           )

           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to evaluate
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
               evaluator=evaluator,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(output_path, save_best_model)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 7, 8

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               evaluator=evaluator,
               output_path="my/path",
               save_best_model=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6, 19

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               load_best_model_at_end=True,
               metric_for_best_model="all_nli_cosine_accuracy", # E.g. `evaluator.primary_metric`
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

           # Save the best model at my output path
           model.save_pretrained("my/path")

.. collapse:: SentenceTransformer.fit(max_grad_norm)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               max_grad_norm=1,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               max_grad_norm=1,
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(use_amp)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               use_amp=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5, 6

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               fp16=True,
               bf16=False, # If your GPU supports it, you can also use bf16 instead
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(callback)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 3, 4, 9

           ...

           def printer_callback(score, epoch, steps):
               print(f"Score: {score:.4f} at epoch {epoch:d}, step {steps:d}")

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               callback=printer_callback,
           )
        - .. code-block:: python
           :emphasize-lines: 1, 5-10, 17

           from transformers import TrainerCallback

           ...

           class PrinterCallback(TrainerCallback):
               # Subclass any method from https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
               def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                   print(f"Metrics: {metrics} at epoch {state.epoch:d}, step {state.global_step:d}")

           printer_callback = PrinterCallback()

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               train_dataset=train_dataset,
               loss=loss,
               callbacks=[printer_callback],
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(show_progress_bar)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               show_progress_bar=True,
           )
        - .. code-block:: python
           :emphasize-lines: 5

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               disable_tqdm=False,
           )

           # Finetune the model
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               loss=loss,
           )
           trainer.train()

.. collapse:: SentenceTransformer.fit(checkpoint_path, checkpoint_save_steps, checkpoint_save_total_limit)

   .. list-table::
      :widths: 50 50
      :header-rows: 1

      * - v2.x
        - v3.x (recommended)
      * - .. code-block:: python
           :emphasize-lines: 6-8

           ...

           # Finetune the model
           model.fit(
               train_objectives=[(train_dataloader, train_loss)],
               checkpoint_path="checkpoints",
               checkpoint_save_steps=5000,
               checkpoint_save_total_limit=2,
           )
        - .. code-block:: python
           :emphasize-lines: 7-9, 13, 18

           ...

           # Prepare the Training Arguments
           args = SentenceTransformerTrainingArguments(
               eval_strategy="steps",
               eval_steps=5000,
               save_strategy="steps",
               save_steps=5000,
               save_total_limit=2,
           )

           # Finetune the model
           # Note: You need an eval_dataset and/or evaluator to checkpoint
           trainer = SentenceTransformerTrainer(
               model=model,
               args=args,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               loss=loss,
           )
           trainer.train()
```

<br>

### Migration for custom Datasets and DataLoaders used in `SentenceTransformer.fit`

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - v2.x
     - v3.x (recommended)
   * - ``ParallelSentencesDataset``
     - Manually creating a :class:`~datasets.Dataset` and adding a ``label`` column for embeddings. Alternatively, consider loading one of our pre-provided `Parallel Sentences Datasets <https://huggingface.co/collections/sentence-transformers/parallel-sentences-datasets-6644d644123d31ba5b1c8785>`_.
   * - ``SentenceLabelDataset``
     - Loading or creating a :class:`~datasets.Dataset` and using ``SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.GROUP_BY_LABEL)`` (uses the :class:`~sentence_transformers.sampler.GroupByLabelBatchSampler`). Recommended for the BatchTripletLosses.
   * - ``DenoisingAutoEncoderDataset``
     - Manually adding a column with noisy text to a :class:`~datasets.Dataset` with texts, e.g. with :func:`Dataset.map <datasets.Dataset.map>`.
   * - ``NoDuplicatesDataLoader``
     - Loading or creating a :class:`~datasets.Dataset` and using ``SentenceTransformerTrainingArguments(batch_sampler=BatchSamplers.NO_DUPLICATES)`` (uses the :class:`~sentence_transformers.sampler.NoDuplicatesBatchSampler`). Recommended for :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`.
```
