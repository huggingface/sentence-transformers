"""
In this example we train a semantic search model to search through Wikipedia
articles about programming articles & technologies.

We use the text paragraphs from the following Wikipedia articles:
Assembly language, C , C Sharp , C++, Go , Java , JavaScript, Keras, Laravel, MATLAB, Matplotlib, MongoDB, MySQL, Natural Language Toolkit, NumPy, pandas (software), Perl, PHP, PostgreSQL, Python , PyTorch, R , React, Rust , Scala , scikit-learn, SciPy, Swift , TensorFlow, Vue.js

In:
1_programming_query_generation.py - We generate queries for all paragraphs from these articles
2_programming_train_bi-encoder.py - We train a SentenceTransformer bi-encoder with these generated queries. This results in a model we can then use for semantic search (for the given Wikipedia articles).
3_programming_semantic_search.py - Shows how the trained model can be used for semantic search
"""

import os

from datasets import Dataset

from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

train_examples = []
with open("generated_queries.tsv", encoding="utf-8") as fIn:
    for line in fIn:
        query, paragraph = line.strip().split("\t", maxsplit=1)
        train_examples.append({"sentence1": query, "sentence2": paragraph})

train_dataset = Dataset.from_list(train_examples)

# Now we create a SentenceTransformer model from scratch
word_emb = models.Transformer("distilbert-base-uncased")
pooling = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pooling])

# MultipleNegativesRankingLoss requires input pairs (query, relevant_passage)
# and trains the model so that is is suitable for semantic search
train_loss = losses.MultipleNegativesRankingLoss(model)


# Tune the model
num_epochs = 3
batch_size = 64

# Prepare the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="output/programming-model",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    warmup_ratio=0.1,
    learning_rate=2e-5,
    save_strategy="no",
    logging_steps=0.01,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
)

# Train the model
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

trainer.train()

os.makedirs("output", exist_ok=True)
model.save("output/programming-model")
