# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[train]>=6.3.0",
# ]
# ///
"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
"""

import logging
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.modules import Dense

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Model for which we apply dimensionality reduction
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# New size for the embeddings
new_dimension = 128

# We measure the performance of the original model
# and later we will measure the performance with the reduces dimension size
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
stsb_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    name="sts-test",
)

logging.info("Original model performance:")
stsb_evaluator(model)

######## Reduce the embedding dimensions ########

train_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")

nli_sentences = train_dataset["sentence1"] + train_dataset["sentence2"]
random.shuffle(nli_sentences)

# To determine the PCA matrix, we need some example sentence embeddings.
# Here, we compute the embeddings for 20k random sentences from the AllNLI dataset
pca_train_sentences = nli_sentences[0:20000]
train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

# Compute PCA on the train embeddings matrix
pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)

# We add a dense layer to the model, so that it will produce directly embeddings with the new size
dense = Dense(
    in_features=model.get_embedding_dimension(),
    out_features=new_dimension,
    bias=False,
    activation_function=torch.nn.Identity(),
)
dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
model.add_module("dense", dense)

# Evaluate the model with the reduce embedding size
logging.info(f"Model with {new_dimension} dimensions:")
stsb_evaluator(model)


model_name = model_name.split("/")[-1]
final_output_dir = f"{model_name}-{new_dimension}dim"
model.save_pretrained(final_output_dir)

# Uploading requires a token with write access, provided via `hf auth login` or `HF_TOKEN`.
try:
    model.push_to_hub(final_output_dir)
except Exception:
    logging.exception("Could not upload the model. It is saved locally in %s.", final_output_dir)
