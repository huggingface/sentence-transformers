"""
This file contains the templating for model cards prior to the v3.0 release. It still exists to be used alongside
SentenceTransformer.old_fit for backwards compatibility, but will be removed in a future release.
"""

from __future__ import annotations

import logging

from ..util import fullname


class ModelCardTemplate:
    __TAGS__ = ["sentence-transformers", "feature-extraction", "sentence-similarity"]
    __DEFAULT_VARS__ = {
        "{PIPELINE_TAG}": "sentence-similarity",
        "{MODEL_DESCRIPTION}": "<!--- Describe your model here -->",
        "{TRAINING_SECTION}": "",
        "{USAGE_TRANSFORMERS_SECTION}": "",
        "{EVALUATION}": "<!--- Describe how your model was evaluated -->",
        "{CITING}": "<!--- Describe where people can find more information -->",
    }

    __MODEL_CARD__ = """
---
library_name: sentence-transformers
pipeline_tag: {PIPELINE_TAG}
tags:
{TAGS}
{DATASETS}
---

# {MODEL_NAME}

This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a {NUM_DIMENSIONS} dimensional dense vector space and can be used for tasks like clustering or semantic search.

{MODEL_DESCRIPTION}

## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('{MODEL_NAME}')
embeddings = model.encode(sentences)
print(embeddings)
```

{USAGE_TRANSFORMERS_SECTION}

## Evaluation Results

{EVALUATION}

For an automated evaluation of this model, see the *Sentence Embeddings Benchmark*: [https://seb.sbert.net](https://seb.sbert.net?model_name={MODEL_NAME})

{TRAINING_SECTION}

## Full Model Architecture
```
{FULL_MODEL_STR}
```

## Citing & Authors

{CITING}

"""

    __TRAINING_SECTION__ = """
## Training
The model was trained with the parameters:

{LOSS_FUNCTIONS}

Parameters of the fit()-Method:
```
{FIT_PARAMETERS}
```
"""

    __USAGE_TRANSFORMERS__ = """\n
## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch

{POOLING_FUNCTION}

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('{MODEL_NAME}')
model = AutoModel.from_pretrained('{MODEL_NAME}')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, {POOLING_MODE} pooling.
sentence_embeddings = {POOLING_FUNCTION_NAME}(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)
```

"""

    @staticmethod
    def model_card_get_pooling_function(pooling_mode):
        if pooling_mode == "max":
            return (
                "max_pooling",
                """
# Max Pooling - Take the max value over time for every dimension.
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(token_embeddings, 1)[0]
""",
            )
        elif pooling_mode == "mean":
            return (
                "mean_pooling",
                """
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
""",
            )

        elif pooling_mode == "cls":
            return (
                "cls_pooling",
                """
def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]
""",
            )

    @staticmethod
    def get_train_objective_info(dataloader, loss):
        try:
            if hasattr(dataloader, "get_config_dict"):
                loader_params = dataloader.get_config_dict()
            else:
                loader_params = {}
                loader_params["batch_size"] = dataloader.batch_size if hasattr(dataloader, "batch_size") else "unknown"
                if hasattr(dataloader, "sampler"):
                    loader_params["sampler"] = fullname(dataloader.sampler)
                if hasattr(dataloader, "batch_sampler"):
                    loader_params["batch_sampler"] = fullname(dataloader.batch_sampler)

            dataloader_str = f"""**DataLoader**:\n\n`{fullname(dataloader)}` of length {len(dataloader)} with parameters:
```
{loader_params}
```"""

            loss_str = "**Loss**:\n\n`{}` {}".format(
                fullname(loss),
                f"""with parameters:
  ```
  {loss.get_config_dict()}
  ```"""
                if hasattr(loss, "get_config_dict")
                else "",
            )

            return [dataloader_str, loss_str]

        except Exception as e:
            logging.WARN(f"Exception when creating get_train_objective_info: {str(e)}")
            return ""
