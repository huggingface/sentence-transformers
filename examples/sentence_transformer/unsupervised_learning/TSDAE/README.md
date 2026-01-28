# TSDAE

This section shows an example, of how we can train an unsupervised [TSDAE (Transformer-based Denoising AutoEncoder)](https://huggingface.co/papers/2104.06979) model with pure sentences as training data.

## Background

During training, TSDAE encodes damaged sentences into fixed-sized vectors and requires the decoder to reconstruct the original sentences from these sentence embeddings. For good reconstruction quality, the semantics must be captured well in the sentence embeddings from the encoder. Later, at inference, we only use the encoder for creating sentence embeddings. The architecture is illustrated in the figure below:

![](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/TSDAE.png)

## Unsupervised Training with TSDAE

Training with TSDAE is simple. You just need a set of sentences:

```python
import random

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import DenoisingAutoEncoderLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# 1. Define a SentenceTransformer model
model = SentenceTransformer("bert-base-uncased")

# 2. Some example sentences
sentences = [
    "This is an example sentence.",
    "Each sentence will be noised and reconstructed.",
    "TSDAE learns good sentence embeddings.",
    "Sentence Transformers make it easy to train models.",
]

dataset = Dataset.from_dict({"text": sentences})


def noise_transform(batch, del_ratio=0.6):
    noisy = []
    for text in batch["text"]:
        words = text.split()
        keep_prob = 1.0 - del_ratio
        kept_words = [w for w in words if random.random() < keep_prob]
        noisy.append(" ".join(kept_words))
    return {"noisy": noisy, "text": batch["text"]}


# 3. Add a lazy transform that adds noise to the sentences on-the-fly
dataset.set_transform(transform=lambda batch: noise_transform(batch), columns=["text"], output_all_columns=True)

# 4. Define the TSDAE loss
train_loss = DenoisingAutoEncoderLoss(
    model,
    decoder_name_or_path="bert-base-uncased",
    tie_encoder_decoder=True,
)

# 5. Initialize a simple training arguments & trainer
args = SentenceTransformerTrainingArguments(
    output_dir="output/tsdae-example",
    num_train_epochs=1,
    per_device_train_batch_size=4,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=train_loss,
)

# 6. Train and save the model
trainer.train()
model.save_pretrained("output/tsdae-example/final")
```

See **[train_stsb_tsdae.py](train_stsb_tsdae.py)** for the complete code.

## TSDAE from Sentences File

**[train_tsdae_from_file.py](train_tsdae_from_file.py)** loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

TSDAE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

## TSDAE on AskUbuntu Dataset

The [AskUbuntu dataset](https://github.com/taolei87/askubuntu) is a manually annotated dataset for the [AskUbuntu forum](https://askubuntu.com/). For 400 questions, experts annotated for each question 20 other questions if they are related or not. The questions are split into train & development set.

**[train_askubuntu_tsdae.py](train_askubuntu_tsdae.py)** - Shows an example how to train a model on AskUbuntu using only sentences without any labels. As sentences, we use the titles that are not used in the dev / test set.

| Model | MAP-Score on test set |
| ---- | :----: |
| TSDAE (bert-base-uncased) | 59.4 |
| **pretrained SentenceTransformer models** | |
| nli-bert-base | 50.7 |
| paraphrase-distilroberta-base-v1 | 54.8 |
| stsb-roberta-large | 54.6 |


## TSDAE as Pre-Training Task

As we show in our [TSDAE paper](https://huggingface.co/papers/2104.06979), TSDAE also a powerful pre-training method outperforming the classical Mask Language Model (MLM) pre-training task.

You first train your model with the TSDAE loss. After you have trained for a certain number of steps / after the model converges, you can further fine-tune your pre-trained model like any other SentenceTransformer model.

## Citation

If you use the code for augmented sbert, feel free to cite our publication [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning](https://huggingface.co/papers/2104.06979):

```bibtex
@article{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and  Gurevych, Iryna", 
    journal= "arXiv preprint arXiv:2104.06979",
    month = "4",
    year = "2021",
    url = "https://arxiv.org/abs/2104.06979",
}
```
