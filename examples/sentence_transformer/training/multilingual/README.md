# Multilingual Models

The issue with multilingual BERT (mBERT) as well as with XLM-RoBERTa is that those produce rather bad sentence representation out-of-the-box. Further, the vectors spaces between languages are not aligned, i.e., the sentences with the same content in different languages would be mapped to different locations in the vector space.

In my publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://huggingface.co/papers/2004.09813) I describe an easy approach to extend sentence embeddings to further languages.

Chien Vu also wrote a nice blog article on this technique: [A complete guide to transfer learning from English to other Languages using Sentence Embeddings BERT Models](https://medium.com/data-science/a-complete-guide-to-transfer-learning-from-english-to-other-languages-using-sentence-embeddings-8c427f8804a9)

## Extend your own models

![Multilingual Knowledge Distillation](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/multilingual-distillation.png)

The idea is based on a fixed (monolingual) **teacher model** that produces sentence embeddings with our desired properties in one language (e.g. English). The **student model** is supposed to mimic the teacher model, i.e., the same English sentence should be mapped to the same vector by the teacher and by the student model. Additionally, in order to make the student model work for other languages, we train the student model on parallel (translated) sentences. The translation of each sentence should also be mapped to the same vector as the original sentence.

In the above figure, the student model should map *Hello World* and the German translation *Hallo Welt* to the vector of `teacher_model('Hello World')`. We achieve this by training the student model using mean squared error (MSE) loss.

In our experiments we initialized the student model with the multilingual [XLM-RoBERTa model](https://huggingface.co/FacebookAI/xlm-roberta-base).

## Training

For a **fully automatic code example**, see [make_multilingual.py](make_multilingual.py).

This scripts downloads the parallel sentences corpus, a corpus with transcripts and translations from talks. It than extends a monolingual model to several languages (en, de, es, it, fr, ar, tr). This corpus contains parallel data for more than 100 languages, hence, you can simple change the script and train a multilingual model in your favorite languages.

## Datasets

```{eval-rst}
As training data we require parallel sentences, i.e., sentences translated in various languages. In particular, we will use :class:`~datasets.Dataset` instances with ``"english"`` and ``"non_english"`` columns. We have prepared a large collection of such datasets in our `Parallel Sentences dataset collection <https://huggingface.co/collections/sentence-transformers/parallel-sentences-datasets-6644d644123d31ba5b1c8785>`_.
```

The training script will take the `"english"` column and add a `"label"` column containing the embeddings of the english texts. Then, the student model `"english"` and `"non_english"` will be trained to be similar to this `"label"`. You can load such a training dataset like so:

```python
from datasets import load_dataset

train_dataset = load_dataset("sentence-transformers/parallel-sentences-talks", "en-de", split="train")
print(train_dataset[0])
# {"english": "So I think practicality is one case where it's worth teaching people by hand.", "non_english": "Ich denke, dass es sich aus diesem Grund lohnt, den Leuten das Rechnen von Hand beizubringen."}
```

## Sources for Training Data

A great website for a vast number of parallel (translated) datasets is [OPUS](http://opus.nlpl.eu/). There, you find parallel datasets for more than 400 languages. You can use these to create your own parallel sentence datasets, if you wish.

## Evaluation

Training can be evaluated in different ways. For an example how to use these evaluation methods, see [make_multilingual.py](make_multilingual.py).

### MSE Evaluation

You can measure the mean squared error (MSE) between the student embeddings and teacher embeddings.

```python
from datasets import load_dataset

eval_dataset = load_dataset("sentence-transformers/parallel-sentences-talks", "en-fr", split="dev")

dev_mse = MSEEvaluator(
    source_sentences=eval_dataset["english"],
    target_sentences=eval_dataset["non_english"],
    name="en-fr-dev",
    teacher_model=teacher_model,
    batch_size=32,
)
```

This evaluator computes the teacher embeddings for the `source_sentences`, for example, for English. During training, the student model is used to compute embeddings for the `target_sentences`, for example, for French. The distance between teacher and student embeddings is measures. Lower scores indicate a better performance.

### Translation Accuracy

You can also measure the translation accuracy. As inputs, this evaluator accepts a list of `source_sentences` (e.g. English), and a list of `target_sentences` (e.g. Spanish), such that `target_sentences[i]` is a translation of `source_sentences[i]`.

For each sentence pair, we check if `source_sentences[i]` we check if `target_sentences[i]` has the highest similarity out of all target sentences. If this is the case, we have a hit, otherwise an error. This evaluator reports accuracy (higher = better).

```python
from datasets import load_dataset

eval_dataset = load_dataset("sentence-transformers/parallel-sentences-talks", "en-fr", split="dev")

dev_trans_acc = TranslationEvaluator(
    source_sentences=eval_dataset["english"],
    target_sentences=eval_dataset["non_english"],
    name="en-fr-dev",
    batch_size=32,
)
```

### Multilingual Semantic Textual Similarity

You can also measure the semantic textual similarity (STS) between sentence pairs in different languages:

```python
from datasets import load_dataset

test_dataset = load_dataset("mteb/sts17-crosslingual-sts", "nl-en", split="test")

test_emb_similarity = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=[score / 5.0 for score in test_dataset["score"]],  # Convert 0-5 scores to 0-1 scores
    batch_size=32,
    name=f"sts17-nl-en-test",
    show_progress_bar=False,
)
```

Where `sentences1` and `sentences2` are lists of sentences and score is numeric value indicating the semantic similarity between `sentences1[i]` and `sentences2[i]`.

## Available Pre-trained Models

For a list of available models, see [Pretrained Models](../../../../docs/sentence_transformer/pretrained_models.md#multilingual-models).

## Usage

You can use the models in the following way:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(["Hello World", "Hallo Welt", "Hola mundo", "Bye, Moon!"])
similarities = model.similarity(embeddings, embeddings)
# tensor([[1.0000, 0.9429, 0.8880, 0.4558],
#         [0.9429, 1.0000, 0.9680, 0.5307],
#         [0.8880, 0.9680, 1.0000, 0.4933],
#         [0.4558, 0.5307, 0.4933, 1.0000]])
```

## Performance

The performance was evaluated on the [Semantic Textual Similarity (STS) 2017 dataset](https://web.archive.org/web/20230321115929/http://ixa2.si.ehu.eus/stswiki/index.php/Main_Page). The task is to predict the semantic similarity (on a scale 0-5) of two given sentences. STS2017 has monolingual test data for English, Arabic, and Spanish, and cross-lingual test data for English-Arabic, -Spanish and -Turkish.

We extended the STS2017 and added cross-lingual test data for English-German, French-English, Italian-English, and Dutch-English ([STS2017-extended.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip)). The performance is measured using Spearman correlation between the predicted similarity score and the gold score.

<table class="docutils">
  <tr>
    <th>Model</th>
    <th>AR-AR</th>
    <th>AR-EN</th>
    <th>ES-ES</th>
    <th>ES-EN</th>
    <th>EN-EN</th>
    <th>TR-EN</th>
    <th>EN-DE</th>
    <th>FR-EN</th>
    <th>IT-EN</th>
    <th>NL-EN</th>
    <th>Average</th>
  </tr>
  <tr>
    <td>XLM-RoBERTa mean pooling </td>
    <td align="center">25.7</td>
    <td align="center">17.4</td>
    <td align="center">51.8</td>
    <td align="center">10.9</td>
    <td align="center">50.7</td>
    <td align="center">9.2</td>
    <td align="center">21.3</td>
    <td align="center">16.6</td>
    <td align="center">22.9</td>
    <td align="center">26.0</td>
    <td align="center">25.2</td>
  </tr>
  <tr>
    <td>mBERT mean pooling </td>
    <td align="center">50.9</td>
    <td align="center">16.7</td>
    <td align="center">56.7</td>
    <td align="center">21.5</td>
    <td align="center">54.4</td>
    <td align="center">16.0</td>
    <td align="center">33.9</td>
    <td align="center">33.0</td>
    <td align="center">34.0</td>
    <td align="center">35.6</td>
    <td align="center">35.3</td>
  </tr>
  <tr>
    <td>LASER</td>
    <td align="center">68.9</td>
    <td align="center">66.5</td>
    <td align="center">79.7</td>
    <td align="center">57.9</td>
    <td align="center">77.6</td>
    <td align="center">72.0</td>
    <td align="center">64.2</td>
    <td align="center">69.1</td>
    <td align="center">70.8</td>
    <td align="center">68.5</td>
    <td align="center">69.5</td>
  </tr> 
  <tr>
    <td colspan="12"><b>Sentence Transformer Models</b></td>
  </tr>
  <tr>
  <td>distiluse-base-multilingual-cased</td>
    <td align="center">75.9</td>
    <td align="center">77.6</td>
    <td align="center">85.3</td>
    <td align="center">78.7</td>
    <td align="center">85.4</td>
    <td align="center">75.5</td>
    <td align="center">80.3</td>
    <td align="center">80.2</td>
    <td align="center">80.5</td>
    <td align="center">81.7</td>
    <td align="center">80.1</td>
    </tr>
</table>

## Citation

If you use the code for multilingual models, feel free to cite our publication [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://huggingface.co/papers/2004.09813):

```
@article{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    journal= "arXiv preprint arXiv:2004.09813",
    month = "04",
    year = "2020",
    url = "http://arxiv.org/abs/2004.09813",
}
```
