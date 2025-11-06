# Translated Sentence Mining

Bitext mining describes the process of finding parallel (translated) sentence pairs in monolingual corpora. For example, you have an set of English sentences:

```
This is an example sentences.
Hello World!
My final third sentence in this list.
```

And a set of German sentences:

```
Hallo Welt!
Dies ist ein Beispielsatz.
Dieser Satz taucht im Englischen nicht auf.
```

Here, you want to find all translation pairs between the English set and the German set of languages.

The correct (two) translations are:

```
Hello World!    Hallo Welt!
This is an example sentences.   Dies ist ein Beispielsatz.
```

Usually you apply this method to large corpora, for example, you want to find all translated sentences in the English Wikipedia and the Chinese Wikipedia.

## Margin Based Mining

We follow the setup from [Artetxe and Schwenk, Section 4.3](https://huggingface.co/papers/1812.10464) to find translated sentences in two datasets:

1. First, we encode all sentences to their respective embedding. As shown in [our paper](https://huggingface.co/papers/2004.09813) is [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) currently the best method for bitext mining. The model is integrated in Sentence-Transformers
1. Once we have all embeddings, we find the *k* nearest neighbor sentences for all sentences in both directions. Typical choices for k are between 4 and 16.
1. Then, we score all possible sentence combinations using the formula mentioned in Section 4.3.
1. The pairs with the highest scores are most likely translated sentences. Note, that the score can be larger than 1. Usually you have to find some cut-off where you ignore pairs below that threshold. For a high quality, a threshold of about 1.2 - 1.3 works quite well.

## Examples

- **[bucc2018.py](bucc2018.py)** - This script contains an example for the [BUCC 2018 shared task](https://comparable.limsi.fr/bucc2018/bucc2018-task.html) on finding parallel sentences. This dataset can be used to evaluate different strategies, as we know which sentences are parallel in the two corpora. The script mines for parallel sentences and then prints the optimal threshold that leads to the highest F1-score.
- **[bitext_mining.py](bitext_mining.py)** - This file reads in two text files (with a single sentence in each line) and outputs parallel sentences to \*parallel-sentences-out.tsv.gz.
- **[In-domain Data Selection for MT](https://www.clinjournal.org/clinj/article/view/137)** - This paper also employed Sentence Transformers to generate/select in-domain parallel data for machine translation systems – using monolingual texts.
