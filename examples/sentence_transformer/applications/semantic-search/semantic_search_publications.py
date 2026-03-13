"""
This example demonstrates how we can perform semantic search for scientific publications.

As model, we use SPECTER (https://github.com/allenai/specter), which encodes paper titles and abstracts
into a vector space.

When can then use util.semantic_search() to find the most similar papers.

Colab example: https://colab.research.google.com/drive/12hfBveGHRsxhPIUMmJYrll2lFU4fOX06
"""

import json
import os

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import http_get, semantic_search

# First, we load the papers dataset (with title and abstract information)
dataset_file = "emnlp2016-2018.json"

if not os.path.exists(dataset_file):
    http_get("https://sbert.net/datasets/emnlp2016-2018.json", dataset_file)

with open(dataset_file) as fIn:
    papers = json.load(fIn)

print(len(papers), "papers loaded")

# We then load the allenai-specter model with SentenceTransformers
model = SentenceTransformer("allenai-specter")

# To encode the papers, we must combine the title and the abstracts to a single string
paper_texts = [paper["title"] + "[SEP]" + paper["abstract"] for paper in papers]

# Compute embeddings for all papers
corpus_embeddings = model.encode(paper_texts, convert_to_tensor=True)


# We define a function, given title & abstract, searches our corpus for relevant (similar) papers
def search_papers(title, abstract):
    query_embedding = model.encode(title + "[SEP]" + abstract, convert_to_tensor=True)

    search_hits = semantic_search(query_embedding, corpus_embeddings)
    search_hits = search_hits[0]  # Get the hits for the first query

    print("\n\nPaper:", title)
    print("Most similar papers:")
    for hit in search_hits:
        related_paper = papers[hit["corpus_id"]]
        print(
            "{:.2f}\t{}\t{} {}".format(
                hit["score"], related_paper["title"], related_paper["venue"], related_paper["year"]
            )
        )


# This paper was the EMNLP 2019 Best Paper
search_papers(
    title="Specializing Word Embeddings (for Parsing) by Information Bottleneck",
    abstract="Pre-trained word embeddings like ELMo and BERT contain rich syntactic and semantic information, resulting in state-of-the-art performance on various tasks. We propose a very fast variational information bottleneck (VIB) method to nonlinearly compress these embeddings, keeping only the information that helps a discriminative parser. We compress each word embedding to either a discrete tag or a continuous vector. In the discrete version, our automatically compressed tags form an alternative tag set: we show experimentally that our tags capture most of the information in traditional POS tag annotations, but our tag sequences can be parsed more accurately at the same level of tag granularity. In the continuous version, we show experimentally that moderately compressing the word embeddings by our method yields a more accurate parser in 8 of 9 languages, unlike simple dimensionality reduction.",
)

# This paper was the EMNLP 2020 Best Paper
search_papers(
    title="Digital Voicing of Silent Speech",
    abstract="In this paper, we consider the task of digitally voicing silent speech, where silently mouthed words are converted to audible speech based on electromyography (EMG) sensor measurements that capture muscle impulses. While prior work has focused on training speech synthesis models from EMG collected during vocalized speech, we are the first to train from EMG collected during silently articulated speech. We introduce a method of training on silent EMG by transferring audio targets from vocalized to silent signals. Our method greatly improves intelligibility of audio generated from silent EMG compared to a baseline that only trains with vocalized data, decreasing transcription word error rate from 64% to 4% in one data condition and 88% to 68% in another. To spur further development on this task, we share our new dataset of silent and vocalized facial EMG measurements.",
)

# This paper was a EMNLP 2020 Honourable Mention Papers
search_papers(
    title="If beam search is the answer, what was the question?",
    abstract="Quite surprisingly, exact maximum a posteriori (MAP) decoding of neural language generators frequently leads to low-quality results. Rather, most state-of-the-art results on language generation tasks are attained using beam search despite its overwhelmingly high search error rate. This implies that the MAP objective alone does not express the properties we desire in text, which merits the question: if beam search is the answer, what was the question? We frame beam search as the exact solution to a different decoding objective in order to gain insights into why high probability under a model alone may not indicate adequacy. We find that beam search enforces uniform information density in text, a property motivated by cognitive science. We suggest a set of decoding objectives that explicitly enforce this property and find that exact decoding with these objectives alleviates the problems encountered when decoding poorly calibrated language generation models. Additionally, we analyze the text produced using various decoding strategies and see that, in our neural machine translation experiments, the extent to which this property is adhered to strongly correlates with BLEU.",
)

# This paper was a EMNLP 2020 Honourable Mention Papers
search_papers(
    title="Spot The Bot: A Robust and Efficient Framework for the Evaluation of Conversational Dialogue Systems",
    abstract="The lack of time efficient and reliable evalu-ation methods is hampering the development of conversational dialogue systems (chat bots). Evaluations that require humans to converse with chat bots are time and cost intensive, put high cognitive demands on the human judges, and tend to yield low quality results. In this work, we introduce Spot The Bot, a cost-efficient and robust evaluation framework that replaces human-bot conversations with conversations between bots. Human judges then only annotate for each entity in a conversation whether they think it is human or not (assuming there are humans participants in these conversations). These annotations then allow us to rank chat bots regarding their ability to mimic conversational behaviour of humans. Since we expect that all bots are eventually recognized as such, we incorporate a metric that measures which chat bot is able to uphold human-like be-havior the longest, i.e.Survival Analysis. This metric has the ability to correlate a bot’s performance to certain of its characteristics (e.g.fluency or sensibleness), yielding interpretable results. The comparably low cost of our frame-work allows for frequent evaluations of chatbots during their evaluation cycle. We empirically validate our claims by applying Spot The Bot to three domains, evaluating several state-of-the-art chat bots, and drawing comparisonsto related work. The framework is released asa ready-to-use tool.",
)

# EMNLP 2020 paper on making Sentence-BERT multilingual
search_papers(
    title="Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    abstract="We present an easy and efficient method to extend existing sentence embedding models to new languages. This allows to create multilingual versions from previously monolingual models. The training is based on the idea that a translated sentence should be mapped to the same location in the vector space as the original sentence. We use the original (monolingual) model to generate sentence embeddings for the source language and then train a new system on translated sentences to mimic the original model. Compared to other methods for training multilingual sentence embeddings, this approach has several advantages: It is easy to extend existing models with relatively few samples to new languages, it is easier to ensure desired properties for the vector space, and the hardware requirements for training is lower. We demonstrate the effectiveness of our approach for 50+ languages from various language families. Code to extend sentence embeddings models to more than 400 languages is publicly available.",
)
