"""
This application demonstrates how to find duplicate questions (paraphrases) in a long
list of sentences.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

# Questions can be a long list of sentences up to 100k sentences or more.
# For demonstration purposes, we limit it to a few questions which all have on duplicate
questions = [
    "How did you catch your spouse cheating?",
    "How can I find out if my husband is cheating?",
    "Is my wife cheating?",
    "How do I know if my partner is cheating?",
    "Why is Starbucks in India overrated?",
    "Is Starbucks overrated in india?",
    "How can I lose weight fast without exercise?",
    "Can I lose weight without exercise?",
    "Which city is the best in India? Why?",
    "Which is the best city in India?",
    "How can I stay focused in class?",
    "How can I stay focused on my school work?",
    "How can I Remotely hack a mobile phone?",
    "How can I hack my phone?",
    "Where should I stay in Goa?",
    "Which are the best hotels in Goa?",
    "Why does hair turn white?",
    "What causes older peoples hair to turn grey?",
    "What is the easiest way to get followers on Quora?",
    "How do I get more followers for my Quora?",
]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Given a model and a List of strings (texts), evaluation.ParaphraseMiningEvaluator.paraphrase_mining performs a
# mining task by computing cosine similarity between all possible combinations and returning the ones with the highest scores.
# It returns a list of tuples (score, i, j) with i, j representing the index in the questions list.
pairs = paraphrase_mining(model, questions)

# Output Top-20 pairs:
for score, qid1, qid2 in pairs[0:20]:
    print(f"{score:.3f}\t{questions[qid1]}\t\t\t{questions[qid2]}")
