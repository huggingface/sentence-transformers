"""
This script contains an example how to perform re-ranking with a Cross-Encoder for semantic search.

First, we use an efficient Bi-Encoder to retrieve similar questions from the Natural Questions dataset:
https://huggingface.co/datasets/sentence-transformers/natural-questions

Then, we re-rank the hits from the Bi-Encoder (retriever) using a Cross-Encoder (reranker).
"""

import os
import pickle
import time

from datasets import load_dataset

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search

# We use a BiEncoder (SentenceTransformer) that produces embeddings for questions.
# We then search for similar questions using cosine similarity and identify the top 100 most similar questions
# Loading https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
num_candidates = 500
batch_size = 128

# To refine the results, we use a CrossEncoder. A CrossEncoder gets both inputs (input_question, retrieved_answer)
# and outputs a score indicating the similarity.
# Loading https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Load the dataset
max_corpus_size = 20_000
dataset = load_dataset("sentence-transformers/natural-questions", split=f"train[:{max_corpus_size}]")

# Some local file to cache computed embeddings
embedding_cache_path = "natural-questions-embeddings-{}-size-{}.pkl".format(
    model_name.replace("/", "_"), max_corpus_size
)

# Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    print("Encode the questions and answers. This might take a while")
    answers = list(set(dataset["answer"]))
    answer_embeddings = model.encode(answers, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    print("Store file on disk")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({"answers": dataset["answer"], "answer_embeddings": answer_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disk")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        answers = cache_data["answers"][:max_corpus_size]
        answer_embeddings = cache_data["answer_embeddings"][:max_corpus_size]

###############################
print(f"Corpus loaded with {len(answers)} answers / embeddings")

while True:
    query = input("Please enter a question: ")
    print("Input question:", query)

    # First, retrieve candidates using cosine similarity search
    start_time = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = semantic_search(query_embedding, answer_embeddings, top_k=num_candidates)
    hits = hits[0]  # Get the hits for the first query

    print(f"Cosine-Similarity search took {time.time() - start_time:.3f} seconds")
    print("Top 5 hits with cosine-similarity:")
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit["score"], answers[hit["corpus_id"]]))

    # Now, do the re-ranking with the cross-encoder
    start_time = time.time()
    sentence_pairs = [[query, answers[hit["corpus_id"]]] for hit in hits]
    ranked_results = cross_encoder_model.rank(
        query, [answers[hit["corpus_id"]] for hit in hits], return_documents=True, top_k=5
    )

    print(f"\nRe-ranking with CrossEncoder took {time.time() - start_time:.3f} seconds")
    print("Top 5 hits with CrossEncoder:")
    for hit in ranked_results:
        print("\t{:.3f}\t{}".format(hit["score"], hit["text"]))

    print("\n\n========\n")

"""
Input question: apple sayings
Cosine-Similarity search took 0.063 seconds
Top 5 hits with cosine-similarity:
    0.602   Apple Inc. Apple Inc. is an American multinational technology company headquartered in Cupertino, California that designs, develops, and sells consumer electronics, computer software, and online services. The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple Watch smartwatch, the Apple TV digital media player, and the HomePod smart speaker. Apple's consumer software includes the macOS and iOS operating systems, the iTunes media player, the Safari web browser, and the iLife and iWork creativity and productivity suites. Its online services include the iTunes Store, the iOS App Store and Mac App Store, Apple Music, and iCloud.
    0.570   How do you like them apples The phrase is thought to have originated in World War I, with the "Toffee Apple" trench mortar used by British troops. These mortars were later rendered obsolete by the Stokes mortar, which used a more modern bullet-shaped shell.
    0.547   History of Apple Inc. Apple Inc., formerly Apple Computer, Inc., is a multinational corporation that creates consumer electronics, personal computers, servers, and computer software, and is a digital distributor of media content. The company also has a chain of retail stores known as Apple Stores. Apple's core product lines are the iPhone smart phone, iPad tablet computer, iPod portable media players, and Macintosh computer line. Founders Steve Jobs and Steve Wozniak created Apple Computer on April 1, 1976,[1] and incorporated the company on January 3, 1977,[2] in Cupertino, California.
    0.528   History of Apple Inc. On January 9, 2007, Apple Computer, Inc. shortened its name to simply Apple Inc. In his Macworld Expo keynote address, Steve Jobs explained that with their current product mix consisting of the iPod and Apple TV as well as their Macintosh brand, Apple really wasn't just a computer company anymore. At the same address, Jobs revealed a product that would revolutionize an industry in which Apple had never previously competed: the Apple iPhone. The iPhone combined Apple's first widescreen iPod with the world's first mobile device boasting visual voicemail, and an internet communicator able to run a fully functional version of Apple's web browser, Safari, on the then-named iPhone OS (later renamed iOS).
    0.522   Siri In June 2016, The Verge's Sean O'Kane wrote about the then-upcoming major iOS 10 updates, with a headline stating "Siri's big upgrades won't matter if it can't understand its users". O'Kane wrote that "What Apple didn’t talk about was solving Siri’s biggest, most basic flaws: it’s still not very good at voice recognition, and when it gets it right, the results are often clunky. And these problems look even worse when you consider that Apple now has full-fledged competitors in this space: Amazon’s Alexa, Microsoft’s Cortana, and Google’s Assistant."[61] Also writing for The Verge, Walt Mossberg had previously questioned Apple's efforts in cloud-based services, writing:[62]

Re-ranking with CrossEncoder took 0.808 seconds
Top 5 hits with CrossEncoder:
    4.776   An apple a day keeps the doctor away First recorded in the 1860s, the proverb originated in Wales, and was particularly prevalent in Pembrokshire. The first English version of the saying was "Eat an apple on going to bed, and youâ€™ll keep the doctor from earning his bread." The current phrasing ("An apple a day keeps the doctor away") was first used in print in 1922.[1][2]
    4.636   An apple a day keeps the doctor away First recorded in the 1860s, the proverb originated in Wales, and was particularly prevalent in Pembrokeshire. The first English version of the saying was "Eat an apple on going to bed, and youâ€™ll keep the doctor from earning his bread." The current english phrasing, "An apple a day keeps the doctor away", began usage at the end of the 19th century, [1][2] early print examples found as early as 1899.[3]
    2.349   Apple of my eye The Bible references below (from the King James Version, translated in 1611) contain the English idiom "apple of my eye." However the Hebrew literally says, "little man of the eye." The Hebrew idiom also refers to the pupil, and has the same meaning, but does not parallel the English use of "apple."
    2.091   Apple of my eye The Bible references below (from the King James Version, translated in 1611) contain the English idiom "apple of my eye." However the "apple" reference comes from English idiom, not biblical Hebrew. The Hebrew literally says, "dark part of the eye." The Hebrew idiom also refers to the pupil, and has the same meaning, but does not parallel the English use of "apple."
    1.445   Apple of my eye The phrase apple of my eye refers to something or someone that one cherishes above all others.[1]
"""
