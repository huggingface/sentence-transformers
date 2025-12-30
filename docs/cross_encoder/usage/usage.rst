
Usage
=====

Characteristics of Cross Encoder (a.k.a reranker) models:

1. Calculates a **similarity score** given **pairs of texts**.
2. Generally provides **superior performance** compared to a Sentence Transformer (a.k.a. bi-encoder) model.
3. Often **slower** than a Sentence Transformer model, as it requires computation for each pair rather than each text.
4. Due to the previous 2 characteristics, Cross Encoders are often used to **re-rank the top-k results** from a Sentence Transformer model.

Once you have `installed <../../installation.html>`_ Sentence Transformers, you can easily use Cross Encoder models:

.. sidebar:: Documentation

   1. :class:`~sentence_transformers.cross_encoder.model.CrossEncoder`
   2. :meth:`CrossEncoder.predict <sentence_transformers.cross_encoder.model.CrossEncoder.predict>`
   3. :meth:`CrossEncoder.rank <sentence_transformers.cross_encoder.model.CrossEncoder.rank>`

   .. note::
      MS Marco models return logits rather than scores between 0 and 1. Load the :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` with ``activation_fn=torch.nn.Sigmoid()`` to get scores between 0 and 1. This does not affect the ranking.

::

   from sentence_transformers import CrossEncoder
   
   # 1. Load a pre-trained CrossEncoder model
   model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

   # 2. Predict scores for a pair of sentences
   scores = model.predict([
       ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
       ("How many people live in Berlin?", "Berlin is well known for its museums."),
   ])
   # => array([ 8.607138 , -4.3200774], dtype=float32)
   
   # 3. Rank a list of passages for a query
   query = "How many people live in Berlin?"
   passages = [
       "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
       "Berlin is well known for its museums.",
       "In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
       "The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
       "The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
       "An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
       "Berlin is subdivided into 12 boroughs or districts (Bezirke).",
       "In 2015, the total labour force in Berlin was 1.85 million.",
       "In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
       "Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
   ]
   ranks = model.rank(query, passages)
   
   # Print the scores
   print("Query:", query)
   for rank in ranks:
       print(f"{rank['score']:.2f}\t{passages[rank['corpus_id']]}")
   """
   Query: How many people live in Berlin?
   8.92    The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.
   8.61    Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.
   8.24    An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.
   7.60    In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.
   6.35    In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.
   5.42    Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.
   3.45    In 2015, the total labour force in Berlin was 1.85 million.
   0.33    Berlin is subdivided into 12 boroughs or districts (Bezirke).
   -4.24   The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019
   -4.32   Berlin is well known for its museums.
   """

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   Cross-Encoder vs Bi-Encoder <../../../examples/cross_encoder/applications/README>
   ../../../examples/sentence_transformer/applications/retrieve_rerank/README
   efficiency