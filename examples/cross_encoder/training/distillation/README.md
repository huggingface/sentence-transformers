# Model Distillation

Model distillation refers to training an (often smaller) student model to mimic the behaviour of an (often larger) teacher model, or collection of teacher models. This is commonly used to make models **faster, cheaper and lighter**.

## Cross Encoder Knowledge Distillation

The goal is to minimize the difference between the student logits (a.k.a. raw model outputs) and the teacher logits on the same input pair (often a query-answer pair).

![](https://github.com/huggingface/sentence-transformers/raw/master/docs/img/msmarco-training-ce-distillation.png)

Here are two training scripts that use pre-computed logits from [Hostätter et al.](https://huggingface.co/papers/2010.02666), who trained an ensemble of 3 (large) models for the MS MARCO dataset and predicted the scores for various (query, passage)-pairs (50% positive, 50% negative).

- **[train_cross_encoder_kd_mse.py](train_cross_encoder_kd_mse.py)**
  ```{eval-rst}
  In this example, we use knowledge distillation with a small & fast model and learn the logits scores from the teacher ensemble. This yields performances comparable to large models, while being 18 times faster.

  It uses the :class:`~sentence_transformers.cross_encoder.losses.MSELoss` to minimize the distance between predicted student logits and precomputed teacher logits for (query, answer) pairs.
  ```
- **[train_cross_encoder_kd_margin_mse.py](train_cross_encoder_kd_margin_mse.py)**
  ```{eval-rst}
  This is the same setup as the previous script, but now using the :class:`~sentence_transformers.cross_encoder.losses.MarginMSELoss` as used in the aforementioned `Hostätter et al. <https://huggingface.co/papers/2010.02666>`_.

  :class:`~sentence_transformers.cross_encoder.losses.MarginMSELoss` does not work with (query, answer) pairs and a precomputed logit, but with (query, correct_answer, incorrect_answer) triplets and a precomputed logit that corresponds to ``teacher.predict([query, correct_answer]) - teacher.predict([query, incorrect_answer])``. In short, this precomputed logit is the *difference* between (query, correct_answer) and (query, incorrect_answer).
  ```

## Inference

The [tomaarsen/reranker-MiniLM-L12-H384-margin-mse](https://huggingface.co/tomaarsen/reranker-MiniLM-L12-H384-margin-mse) model was trained with the second script. If you want to try out the model before distilling a model yourself, feel free to use this script:

```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("tomaarsen/reranker-modernbert-base-msmarco-margin-mse")
# Get scores for pairs of texts
pairs = [
    ["where is joplin airport", "Scott Joplin is important both as a composer for bringing ragtime to the concert hall, setting the stage (literally) for the rise of jazz; and as an early advocate for civil rights and education among American blacks. Joplin is a hero, and a national treasure of the United States."],
    ["where is joplin airport", "Flights from Jos to Abuja will get you to this shimmering Nigerian capital within approximately 19 hours. Flights depart from Yakubu Gowon Airport/ Jos Airport (JOS) and arrive at Nnamdi Azikiwe International Airport (ABV). Arik Air is the main airline flying the Jos to Abuja route."],
    ["where is joplin airport", "Janis Joplin returned to the music scene, knowing it was her destiny, in 1966. A friend, Travis Rivers, recruited her to audition for the psychedelic band, Big Brother and the Holding Company, based in San Francisco. The band was quite big in San Francisco at the time, and Joplin landed the gig."],
    ["where is joplin airport", "Joplin Regional Airport. Joplin Regional Airport (IATA: JLN, ICAO: KJLN, FAA LID: JLN) is a city-owned airport four miles north of Joplin, in Jasper County, Missouri. It has airline service subsidized by the Essential Air Service program. Airline flights and general aviation are in separate terminals."],
    ["where is joplin airport", 'Trolley and rail lines made Joplin the hub of southwest Missouri. As the center of the "Tri-state district", it soon became the lead- and zinc-mining capital of the world. As a result of extensive surface and deep mining, Joplin is dotted with open-pit mines and mine shafts.'],
]
scores = model.predict(pairs)
print(scores)
# [0.00410349 0.03430534 0.5108879  0.999984   0.91639173]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "where is joplin airport",
    [
        "Scott Joplin is important both as a composer for bringing ragtime to the concert hall, setting the stage (literally) for the rise of jazz; and as an early advocate for civil rights and education among American blacks. Joplin is a hero, and a national treasure of the United States.",
        "Flights from Jos to Abuja will get you to this shimmering Nigerian capital within approximately 19 hours. Flights depart from Yakubu Gowon Airport/ Jos Airport (JOS) and arrive at Nnamdi Azikiwe International Airport (ABV). Arik Air is the main airline flying the Jos to Abuja route.",
        "Janis Joplin returned to the music scene, knowing it was her destiny, in 1966. A friend, Travis Rivers, recruited her to audition for the psychedelic band, Big Brother and the Holding Company, based in San Francisco. The band was quite big in San Francisco at the time, and Joplin landed the gig.",
        "Joplin Regional Airport. Joplin Regional Airport (IATA: JLN, ICAO: KJLN, FAA LID: JLN) is a city-owned airport four miles north of Joplin, in Jasper County, Missouri. It has airline service subsidized by the Essential Air Service program. Airline flights and general aviation are in separate terminals.",
        'Trolley and rail lines made Joplin the hub of southwest Missouri. As the center of the "Tri-state district", it soon became the lead- and zinc-mining capital of the world. As a result of extensive surface and deep mining, Joplin is dotted with open-pit mines and mine shafts.',
    ],
)
print(ranks)
# [
#     {"corpus_id": 3, "score": 0.999984},
#     {"corpus_id": 4, "score": 0.91639173},
#     {"corpus_id": 2, "score": 0.5108879},
#     {"corpus_id": 1, "score": 0.03430534},
#     {"corpus_id": 0, "score": 0.004103488},
# ]
```
