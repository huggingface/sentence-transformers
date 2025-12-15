from __future__ import annotations

import pytest

from sentence_transformers.cross_encoder import CrossEncoder


@pytest.mark.parametrize(
    "model_name, expected_score",
    [
        ("cross-encoder/ms-marco-MiniLM-L6-v2", [8.12545108795166, -3.045016050338745, -3.1524128913879395]),
        ("cross-encoder/ms-marco-TinyBERT-L2-v2", [8.142767906188965, 1.2057735919952393, -2.7283530235290527]),
        ("cross-encoder/stsb-distilroberta-base", [0.4977430999279022, 0.255491703748703, 0.28261035680770874]),
        ("mixedbread-ai/mxbai-rerank-xsmall-v1", [0.9224735498428345, 0.04793589934706688, 0.03315146267414093]),
    ],
)
def test_pretrained_model(model_name: str, expected_score: list[float]) -> None:
    # Ensure that pretrained models are not accidentally changed
    model = CrossEncoder(model_name, model_kwargs={"torch_dtype": "float32"})

    query = "is toprol xl the same as metoprolol?"
    answers = [
        "Metoprolol succinate is also known by the brand name Toprol XL. It is the extended-release form of metoprolol. Metoprolol succinate is approved to treat high blood pressure, chronic chest pain, and congestive heart failure.",
        "Pill with imprint 1 is White, Round and has been identified as Metoprolol Tartrate 25 mg.",
        "Interactions between your drugs No interactions were found between Allergy Relief and metoprolol. This does not necessarily mean no interactions exist. Always consult your healthcare provider.",
    ]
    scores = model.predict([(query, answer) for answer in answers])
    assert scores.tolist() == pytest.approx(expected_score, rel=1e-4)
