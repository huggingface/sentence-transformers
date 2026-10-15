from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from packaging.version import Version
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import __version__ as transformers_version

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules.static_embedding import StaticEmbedding

try:
    import model2vec
except ImportError:
    model2vec = None

skip_if_no_model2vec = pytest.mark.skipif(model2vec is None, reason="The model2vec library is not installed.")
skip_if_transformers_5_or_higher = pytest.mark.skipif(
    Version(transformers_version) >= Version("5.0.0rc0"), reason="Transformers version is v5.0.0rc0 or higher."
)


def test_initialization_with_embedding_weights(tokenizer: Tokenizer, embedding_weights) -> None:
    model = StaticEmbedding(tokenizer, embedding_weights=embedding_weights)
    assert model.embedding.weight.shape == (30522, 768)


def test_initialization_with_embedding_dim(tokenizer: Tokenizer) -> None:
    model = StaticEmbedding(tokenizer, embedding_dim=768)
    assert model.embedding.weight.shape == (30522, 768)


def test_tokenize(static_embedding: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.preprocess(texts)
    assert "input_ids" in tokens
    assert "offsets" in tokens


def test_forward(static_embedding: StaticEmbedding) -> None:
    texts = ["Hello world!", "How are you?"]
    tokens = static_embedding.preprocess(texts)
    output = static_embedding(tokens)
    assert "sentence_embedding" in output


def test_save_and_load(tmp_path: Path, static_embedding: StaticEmbedding) -> None:
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    static_embedding.save(str(save_dir))

    loaded_model = StaticEmbedding.load(str(save_dir))
    assert loaded_model.embedding.weight.shape == static_embedding.embedding.weight.shape


@skip_if_transformers_5_or_higher()  # Model2vec distillation is not yet compatible with transformers v5+
@skip_if_no_model2vec()
def test_from_distillation() -> None:
    model = StaticEmbedding.from_distillation("sentence-transformers-testing/stsb-bert-tiny-safetensors", pca_dims=32)
    # The shape has been 29528 for <0.5.0, 29525 for 0.5.0, and 29524 for >=0.6.0, so let's make a safer test
    # that checks the first dimension is close to 29525 and the second dimension is 32.
    assert abs(model.embedding.weight.shape[0] - 29525) < 5
    assert model.embedding.weight.shape[1] == 32


@pytest.mark.parametrize(
    "device_kwargs", [{}, {"device": None}, {"device": "cpu"}, {"device": "cuda:1"}, {"device": "mps"}]
)
def test_from_distillation_passes_device_to_model2vec(
    monkeypatch: pytest.MonkeyPatch, device_kwargs: dict[str, str | None]
) -> None:
    captured: dict[str, Any] = {}
    dummy = SimpleNamespace(
        embedding=np.zeros((1, 3), dtype=np.float32),
        tokenizer=Tokenizer(WordLevel({"test": 0})),
    )

    def fake_distill(
        model_name: str,
        vocabulary: list[str] | None = None,
        device: str | None = None,
        pca_dims: int | None = 256,
        apply_zipf: bool = True,
        use_subword: bool = True,
        quantize_to: str = "float32",
        sif_coefficient: float | None = 1e-4,
        token_remove_pattern: str | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        captured["model_name"] = model_name
        captured["device"] = device
        return dummy

    distill_mod = SimpleNamespace(distill=fake_distill)
    monkeypatch.setitem(sys.modules, "model2vec.distill", distill_mod)

    StaticEmbedding.from_distillation("dummy-teacher", **device_kwargs)
    assert captured == {"model_name": "dummy-teacher", "device": device_kwargs.get("device")}


@skip_if_no_model2vec()
def test_from_model2vec() -> None:
    model = StaticEmbedding.from_model2vec("minishlab/M2V_base_output")
    assert model.embedding.weight.shape == (29528, 256)


def test_load_model2vec_mapping_and_weights(tmp_path: Path) -> None:
    # model2vec stores vocabulary-quantized / weighted models as "embeddings" plus a per-token "mapping" into
    # those rows and per-token "weights", and averages embeddings[mapping[token_id]] * weights[token_id].
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "foo": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mapping = torch.tensor([0, 2, 1, 2])
    weights = torch.tensor([1.0, 0.5, 2.0, 4.0], dtype=torch.float64)
    save_file({"embeddings": embeddings, "mapping": mapping, "weights": weights}, str(tmp_path / "model.safetensors"))

    static_embedding = StaticEmbedding.load(str(tmp_path))
    expected = (embeddings[mapping] * weights[:, None]).float()
    assert torch.equal(static_embedding.embedding.weight.data, expected)

    texts = ["hello world", "foo"]
    output = static_embedding(static_embedding.preprocess(texts))
    assert torch.allclose(output["sentence_embedding"], torch.tensor([[0.25, 1.25], [4.0, 4.0]]))

    static_embedding.save(str(tmp_path))
    reloaded = StaticEmbedding.load(str(tmp_path))
    reloaded_output = reloaded(reloaded.preprocess(texts))
    torch.testing.assert_close(reloaded_output["sentence_embedding"], output["sentence_embedding"], rtol=0, atol=0)


@pytest.mark.parametrize("embedding_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.float32])
def test_load_model2vec_small_weights(tmp_path: Path, embedding_dtype: torch.dtype, weight_dtype: torch.dtype) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    embeddings = torch.tensor([[0.0, 0.0], [1e-4, 2e-4]], dtype=embedding_dtype)
    weights = torch.tensor([1.0, 1e-4], dtype=weight_dtype)
    save_file({"embeddings": embeddings, "weights": weights}, str(tmp_path / "model.safetensors"))

    model = StaticEmbedding.load(str(tmp_path))
    output = model(model.preprocess(["hello"]))["sentence_embedding"]
    expected_dtype = torch.float64 if embedding_dtype == torch.float64 else torch.float32
    expected = (embeddings[1:].double() * weights[1].double()).to(expected_dtype)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=0)


@skip_if_no_model2vec()
def test_from_model2vec_mapping_and_weights(tmp_path: Path) -> None:
    import numpy as np
    from model2vec import StaticModel

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "foo": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    static_model = StaticModel(
        vectors=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        tokenizer=tokenizer,
        weights=np.array([1.0, 0.5, 2.0, 4.0]),
        token_mapping=np.array([0, 2, 1, 2]),
    )
    static_model.save_pretrained(str(tmp_path))

    model = SentenceTransformer(modules=[StaticEmbedding.from_model2vec(str(tmp_path))], device="cpu")
    texts = ["hello world", "foo", "world foo hello"]
    expected = np.stack([StaticModel.from_pretrained(str(tmp_path)).encode(text) for text in texts])
    assert np.allclose(model.encode(texts), expected)


def test_unsupported_modality(static_embedding: StaticEmbedding) -> None:
    from PIL import Image

    model = SentenceTransformer(modules=[static_embedding])
    dummy_image = Image.new("RGB", (10, 10))

    # Image-only input
    with pytest.raises(
        ValueError,
        match="Modality 'image' is not supported by this SentenceTransformer model. Supported modalities: text",
    ):
        model.encode([dummy_image])

    # Mixed text+image input via multimodal dict
    with pytest.raises(
        ValueError,
        match="Modality 'image\\+text' is not supported by this SentenceTransformer model. Supported modalities: text",
    ):
        model.encode([{"text": "a cat", "image": dummy_image}])

    # Mixed-modality batch (a multimodal dict alongside a text-only dict) is inferred as 'message'.
    # The model genuinely cannot handle image, so it is named as the unsupported modality.
    with pytest.raises(
        ValueError,
        match=(
            r"This batch mixes multiple modalities \(image\+text, text\), but this SentenceTransformer "
            r"model does not support image\. Supported modalities: text"
        ),
    ):
        model.encode([{"text": "a cat", "image": dummy_image}, {"text": "a dog"}])


def test_loading_model2vec() -> None:
    model = SentenceTransformer("minishlab/potion-base-8M")
    assert model.get_embedding_dimension() == 256
    assert model.max_seq_length == math.inf

    test_sentences = ["It's so sunny outside!", "The sun is shining outside!"]
    embeddings = model.encode(test_sentences)
    assert embeddings.shape == (2, 256)
    similarity = model.similarity(embeddings[0], embeddings[1])
    assert similarity.item() > 0.7
