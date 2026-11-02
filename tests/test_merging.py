from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Dense, Router, Transformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.sparse_encoder.modules import MLMTransformer

pytest.importorskip("mergekit")
if Version(version("mergekit")) < Version("0.1.5"):
    pytest.skip("Model merging requires mergekit>=0.1.5", allow_module_level=True)
merge_methods = pytest.importorskip("mergekit.merge_methods")


@pytest.fixture
def model():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    model = SentenceTransformer(
        modules=[
            StaticEmbedding(tokenizer, embedding_weights=torch.ones(3, 2)),
            Dense(2, 2, bias=False, activation_function=None, init_weight=torch.ones(2, 2)),
        ],
        device="cpu",
        prompts={"query": "hello "},
    )
    model.model_card_data.generate_widget_examples = False
    return model


def test_linear_merge_and_roundtrip(model, tmp_path):
    other = deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.fill_(3)
    model._model_card_text = "This is the first parent's model card."
    merged = SentenceTransformer.merge([model, other], parameters={"weight": [1, 3]})
    for name, value in merged.state_dict().items():
        torch.testing.assert_close(value, torch.full_like(value, 2.5))
        torch.testing.assert_close(model.state_dict()[name], torch.ones_like(value))
        torch.testing.assert_close(other.state_dict()[name], torch.full_like(value, 3))
    assert merged.prompts == model.prompts
    assert not merged.training
    assert model._model_card_text is not None
    merged.save_pretrained(str(tmp_path))
    assert model._model_card_text not in (tmp_path / "README.md").read_text(encoding="utf-8")
    reloaded = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)
    torch.testing.assert_close(merged.encode(["hello world"]), reloaded.encode(["hello world"]))


def test_loading_paths_and_separate_base(model, tmp_path):
    paths = [tmp_path / name for name in ("base", "a", "b")]
    for path, value in zip(paths, (1, 2, 4)):
        copy = deepcopy(model)
        with torch.no_grad():
            for parameter in copy.parameters():
                parameter.fill_(value)
        copy.save_pretrained(str(path), create_model_card=False)
    merged = SentenceTransformer.merge(
        paths[1:], base_model=paths[0], method="task_arithmetic", parameters={"weight": [2, -1]}, local_files_only=True
    )
    for value in merged.state_dict().values():
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_slerp_merges_every_module(model):
    other = deepcopy(model)
    with torch.no_grad():
        for a, b in zip(model.parameters(), other.parameters()):
            a[:, 0], a[:, 1] = 1, 0
            b[:, 0], b[:, 1] = 0, 1
    merged = SentenceTransformer.merge([model, other], method="slerp")
    for value in merged.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 2**-0.5))


def test_explicit_slerp_base_and_interpolation(model):
    other = deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.fill_(3)
    merged = SentenceTransformer.merge([model, other], method="slerp", base_model=other, parameters={"t": 1})
    for name, value in merged.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name])


def test_mergekit_normalization_defaults(model):
    base, other = deepcopy(model), deepcopy(model)
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.zero_()
        for parameter in other.parameters():
            parameter.fill_(3)
    ties = SentenceTransformer.merge([model, other], method="ties", base_model=base)
    arithmetic = SentenceTransformer.merge([model, other], method="task_arithmetic", base_model=base)
    for value in ties.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 2))
    for value in arithmetic.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 4))


def test_tied_weights_merged_once(model):
    model[1] = Dense(2, 3, bias=False, activation_function=None)
    model[1].linear.weight = model[0].embedding.weight
    base, other = deepcopy(model), deepcopy(model)
    with torch.no_grad():
        base[0].embedding.weight.zero_()
        other[0].embedding.weight.fill_(2)
    torch.manual_seed(42)
    expected = merge_methods.merge_state_dicts(
        [model[0], other[0], base[0]], "dare_linear", base=2, parameters={"weight": 1, "density": 0.5}
    )["embedding.weight"]
    torch.manual_seed(42)
    merged = SentenceTransformer.merge(
        [model, other], method="dare_linear", base_model=base, parameters={"density": 0.5}, out_dtype="float64"
    )
    assert merged[0].embedding.weight is merged[1].linear.weight
    torch.testing.assert_close(merged[0].embedding.weight, expected.double())
    assert model[0].embedding.weight is model[1].linear.weight
    assert model[0].embedding.weight.dtype == torch.float32


def test_output_owns_tensors_when_method_returns_inputs(model, monkeypatch):
    @merge_methods.merge_method(name="select_first")
    def select_first(group: merge_methods.TensorGroup) -> torch.Tensor:
        return group.entries[0].tensor

    monkeypatch.setattr(merge_methods, "get", lambda _: select_first)
    model[0].embedding.weight.requires_grad_(False)
    model.register_buffer("running", torch.ones(2, requires_grad=True))
    model.register_buffer("running_alias", model.running)
    model.register_buffer("cache", torch.ones(2), persistent=False)
    other = deepcopy(model)

    merged = SentenceTransformer.merge([model, other], method="select_first")
    assert not merged[0].embedding.weight.requires_grad
    assert merged[1].linear.weight.requires_grad
    assert merged.running.requires_grad
    assert merged.running is merged.running_alias
    assert merged.running is not model.running
    with torch.no_grad():
        for tensor in (*merged.parameters(), *merged.buffers()):
            tensor.zero_()
    for parent in (model, other):
        for tensor in (*parent.parameters(), *parent.buffers()):
            torch.testing.assert_close(tensor, torch.ones_like(tensor))


@pytest.mark.parametrize("dtype", ["float16", torch.bfloat16, "float32", torch.float64])
def test_output_dtype_and_buffers(model, dtype):
    model.register_buffer("counter", torch.tensor(1))
    merged = SentenceTransformer.merge([model, model], out_dtype=dtype)
    assert merged[0].embedding.weight.dtype == (getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    assert merged.counter.dtype == torch.int64
    assert model[0].embedding.weight.dtype == torch.float32


def test_input_precision_and_output_cast(model):
    other = deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.fill_(1.0001)
    parameters = {"weight": [-1, 1], "normalize": False}

    merged = SentenceTransformer.merge([model, other], parameters=parameters, out_dtype=torch.float16)
    rounded = SentenceTransformer.merge([model, other], parameters=parameters, dtype="float16")
    for name, value in merged.state_dict().items():
        expected = (other.state_dict()[name] - model.state_dict()[name]).half()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert torch.count_nonzero(value) == value.numel()
        torch.testing.assert_close(rounded.state_dict()[name], torch.zeros_like(value))


def test_dtype_promotion_per_tensor(model):
    other = deepcopy(model)
    model[0].half()
    other[0].bfloat16()
    other[1].double()
    model.register_buffer("running", torch.ones(2, dtype=torch.float16))
    other.register_buffer("running", torch.full((2,), 3, dtype=torch.bfloat16))
    model.register_buffer("counter", torch.tensor(1))
    other.register_buffer("counter", torch.tensor(1))

    merged = SentenceTransformer.merge([model, other], dtype=None)
    assert merged[0].embedding.weight.dtype == torch.float32
    assert merged[1].linear.weight.dtype == torch.float64
    torch.testing.assert_close(merged.running, torch.full((2,), 2.0))
    assert merged.counter.dtype == torch.int64

    with torch.no_grad():
        for tensor in merged.state_dict().values():
            tensor.zero_()
    assert model[0].embedding.weight.dtype == torch.float16
    assert other[0].embedding.weight.dtype == torch.bfloat16
    assert model[1].linear.weight.dtype == torch.float32
    for parent in (model, other):
        for tensor in parent.state_dict().values():
            assert torch.count_nonzero(tensor) == tensor.numel()


@pytest.mark.parametrize(
    "batch_options,expected_batch_sizes",
    [
        (None, [1, 2]),
        (merge_methods.BatchOptions(max_groups=1), [1, 1, 1]),
        (merge_methods.BatchOptions(max_bytes=1), [1, 1, 1]),
    ],
)
def test_batch_options(model, batch_options, expected_batch_sizes, monkeypatch):
    model.append(Dense(2, 2, bias=False, activation_function=None, init_weight=torch.full((2, 2), 2.0)))
    other = deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(2)

    method = merge_methods.get("linear")
    merge_batch = method.merge_batch
    batch_sizes = []

    def record_batch(batch, **parameters):
        batch_sizes.append(batch.tensors[0].shape[0])
        return merge_batch(batch, **parameters)

    monkeypatch.setattr(method, "merge_batch", record_batch)
    merged = SentenceTransformer.merge([model, other], parameters={"weight": [1, 3]}, batch_options=batch_options)
    assert sorted(batch_sizes) == expected_batch_sizes
    for name, value in merged.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name] + 1.5)


@pytest.mark.parametrize("problem", ["config", "type", "keys", "shape", "tokenizer", "buffer", "ties"])
def test_incompatible_models(model, problem):
    other = deepcopy(model)
    if problem == "config":
        other[1].module_output_name = "different_embedding"
    elif problem == "type":
        other[1].activation_function = torch.nn.ReLU()
    elif problem == "keys":
        other.register_buffer("extra", torch.ones(2))
    elif problem == "shape":
        other[0].embedding.weight = torch.nn.Parameter(torch.ones(4, 2))
    elif problem == "tokenizer":
        other[0].tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "world": 1, "hello": 2}, unk_token="[UNK]"))
    elif problem == "buffer":
        model.register_buffer("counter", torch.tensor(1))
        other.register_buffer("counter", torch.tensor(2))
    else:
        model[1].linear.weight = model[0].embedding.weight
    with pytest.raises(ValueError):
        SentenceTransformer.merge([model, other])


@pytest.mark.parametrize(
    "activation,other_activation",
    [
        (torch.nn.LeakyReLU(0.1), torch.nn.LeakyReLU(0.9)),
        (torch.nn.GELU("none"), torch.nn.GELU("tanh")),
        (torch.nn.LayerNorm(2, eps=1e-5), torch.nn.LayerNorm(2, eps=1e-3)),
    ],
)
def test_incompatible_torch_module_settings(model, activation, other_activation):
    model[1].activation_function = activation
    other = deepcopy(model)
    other[1].activation_function = other_activation

    with pytest.raises(ValueError, match="different configuration"):
        SentenceTransformer.merge([model, other], parameters={"weight": [0, 1]})


@pytest.mark.parametrize(
    "activation",
    [torch.nn.LeakyReLU(0.1), torch.nn.GELU("tanh"), torch.nn.LayerNorm(2, eps=1e-3)],
)
def test_matching_torch_module_settings(model, activation):
    model[1].activation_function = activation
    merged = SentenceTransformer.merge([model, deepcopy(model)])
    torch.testing.assert_close(merged.encode(["hello world"]), model.encode(["hello world"]))


@pytest.mark.parametrize("lora_alpha", [1, 10])
def test_peft_adapters_rejected(stsb_bert_tiny_model, lora_alpha):
    peft = pytest.importorskip("peft")
    model = stsb_bert_tiny_model
    other = deepcopy(model)
    model.add_adapter(peft.LoraConfig(r=1, lora_alpha=1, target_modules=["query"]))
    other.add_adapter(peft.LoraConfig(r=1, lora_alpha=lora_alpha, target_modules=["query"]))

    with pytest.raises(ValueError, match="PEFT adapters"):
        SentenceTransformer.merge([model, other], parameters={"weight": [0, 1]})


@pytest.mark.parametrize("position", ["first", "second", "base"])
def test_peft_adapter_on_single_input_or_base_rejected(stsb_bert_tiny_model, position):
    peft = pytest.importorskip("peft")
    model = stsb_bert_tiny_model
    adapted = deepcopy(model)
    adapted.add_adapter(peft.LoraConfig(r=1, lora_alpha=1, target_modules=["query"]))
    models = [adapted, model] if position == "first" else [model, adapted]
    kwargs = {}
    if position == "base":
        models = [model, model]
        kwargs = {"method": "task_arithmetic", "base_model": adapted}

    with pytest.raises(ValueError, match="PEFT adapters"):
        SentenceTransformer.merge(models, **kwargs)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"parameters": {"weight": [1]}}, ValueError),
        ({"dtype": "int8"}, ValueError),
        ({"dtype": "unknown"}, ValueError),
        ({"out_dtype": torch.int64}, ValueError),
        ({"out_dtype": "unknown"}, ValueError),
        ({"method": "ties"}, ValueError),
        ({"parameters": {"typo": True}}, TypeError),
    ],
)
def test_invalid_arguments(model, kwargs, exception):
    with pytest.raises(exception):
        SentenceTransformer.merge([model, model], **kwargs)


def test_invalid_base_and_arity(model):
    with pytest.raises(ValueError, match="does not use a base_model"):
        SentenceTransformer.merge([model, model], base_model=model)
    with pytest.raises(ValueError, match="At least two"):
        SentenceTransformer.merge([model])
    with pytest.raises(ValueError):
        SentenceTransformer.merge([model, model, model], method="slerp")


def test_old_mergekit_error(model, monkeypatch):
    monkeypatch.setattr("sentence_transformers.base.merging.version", lambda _: "0.1.4")
    with pytest.raises(ImportError, match="requires mergekit>=0.1.5"):
        SentenceTransformer.merge([model, model])


def test_router(model, tmp_path):
    routed = SentenceTransformer(
        modules=[Router.for_query_document(query_modules=[deepcopy(model[0])], document_modules=[deepcopy(model[0])])],
        device="cpu",
    )
    merged = SentenceTransformer.merge([routed, routed])
    merged.save_pretrained(str(tmp_path))
    reloaded = SentenceTransformer(str(tmp_path), device="cpu", local_files_only=True)
    torch.testing.assert_close(merged.encode_query(["hello"]), reloaded.encode_query(["hello"]))
    torch.testing.assert_close(routed.encode_document(["world"]), reloaded.encode_document(["world"]))


@pytest.mark.parametrize("evaluated_first", [True, False])
@pytest.mark.parametrize("dimension", [None, 1])
def test_splade_pooling_dimension_metadata(splade_bert_tiny_model, evaluated_first, dimension):
    evaluated = splade_bert_tiny_model.cpu()
    evaluated[1].embedding_dimension = None
    fresh = deepcopy(evaluated)
    fresh[1].embedding_dimension = dimension
    expected = evaluated.encode(["hello world"], convert_to_tensor=True)
    inputs = [evaluated, fresh] if evaluated_first else [fresh, evaluated]

    merged = SparseEncoder.merge(inputs)
    actual = merged.encode(["hello world"], convert_to_tensor=True)
    torch.testing.assert_close(actual.to_dense(), expected.to_dense())
    assert fresh[1].embedding_dimension == dimension
    assert evaluated[1].embedding_dimension == expected.shape[-1]


@pytest.mark.parametrize("problem", ["pooling", "activation"])
def test_splade_configuration_mismatch(splade_bert_tiny_model, problem):
    unknown = splade_bert_tiny_model.cpu()
    unknown[1].embedding_dimension = None
    other = deepcopy(unknown)
    other[1].embedding_dimension = other[0].model.config.vocab_size
    if problem == "pooling":
        other[1].pooling_strategy = "sum" if unknown[1].pooling_strategy == "max" else "max"
    else:
        other[1].activation_function = "log1p_relu" if unknown[1].activation_function == "relu" else "relu"

    with pytest.raises(ValueError, match="different configuration"):
        SparseEncoder.merge([unknown, other])
    assert unknown[1].embedding_dimension is None


@pytest.mark.parametrize("legacy_first", [True, False])
@pytest.mark.parametrize("method", ["linear", "task_arithmetic"])
def test_legacy_mlm_transformer(splade_bert_tiny_model, tmp_path, legacy_first, method):
    model = splade_bert_tiny_model.cpu()
    model[0].model.save_pretrained(tmp_path)
    model[0].tokenizer.save_pretrained(tmp_path)
    wrappers = [MLMTransformer, Transformer] if legacy_first else [Transformer, MLMTransformer]
    inputs = [
        SparseEncoder(
            modules=[wrapper(str(tmp_path), transformer_task="fill-mask"), deepcopy(model[1])],
            device="cpu",
        )
        for wrapper in wrappers
    ]
    kwargs = {"base_model": deepcopy(inputs[1])} if method == "task_arithmetic" else {}
    merged = SparseEncoder.merge(inputs, method=method, **kwargs)
    assert type(merged[0]) is wrappers[0]
    for name, value in merged.state_dict().items():
        torch.testing.assert_close(value, inputs[0].state_dict()[name])
    torch.testing.assert_close(
        merged.encode(["hello world"], convert_to_tensor=True).to_dense(),
        inputs[0].encode(["hello world"], convert_to_tensor=True).to_dense(),
    )

    inputs[1][0].transformer_task = "feature-extraction"
    with pytest.raises(ValueError, match="different configuration"):
        SparseEncoder.merge(inputs)


@pytest.mark.parametrize(
    "fixture_name",
    ["stsb_bert_tiny_model", "splade_bert_tiny_model", "reranker_bert_tiny_model", "mve_bert_tiny_model"],
)
@pytest.mark.parametrize("method", ["linear", "slerp", "task_arithmetic", "ties", "dare_linear", "nuslerp"])
def test_transformer_models(request, fixture_name, method, tmp_path):
    model = request.getfixturevalue(fixture_name).cpu()
    other = deepcopy(model)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(0.01)
    kwargs = {"base_model": model} if method in {"task_arithmetic", "ties", "dare_linear"} else {}
    merged = type(model).merge([model, other], method=method, **kwargs)
    merged.save_pretrained(str(tmp_path))
    reloaded = type(model)(str(tmp_path), device="cpu", local_files_only=True)
    assert merged.state_dict().keys() == model.state_dict().keys()
    for name, value in merged.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value)
        assert torch.isfinite(value).all()
    if isinstance(model, CrossEncoder):
        output = reloaded.predict([("hello", "world")], convert_to_tensor=True)
    elif isinstance(model, MultiVectorEncoder):
        output = reloaded.encode(["hello world"])[0]
    else:
        output = reloaded.encode(["hello world"], convert_to_tensor=True)
    if isinstance(model, SparseEncoder):
        output = output.to_dense()
    assert torch.isfinite(output).all()


def test_ties_single_element_bias():
    merged = merge_methods.merge_state_dicts(
        [{"bias": torch.tensor([0.0])}, {"bias": torch.tensor([1.0])}],
        "ties",
        base=0,
        parameters={"weight": 1.0, "density": 0.5},
    )
    torch.testing.assert_close(merged["bias"], torch.tensor([1.0]))
