from __future__ import annotations

import pickle
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel, KernelConfig, PreTrainedModel

from sentence_transformers import CrossEncoder, MultiVectorEncoder, SentenceTransformer, SparseEncoder
from sentence_transformers.base.modules import Router
from sentence_transformers.multi_vector_encoder.losses import (
    CachedMultiVectorMultipleNegativesRankingLoss,
    MultiVectorDistillKLDivLoss,
    MultiVectorMarginMSELoss,
    MultiVectorMultipleNegativesRankingLoss,
    MultiVectorTokenPoolingLoss,
)
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
from sentence_transformers.multi_vector_encoder.scoring import (
    colbert_kd_scores,
    colbert_scores,
    colbert_scores_pairwise,
    mean_colbert_kd_scores,
    mean_colbert_scores,
    mean_colbert_scores_pairwise,
)
from sentence_transformers.multi_vector_encoder.scoring.colbert import (
    _colbert_kd_scores,
    _colbert_scores,
    _colbert_scores_pairwise,
    _MaxSimScorer,
)
from sentence_transformers.sentence_transformer.modules import Dense, Normalize
from sentence_transformers.util import maxsim, maxsim_pairwise
from sentence_transformers.util.similarity import _maxsim_score_documents, _maxsim_score_pairs

pytest.importorskip("kernels", minversion="0.17.2")


class ScaledMaxSim(nn.Module):
    can_torch_compile = True

    def forward(self, a, b, a_mask, b_mask):
        return 2 * _maxsim_score_documents(a, b, a_mask, b_mask)


class ScaledMaxSimPairwise(nn.Module):
    can_torch_compile = True

    def forward(self, a, b, a_mask, b_mask):
        return 2 * _maxsim_score_pairs(a, b, a_mask, b_mask)


class InferenceMaxSim(nn.Module):
    has_backward = False

    def forward(self, a, b, a_mask, b_mask):
        return 3 * _maxsim_score_documents(a, b, a_mask, b_mask)


class ReferenceGelu(nn.Module):
    can_torch_compile = True

    def forward(self, hidden_states):
        return torch.nn.functional.gelu(hidden_states)


class SingletonWard(nn.Module):
    def forward(self, distances, lengths, cluster_counts):
        return torch.zeros_like(distances[..., 0], dtype=torch.int64), (lengths > 0).long()


class InferenceWard(SingletonWard):
    has_backward = False


@pytest.fixture
def loader(monkeypatch):
    import kernels.layer.layer as layer

    calls = []
    module = SimpleNamespace(
        layers=SimpleNamespace(
            ScaledMaxSim=ScaledMaxSim,
            ScaledMaxSimPairwise=ScaledMaxSimPairwise,
            InferenceMaxSim=InferenceMaxSim,
            SingletonWard=SingletonWard,
            InferenceWard=InferenceWard,
            MaxSim=ScaledMaxSim,
            MaxSimPairwise=ScaledMaxSimPairwise,
            WardClustering=SingletonWard,
            Gelu=ReferenceGelu,
        )
    )
    monkeypatch.setattr(layer, "get_kernel", lambda *args, **kwargs: calls.append((args, kwargs)) or module)
    monkeypatch.setattr(layer.LayerRepository, "_resolve_revision", lambda self: "test")
    monkeypatch.setattr(layer, "_CACHED_LAYER", {})
    return calls


def config(**mapping):
    return KernelConfig({key: f"test/kernel:{value}" for key, value in mapping.items()}, inherit_mapping=False)


def scorer():
    return _MaxSimScorer(scorer_config())


def scorer_config():
    return config(STMaxSim="ScaledMaxSim", STMaxSimPairwise="ScaledMaxSimPairwise")


@pytest.fixture
def embeddings():
    torch.manual_seed(17)
    return torch.randn(3, 4, 8), torch.randn(3, 6, 8)


@pytest.mark.parametrize("normalize", [False, True])
def test_maxsim_isolation_masks_and_chunking(loader, embeddings, normalize):
    a, b = embeddings
    a_mask = torch.tensor([[1, 0, 1, 1]]).expand(3, -1)
    b_mask = torch.tensor([[0, 1, 1, 0, 1, 0]]).expand(3, -1)
    selected = scorer()
    selected.length_normalize = normalize
    reference = maxsim(a, b, a_mask, b_mask, length_normalize=normalize)
    torch.testing.assert_close(selected(a, b, a_mask, b_mask, chunk_elements=1), 2 * reference)
    torch.testing.assert_close(_MaxSimScorer()(a, b, a_mask, b_mask, length_normalize=normalize), reference)
    reference_pairs = maxsim_pairwise(a, b, a_mask, b_mask, length_normalize=normalize)
    torch.testing.assert_close(selected.pairwise(a, b, a_mask, b_mask, chunk_elements=1), 2 * reference_pairs)
    assert len(loader) == 2


def test_maxsim_ragged_and_empty_inputs(loader, embeddings):
    a, b = embeddings
    a, b = [a[0, :1], a[1]], [b[0, :2], b[1], b[2, :0]]
    selected = scorer()
    torch.testing.assert_close(selected(a, b, chunk_elements=1), 2 * maxsim(a, b))
    torch.testing.assert_close(selected([], b), maxsim([], b))
    torch.testing.assert_close(selected.pairwise([], []), maxsim_pairwise([], []))


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("pairwise", [False, True])
def test_maxsim_generated_masks_support_boolean_mask_providers(loader, monkeypatch, embeddings, padded, pairwise):
    operation = ScaledMaxSimPairwise if pairwise else ScaledMaxSim
    reference = _maxsim_score_pairs if pairwise else _maxsim_score_documents

    def forward(self, a, b, a_mask, b_mask):
        assert a_mask.dtype == b_mask.dtype == torch.bool
        return 2 * reference(a, b, a_mask, b_mask)

    monkeypatch.setattr(operation, "forward", forward)
    a, b = embeddings
    if not padded:
        a, b = [a[0, :1], a[1]], [b[0, :2], b[1]]
    selected = scorer()
    score = selected.pairwise if pairwise else selected
    expected = maxsim_pairwise if pairwise else maxsim
    torch.testing.assert_close(score(a, b), 2 * expected(a, b))
    assert loader


@pytest.mark.parametrize("preferences", [None, KernelConfig(), KernelConfig(inherit_mapping=False)])
def test_model_keeps_reference_scoring_without_matching_kernels(loader, embeddings, preferences):
    model = MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config=preferences)
    torch.testing.assert_close(model.similarity(*embeddings), maxsim(*embeddings))
    torch.testing.assert_close(model.similarity_pairwise(*embeddings), maxsim_pairwise(*embeddings))
    assert not loader


def test_model_kernel_config_is_copied_and_local(loader, embeddings):
    preferences = scorer_config()
    configured = MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config=preferences)
    preferences.kernel_mapping.clear()
    reference = MultiVectorEncoder(modules=[Normalize()], device="cpu")
    torch.testing.assert_close(configured.similarity(*embeddings), 2 * maxsim(*embeddings))
    torch.testing.assert_close(reference.similarity(*embeddings), maxsim(*embeddings))


def test_model_rejects_invalid_kernel_config():
    with pytest.raises(TypeError, match="kernel_config must be a transformers.KernelConfig"):
        MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config={})


def test_model_binding_and_reference_checkpoint(loader, embeddings, tmp_path):
    module = Dense(8, 8, module_input_name="token_embeddings")
    parameters = dict(module.named_parameters())
    model = MultiVectorEncoder(
        modules=[module], device="cpu", similarity_fn_name="meanmaxsim", kernel_config=scorer_config()
    )
    assert not loader
    a, b = embeddings
    torch.testing.assert_close(model.similarity(a, b), 2 * maxsim(a, b) / a.shape[1])
    torch.testing.assert_close(model.similarity_pairwise(a, b), 2 * maxsim_pairwise(a, b) / a.shape[1])
    assert len(model) == 1
    assert list(model.state_dict()) == [f"0.{name}" for name in module.state_dict()]
    assert all(parameter is dict(model.named_parameters())[f"0.{name}"] for name, parameter in parameters.items())
    assert not hasattr(model, "_kernel_scope")
    assert not hasattr(model, "set_kernel_config")
    model.save_pretrained(str(tmp_path), create_model_card=False)
    restored = MultiVectorEncoder(str(tmp_path), device="cpu")
    torch.testing.assert_close(restored.similarity(a, b), maxsim(a, b) / a.shape[1])
    torch.testing.assert_close(restored.state_dict(), model.state_dict())
    configured = MultiVectorEncoder(str(tmp_path), device="cpu", kernel_config=scorer_config())
    torch.testing.assert_close(configured.similarity(a, b), 2 * maxsim(a, b) / a.shape[1])
    model.similarity_fn_name = "maxsim"
    torch.testing.assert_close(model.similarity(a, b), maxsim(a, b))


@pytest.mark.parametrize("round_trip", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_scorer_and_bound_model_round_trip(loader, embeddings, round_trip):
    selected = scorer()
    selected(*embeddings)
    torch.testing.assert_close(round_trip(selected)(*embeddings), 2 * maxsim(*embeddings))
    model = MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config=scorer_config())
    restored = round_trip(model)
    torch.testing.assert_close(restored.similarity(*embeddings), 2 * maxsim(*embeddings))
    torch.testing.assert_close(restored.similarity_pairwise(*embeddings), 2 * maxsim_pairwise(*embeddings))


@pytest.mark.parametrize(
    "loss_class,reference_scorer",
    [
        (MultiVectorMultipleNegativesRankingLoss, colbert_scores),
        (CachedMultiVectorMultipleNegativesRankingLoss, colbert_scores),
        (MultiVectorDistillKLDivLoss, colbert_kd_scores),
        (MultiVectorMarginMSELoss, colbert_scores_pairwise),
    ],
)
@pytest.mark.parametrize("normalize", [False, True])
def test_losses_follow_model_scoring_and_allow_overrides(loader, embeddings, loss_class, reference_scorer, normalize):
    a, b = embeddings
    model = MultiVectorEncoder(
        modules=[Normalize()],
        device="cpu",
        similarity_fn_name="meanmaxsim" if normalize else "maxsim",
        kernel_config=scorer_config(),
    )
    features = [
        {"token_embeddings": value, "attention_mask": torch.ones(value.shape[:2], dtype=torch.bool)}
        for value in (a, b, b.flip(0))
    ]
    features[0]["attention_mask"][:, -1] = False
    features[1]["attention_mask"][:, :2] = False
    labels = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    if loss_class is MultiVectorMarginMSELoss:
        labels = labels[:, 0] - labels[:, 1]

    loss = loss_class(model)
    reference = loss_class(model, similarity_fct=partial(reference_scorer, length_normalize=normalize))
    before = reference(deepcopy(features), labels).detach()
    assert loss.get_config_dict()["similarity_fct"] is None

    def expected_scoring(*args, **kwargs):
        return 2 * reference_scorer(*args, **kwargs, length_normalize=normalize)

    expected = loss_class(model, similarity_fct=expected_scoring)(deepcopy(features), labels)
    torch.testing.assert_close(loss(deepcopy(features), labels), expected)
    torch.testing.assert_close(reference(deepcopy(features), labels), before)
    assert not torch.isclose(before, expected)

    model.similarity_fn_name = "maxsim"
    summed = loss_class(model, similarity_fct=reference_scorer)(deepcopy(features), labels)
    torch.testing.assert_close(loss(deepcopy(features), labels), summed)


@pytest.mark.parametrize(
    "loss_class,reference_scorer",
    [
        (MultiVectorMultipleNegativesRankingLoss, colbert_scores),
        (CachedMultiVectorMultipleNegativesRankingLoss, colbert_scores),
        (MultiVectorDistillKLDivLoss, colbert_kd_scores),
        (MultiVectorMarginMSELoss, colbert_scores_pairwise),
    ],
)
def test_loss_gradients_and_cached_replay(loader, embeddings, loss_class, reference_scorer):
    model = MultiVectorEncoder(
        modules=[Dense(8, 8, module_input_name="token_embeddings")], device="cpu", kernel_config=scorer_config()
    )
    reference = deepcopy(model)
    features = [
        {"token_embeddings": value, "attention_mask": torch.ones(value.shape[:2], dtype=torch.bool)}
        for value in (*embeddings, embeddings[1].flip(0))
    ]
    labels = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    if loss_class is MultiVectorMarginMSELoss:
        labels = labels[:, 0] - labels[:, 1]

    def expected_scoring(*args, **kwargs):
        return 2 * reference_scorer(*args, **kwargs)

    reference_loss_class = (
        MultiVectorMultipleNegativesRankingLoss
        if loss_class is CachedMultiVectorMultipleNegativesRankingLoss
        else loss_class
    )
    expected = reference_loss_class(reference, similarity_fct=expected_scoring)(features, labels)
    loss = loss_class(model, mini_batch_size=2)
    actual = loss(features, labels)
    torch.testing.assert_close(actual, expected)
    expected.backward()
    actual.backward()
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()


@pytest.mark.parametrize(
    "adapter,mean_adapter,layout_adapter,pairwise",
    [
        (colbert_scores, mean_colbert_scores, _colbert_scores, False),
        (colbert_kd_scores, mean_colbert_kd_scores, _colbert_kd_scores, True),
        (colbert_scores_pairwise, mean_colbert_scores_pairwise, _colbert_scores_pairwise, True),
    ],
)
def test_shared_colbert_adapters_keep_masks_grouping_and_normalization(
    loader, embeddings, adapter, mean_adapter, layout_adapter, pairwise
):
    queries, documents = embeddings
    queries_mask = torch.tensor([[1, 0, 1, 1]]).expand(3, -1)
    documents_mask = torch.tensor([[1, 1, 0, 1, 0, 1]]).expand(3, -1)
    if adapter is not colbert_scores_pairwise:
        documents = torch.stack((documents, documents.flip(0)), dim=1)
        documents_mask = documents_mask[:, None].expand(-1, 2, -1)

    arguments = (queries, documents, queries_mask, documents_mask, 1)
    reference = adapter(*arguments)
    token_counts = queries_mask.sum(-1) if adapter is colbert_scores_pairwise else queries_mask.sum(-1, keepdim=True)
    selected = scorer()
    selected.length_normalize = True
    configured = partial(layout_adapter, selected.pairwise if pairwise else selected)
    normalized = 2 * reference / token_counts

    torch.testing.assert_close(configured(*arguments), normalized)
    torch.testing.assert_close(configured(*arguments), 2 * mean_adapter(*arguments))
    torch.testing.assert_close(configured(*arguments, False), 2 * reference)
    torch.testing.assert_close(adapter(*arguments), reference)

    with pytest.raises(TypeError, match="scorer"):
        adapter(*arguments, scorer=selected)

    selected.length_normalize = False
    torch.testing.assert_close(configured(*arguments), 2 * reference)


def test_shared_training_adapter_can_compile_and_round_trip(loader, embeddings):
    model = MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config=scorer_config())
    queries, documents = embeddings
    documents = torch.stack((documents, documents.flip(0)), dim=1)
    configured = partial(_colbert_scores, model.similarity)
    expected = 2 * colbert_scores(queries, documents)
    torch.testing.assert_close(configured(queries, documents), expected)

    for restored in (deepcopy(configured), pickle.loads(pickle.dumps(configured))):
        torch.testing.assert_close(restored(queries, documents), expected)

    compiled = torch.compile(configured, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(queries, documents), colbert_scores(queries, documents))


def test_autograd_mode_selects_inference_only_provider(loader, embeddings):
    model = MultiVectorEncoder(modules=[Normalize()], device="cpu", kernel_config=config(STMaxSim="InferenceMaxSim"))
    a, b = (value.clone().requires_grad_() for value in embeddings)
    torch.testing.assert_close(model.similarity(a, b), maxsim(a, b))
    with torch.no_grad():
        torch.testing.assert_close(model.similarity(a, b), 3 * maxsim(a, b))
    torch.testing.assert_close(model.similarity(a, b), maxsim(a, b))
    model.similarity(a, b).sum().backward()
    assert a.grad is not None and b.grad is not None


def test_compiled_scoring_uses_reference(loader, embeddings):
    selected = scorer()
    compiled = torch.compile(selected, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*embeddings), maxsim(*embeddings))
    assert not loader
    torch.testing.assert_close(selected(*embeddings), 2 * maxsim(*embeddings))


@pytest.mark.parametrize("provider", ["SingletonWard", "InferenceWard"])
def test_ward_batched_shapes_and_gradients(loader, provider):
    documents = [torch.randn(length, 8, requires_grad=True) for length in [0, 1, 2, 7] * 9]
    pooler = HierarchicalTokenPooling(2, normalize_embeddings=False, kernel_config=config(STWardClustering=provider))
    actual = pooler.pool(documents)
    expected = [
        torch.cat((value[:1], value[1:].mean(0, keepdim=True))) if len(value) > 1 else value for value in documents
    ]
    for pooled, reference in zip(actual, expected):
        torch.testing.assert_close(pooled, reference)
    for gradient in torch.autograd.grad(sum(value.sum() for value in actual), documents):
        torch.testing.assert_close(gradient, torch.ones_like(gradient))
    assert loader


def test_custom_ward_supports_inputs_beyond_default_provider_limit(loader):
    document = torch.ones(4098, 2, requires_grad=True)
    pooler = HierarchicalTokenPooling(
        2, normalize_embeddings=False, kernel_config=config(STWardClustering="SingletonWard")
    )
    actual = pooler.pool_one(document)
    expected = torch.cat((document[:1], document[1:].mean(0, keepdim=True)))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(document.grad, torch.ones_like(document))
    assert loader


@pytest.mark.parametrize("pooling_options", [{"max_tokens": 4}, {"distance_threshold": 100.0}])
def test_ward_pooling_controls_keep_reference_behavior(loader, monkeypatch, pooling_options):
    from sentence_transformers.multi_vector_encoder.modules.token_pooling import _WardClustering

    documents = [torch.randn(length, 8).requires_grad_() for length in [0, 1, 2, 7, 11]]
    reference_clustering = _WardClustering(pooling_options.get("distance_threshold"))
    requested_counts = []

    def forward(self, distances, lengths, cluster_counts):
        assert self.distance_threshold == pooling_options.get("distance_threshold")
        requested_counts.append(cluster_counts.tolist())
        return reference_clustering(distances, lengths, cluster_counts)

    monkeypatch.setattr(SingletonWard, "forward", forward)
    reference = HierarchicalTokenPooling(**pooling_options).pool(documents)
    pooler = HierarchicalTokenPooling(**pooling_options, kernel_config=config(STWardClustering="SingletonWard"))
    actual = pooler.pool(documents)
    for pooled, expected in zip(actual, reference):
        torch.testing.assert_close(pooled, expected)
    for gradient, expected_gradient in zip(
        torch.autograd.grad(sum(value.sum() for value in actual), documents, retain_graph=True),
        torch.autograd.grad(sum(value.sum() for value in reference), documents),
    ):
        torch.testing.assert_close(gradient, expected_gradient)
    assert loader
    if "max_tokens" in pooling_options:
        assert requested_counts == [[0, 0, 1, 3, 3]]
    for pooled, restored in zip(actual, deepcopy(pooler).pool(documents)):
        torch.testing.assert_close(pooled, restored)


@pytest.mark.parametrize("round_trip", [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_ward_round_trip_and_runtime_only_config(loader, tmp_path, round_trip):
    pooler = HierarchicalTokenPooling(
        2, normalize_embeddings=False, kernel_config=config(STWardClustering="SingletonWard")
    )
    document = torch.randn(8, 4)
    expected = torch.cat((document[:1], document[1:].mean(0, keepdim=True)))
    torch.testing.assert_close(pooler.pool_one(document), expected)
    restored = round_trip(pooler)
    torch.testing.assert_close(restored.pool_one(document), expected)
    pooler.save(str(tmp_path))
    plain = HierarchicalTokenPooling.load(str(tmp_path))
    assert plain.use_kernels is None
    assert plain.kernel_config is None
    assert "forward" not in plain.clustering.__dict__


def test_pipeline_and_per_call_pooling_keep_pooler_configuration(loader):
    pooler = HierarchicalTokenPooling(2, kernel_config=config(STWardClustering="SingletonWard"))
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")
    text = ["A document containing several words to pool together"]
    result = model.encode_document(text, token_pooling=pooler)
    assert result[0].shape[0] == 2
    model.add_module(str(len(model)), pooler)
    torch.testing.assert_close(model.encode_document(text)[0], result[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("options", [{"use_kernels": True}, {"kernel_config": KernelConfig()}])
def test_ward_default_provider_is_opt_in(loader, options):
    document = torch.randn(12, 8, device="cuda")
    reference = HierarchicalTokenPooling(2, normalize_embeddings=False)
    expected = reference.pool_one(document)
    assert not loader

    pooler = HierarchicalTokenPooling(2, normalize_embeddings=False, **options)
    assert pooler.pool_one(document).shape == (2, 8)
    assert any(args == ("sentence-transformers/ward-clustering",) for args, _ in loader)
    assert all(kwargs["trust_remote_code"] == ["sentence-transformers/ward-clustering"] for _, kwargs in loader)
    torch.testing.assert_close(reference.pool_one(document), expected)


@pytest.mark.parametrize("path", ["pipeline", "router", "default", "encode", "raw"])
@pytest.mark.parametrize("selection", ["inherit", "disabled", "defaults", "custom"])
def test_ward_model_inheritance_and_pooler_overrides(loader, path, selection):
    preferences = KernelConfig({"STWardClustering": "test/model:SingletonWard"}, inherit_mapping=False)
    options = {
        "inherit": {},
        "disabled": {"use_kernels": False},
        "defaults": {"use_kernels": True},
        "custom": {
            "use_kernels": False,
            "kernel_config": KernelConfig({"STWardClustering": "test/pooler:SingletonWard"}, inherit_mapping=False),
        },
    }[selection]
    pooler = HierarchicalTokenPooling(2, **options)
    modules = None
    if path in {"pipeline", "router"}:
        backbone = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")
        modules = [*backbone, pooler]
        if path == "router":
            modules = [Router({"document": modules}, default_route="document")]
    model = MultiVectorEncoder(
        None if modules else "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        modules=modules,
        device="cpu",
        kernel_config=preferences,
        token_pooling=pooler if path == "default" else None,
    )
    kwargs = {} if path in {"pipeline", "router", "default"} else {"token_pooling": pooler}
    text = ["A document containing many different words to cluster and pool together"]
    if path == "raw":
        result = model.encode_document(text, output_value=None, **kwargs)[0]["token_embeddings"]
    else:
        result = model.encode_document(text, **kwargs)[0]

    if selection in {"inherit", "custom"}:
        assert result.shape[0] == 2
        repository = "test/model" if selection == "inherit" else "test/pooler"
        assert [args for args, _ in loader] == [(repository,)]
        assert all(kwargs["trust_remote_code"] is False for _, kwargs in loader)
    else:
        assert result.shape[0] > 2
        assert not loader
    assert preferences.kernel_mapping == {"STWardClustering": "test/model:SingletonWard"}
    if selection == "inherit":
        assert pooler.use_kernels is True
        assert pooler.kernel_config.kernel_mapping == preferences.kernel_mapping
        assert pooler.kernel_config is not model._token_pooling_kernel_config
        assert "forward" in pooler.clustering.__dict__
    if path == "pipeline":
        assert model[-1] is pooler


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("selection", ["inherit", "disabled", "defaults", "custom"])
def test_loss_ward_inheritance_and_pooler_overrides(loader, cached, selection):
    preferences = KernelConfig({"STWardClustering": "test/model:SingletonWard"}, inherit_mapping=False)
    options = {
        "inherit": {},
        "disabled": {"use_kernels": False},
        "defaults": {"use_kernels": True},
        "custom": {"kernel_config": config(STWardClustering="SingletonWard")},
    }[selection]
    pooler = HierarchicalTokenPooling(2, **options)
    model = MultiVectorEncoder(
        modules=[Dense(8, 8, module_input_name="token_embeddings")], device="cpu", kernel_config=preferences
    )
    inner = (
        CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=1, show_progress_bar=False)
        if cached
        else MultiVectorMultipleNegativesRankingLoss(model)
    )
    loss = MultiVectorTokenPoolingLoss(model, inner, token_poolings=[None, pooler])
    features = [
        {"token_embeddings": torch.randn(2, tokens, 8), "attention_mask": torch.ones(2, tokens)} for tokens in (4, 8)
    ]
    value = loss(features, None)
    assert torch.isfinite(value)
    value.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    if selection in {"inherit", "custom"}:
        repository = "test/model" if selection == "inherit" else "test/kernel"
        assert [args for args, _ in loader] == [(repository,)]
        assert pooler.use_kernels is True
        assert pooler.kernel_config is not model._token_pooling_kernel_config
    else:
        assert not loader
        assert pooler.use_kernels is (selection == "defaults")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("path", ["pipeline", "encode", "raw"])
@pytest.mark.parametrize("model_enabled", [False, True])
@pytest.mark.parametrize("pooler_enabled", [None, False, True])
def test_ward_inherits_model_default_kernels(loader, path, model_enabled, pooler_enabled):
    pooler = HierarchicalTokenPooling(2, use_kernels=pooler_enabled)
    modules = None
    if path == "pipeline":
        backbone = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cuda")
        modules = [*backbone, pooler]
    model = MultiVectorEncoder(
        None if modules else "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        modules=modules,
        device="cuda",
        use_kernels=model_enabled,
    )
    kwargs = {} if path == "pipeline" else {"token_pooling": pooler}
    if path == "raw":
        kwargs["output_value"] = None
    result = model.encode_document(["A document with plenty of words to pool into clusters"], **kwargs)[0]
    if path == "raw":
        result = result["token_embeddings"]
    enabled = model_enabled if pooler_enabled is None else pooler_enabled
    assert (result.shape[0] == 2) == enabled
    assert any(args == ("sentence-transformers/ward-clustering",) for args, _ in loader) == enabled
    assert pooler.use_kernels is (True if enabled else pooler_enabled)


@pytest.mark.parametrize("path", ["pipeline", "encode"])
def test_ward_model_settings_configure_existing_pooler_once(loader, monkeypatch, path):
    import sentence_transformers.multi_vector_encoder.modules.token_pooling as token_pooling_module

    calls = []
    original = token_pooling_module._configure_kernel

    def configure(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(token_pooling_module, "_configure_kernel", configure)
    pooler = HierarchicalTokenPooling(2)
    clustering = pooler.clustering
    plain = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")
    modules = [*plain, pooler] if path == "pipeline" else [*plain]
    configured = MultiVectorEncoder(
        modules=modules, device="cpu", kernel_config=config(STWardClustering="SingletonWard")
    )
    text = ["A document containing many different words to cluster and pool together"]
    kwargs = {} if path == "pipeline" else {"token_pooling": pooler}
    assert configured.encode_document(text, **kwargs)[0].shape[0] == 2
    plain_result = plain.encode_document(text, token_pooling=pooler)[0]
    unpooled = plain.encode_document(text)[0]
    assert plain_result.shape[0] == 2
    torch.testing.assert_close(pooler.pool_one(unpooled), plain_result)
    assert configured.encode_document(text, **kwargs)[0].shape[0] == 2
    assert pooler.use_kernels is True
    assert pooler.clustering is clustering
    assert len(calls) == 1


def test_pipeline_ward_inheritance_preserves_training_and_checkpoint_behavior(loader, tmp_path):
    pooler = HierarchicalTokenPooling(2, normalize_embeddings=False)
    model = MultiVectorEncoder(
        modules=[Dense(8, 8, module_input_name="token_embeddings"), pooler],
        device="cpu",
        kernel_config=config(STWardClustering="InferenceWard"),
    )
    embeddings = torch.randn(2, 12, 8, requires_grad=True)

    def forward(selected):
        return selected(
            {"token_embeddings": embeddings, "attention_mask": torch.ones(2, 12, dtype=torch.bool)},
            task="document",
        )["token_embeddings"]

    result = forward(model)
    assert result.shape == (2, 2, 8)
    result[:, 1:].square().sum().backward()
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad[:, 1:].abs().sum() > 0
    assert model[0].linear.weight.grad.abs().sum() > 0

    model.save(str(tmp_path))
    plain = MultiVectorEncoder(str(tmp_path), device="cpu")
    assert plain[-1].use_kernels is None
    assert plain[-1].kernel_config is None
    assert forward(plain).shape[1] > 2
    configured = MultiVectorEncoder(str(tmp_path), device="cpu", kernel_config=model.kernel_config)
    torch.testing.assert_close(forward(configured), result)


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
def test_backbone_settings_pass_through_model_kwargs(loader, monkeypatch, embeddings, model_class):
    calls = []
    original = PreTrainedModel.set_use_kernels
    preferences = KernelConfig(inherit_mapping=False)

    def configure(module, use_kernels, kernel_config=None, **kwargs):
        if use_kernels:
            calls.append((use_kernels, deepcopy(kernel_config)))
        return original(module, use_kernels, kernel_config, **kwargs)

    monkeypatch.setattr(PreTrainedModel, "set_use_kernels", configure)
    options = {"kernel_config": scorer_config()} if model_class is MultiVectorEncoder else {}
    model = model_class(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        device="cpu",
        model_kwargs={"use_kernels": True, "kernel_config": preferences},
        **options,
    )
    assert calls
    assert all(enabled and selected.kernel_mapping == preferences.kernel_mapping for enabled, selected in calls)
    if model_class is MultiVectorEncoder:
        torch.testing.assert_close(model.similarity(*embeddings), 2 * maxsim(*embeddings))


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
def test_model_kwargs_can_disable_backbone_with_top_level_config(loader, embeddings, model_class):
    preferences = scorer_config() if model_class is MultiVectorEncoder else KernelConfig()
    options = {"use_kernels": False}
    model = model_class(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        device="cpu",
        kernel_config=preferences,
        model_kwargs=options,
    )
    assert model.use_kernels
    assert not model.transformers_model.use_kernels
    assert options == {"use_kernels": False}
    if model_class is MultiVectorEncoder:
        torch.testing.assert_close(model.similarity(*embeddings), 2 * maxsim(*embeddings))


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
@pytest.mark.parametrize("selection", ["disabled", "defaults", "config"])
def test_all_models_configure_existing_backbones(monkeypatch, model_class, selection):
    calls = []
    backbone = BertModel(
        BertConfig(hidden_size=8, intermediate_size=16, num_attention_heads=2, num_hidden_layers=1, vocab_size=32)
    )

    def configure(module, use_kernels, kernel_config=None):
        calls.append((module, use_kernels, kernel_config))
        if kernel_config is not None:
            kernel_config.kernel_mapping.clear()

    monkeypatch.setattr(PreTrainedModel, "set_use_kernels", configure)
    options = {}
    preferences = KernelConfig({"RMSNorm": "test/kernel:Norm"}, inherit_mapping=False)
    if selection == "defaults":
        options["use_kernels"] = True
    elif selection == "config":
        options["kernel_config"] = preferences

    model = model_class(modules=[backbone], device="cpu", **options)
    assert model.use_kernels is (selection != "disabled")
    assert len(calls) == (selection != "disabled")
    if calls:
        assert calls[0][0] is backbone
        assert calls[0][1] is True
    if selection == "config":
        assert model.kernel_config.kernel_mapping == preferences.kernel_mapping
        assert preferences.kernel_mapping == {"RMSNorm": "test/kernel:Norm"}
    else:
        assert model.kernel_config is None
    assert "use_kernels" not in model._get_model_config()
    assert "kernel_config" not in model._get_model_config()


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
def test_all_models_enable_loaded_backbone_defaults(loader, model_class):
    model = model_class("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu", use_kernels=True)
    assert model.use_kernels is True
    assert model.transformers_model.use_kernels is True
    assert model.kernel_config is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
@pytest.mark.parametrize("selection", ["defaults", "model_kwargs", "config", "model_kwargs_config"])
@pytest.mark.parametrize("device_map", [None, "cuda"])
def test_loaded_backbone_kernels_are_refreshed_after_device_placement(
    loader, monkeypatch, model_class, selection, device_map
):
    calls = []
    original = PreTrainedModel.set_use_kernels

    def configure(module, use_kernels, kernel_config=None, **kwargs):
        calls.append(module.device.type)
        return original(module, use_kernels, kernel_config, **kwargs)

    monkeypatch.setattr(PreTrainedModel, "set_use_kernels", configure)
    preferences = KernelConfig({"GeLU": {"cuda": "test/kernel:Gelu"}}, inherit_mapping=False)
    options = {
        "defaults": {"use_kernels": True},
        "model_kwargs": {"model_kwargs": {"use_kernels": True}},
        "config": {"kernel_config": preferences},
        "model_kwargs_config": {"model_kwargs": {"kernel_config": preferences}},
    }[selection]
    if device_map is not None:
        options["model_kwargs"] = {**options.get("model_kwargs", {}), "device_map": device_map}
    model = model_class("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cuda", **options)
    assert model.transformers_model.device.type == "cuda"
    assert calls[0] == ("cpu" if device_map is None else "cuda")
    assert calls[-1] == "cuda"
    activations = [
        module for module in model.transformers_model.modules() if getattr(module, "kernel_layer_name", None) == "GeLU"
    ]
    assert activations
    assert all(module.forward.__func__ is ReferenceGelu.forward for module in activations)
    inputs = torch.tensor([[1, 2, 3]], device="cuda")
    with torch.no_grad():
        assert torch.isfinite(model.transformers_model(input_ids=inputs)[0]).all()
    assert preferences.kernel_mapping == {"GeLU": {"cuda": "test/kernel:Gelu"}}


def test_loaded_model_kernel_settings_are_runtime_only(loader, tmp_path, embeddings):
    model = MultiVectorEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu", kernel_config=scorer_config()
    )
    torch.testing.assert_close(model.similarity(*embeddings), 2 * maxsim(*embeddings))
    model.save_pretrained(str(tmp_path), create_model_card=False)
    restored = MultiVectorEncoder(str(tmp_path), device="cpu")
    assert restored.use_kernels is False
    assert restored.kernel_config is None
    assert restored.transformers_model.use_kernels is False
    torch.testing.assert_close(restored.similarity(*embeddings), maxsim(*embeddings))
    torch.testing.assert_close(restored.state_dict(), model.state_dict())


def test_model_config_routes_backbone_maxsim_and_ward_without_mutating_inputs(loader, monkeypatch, embeddings):
    calls = []
    preferences = scorer_config()
    preferences.kernel_mapping["RMSNorm"] = "test/kernel:Norm"
    preferences.kernel_mapping["STWardClustering"] = "test/kernel:SingletonWard"
    original_mapping = deepcopy(preferences.kernel_mapping)
    model_kwargs = {"dtype": torch.float32}

    def configure(module, use_kernels, kernel_config=None):
        if use_kernels:
            calls.append((use_kernels, deepcopy(kernel_config)))

    monkeypatch.setattr(PreTrainedModel, "set_use_kernels", configure)
    model = MultiVectorEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        device="cpu",
        kernel_config=preferences,
        model_kwargs=model_kwargs,
    )
    assert calls[0][0] is True
    assert calls[0][1].kernel_mapping == {"RMSNorm": "test/kernel:Norm"}
    assert calls[0][1].inherit_mapping is False
    assert model.kernel_config.kernel_mapping == original_mapping
    assert preferences.kernel_mapping == original_mapping
    assert model_kwargs == {"dtype": torch.float32}
    torch.testing.assert_close(model.similarity(*embeddings), 2 * maxsim(*embeddings))
    torch.testing.assert_close(model.similarity_pairwise(*embeddings), 2 * maxsim_pairwise(*embeddings))
    assert model._token_pooling_kernel_config.kernel_mapping == {"STWardClustering": "test/kernel:SingletonWard"}


@pytest.mark.parametrize("model_class", [SentenceTransformer, SparseEncoder, CrossEncoder, MultiVectorEncoder])
def test_model_kernel_options_require_torch_backend(model_class):
    with pytest.raises(ValueError, match="require the torch backend"):
        model_class(modules=[Normalize()], backend="onnx", use_kernels=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("options", [{"use_kernels": True}, {"kernel_config": KernelConfig()}])
def test_model_enables_default_cuda_providers(loader, monkeypatch, embeddings, options):
    monkeypatch.setattr("kernels.layer.layer.LayerRepository._resolve_revision", lambda self: f"v{self._version}")
    model = MultiVectorEncoder(modules=[Normalize()], device="cpu", **options)
    gpu = tuple(value.cuda() for value in embeddings)
    assert not loader
    torch.testing.assert_close(model.similarity(*gpu), 2 * maxsim(*gpu))
    torch.testing.assert_close(model.similarity_pairwise(*gpu), 2 * maxsim_pairwise(*gpu))
    assert len(loader) == 2
    assert all(args == ("sentence-transformers/late-interaction-kernels",) for args, _ in loader)
    assert all(kwargs["revision"] == "v0" for _, kwargs in loader)
    assert all(
        kwargs["trust_remote_code"] == ["sentence-transformers/late-interaction-kernels"] for _, kwargs in loader
    )


def test_missing_build_falls_back_and_other_errors_propagate(loader, monkeypatch, embeddings):
    import kernels.layer.layer as layer

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("no build")

    monkeypatch.setattr(layer, "get_kernel", unavailable)
    torch.testing.assert_close(scorer()(*embeddings), maxsim(*embeddings))
    pooler = HierarchicalTokenPooling(2, kernel_config=config(STWardClustering="SingletonWard"))
    torch.testing.assert_close(
        pooler.pool_one(embeddings[0][0]), HierarchicalTokenPooling(2).pool_one(embeddings[0][0])
    )

    def broken(*args, **kwargs):
        raise RuntimeError("broken provider")

    monkeypatch.setattr(layer, "get_kernel", broken)
    with pytest.raises(RuntimeError, match="broken provider"):
        scorer()(*embeddings)


def test_unknown_target_and_global_disabling(loader, monkeypatch, embeddings):
    with pytest.raises(ValueError, match="Unknown kernel targets"):
        _MaxSimScorer(config(STWardClustering="SingletonWard"))(*embeddings)
    monkeypatch.setenv("USE_HUB_KERNELS", "0")
    torch.testing.assert_close(scorer()(*embeddings), maxsim(*embeddings))
    assert not loader


def test_native_mode_mappings_keep_gradient_calls_safe(loader, embeddings):
    from kernels import LayerRepository, Mode

    preferences = KernelConfig(
        {
            "STMaxSim": {
                "cpu": {
                    Mode.TRAINING: LayerRepository("test/kernel", layer_name="ScaledMaxSim", version=1),
                    Mode.INFERENCE: LayerRepository("test/kernel", layer_name="InferenceMaxSim", version=1),
                },
            },
        },
        inherit_mapping=False,
    )
    selected = _MaxSimScorer(preferences)
    torch.testing.assert_close(selected(*embeddings), 2 * maxsim(*embeddings))
    with torch.inference_mode():
        torch.testing.assert_close(selected(*embeddings), 3 * maxsim(*embeddings))
    torch.testing.assert_close(selected(*embeddings), 2 * maxsim(*embeddings))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scorer_and_pooler_select_for_actual_input_device(loader, embeddings):
    selected = scorer()
    selected(*embeddings)
    gpu = tuple(value.cuda() for value in embeddings)
    torch.testing.assert_close(selected(*gpu), 2 * maxsim(*gpu))
    torch.testing.assert_close(selected(*embeddings, device="cuda"), 2 * maxsim(*embeddings, device="cuda"))
    assert selected(*embeddings, device="cuda").device.type == "cpu"
    pooler = HierarchicalTokenPooling(2, kernel_config=config(STWardClustering="SingletonWard"))
    pooler.pool_one(gpu[1][0])
    assert pooler._kernel_device.type == "cuda"
    pooler.pool_one(embeddings[1][0])
    assert pooler._kernel_device.type == "cpu"
