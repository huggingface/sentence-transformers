from __future__ import annotations

import math
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, PreTrainedTokenizerFast

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.base.data_collator import BaseDataCollator
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.util import batch_to_device


def preprocess_minibatch(model, inputs):
    return batch_to_device(model.preprocess(inputs), model.device)


def raw_features(columns):
    rows = [dict(zip(("query", "image"), values)) for values in zip(*columns)]
    collator = BaseDataCollator(
        lambda *args, **kwargs: pytest.fail("Unexpected preprocessing"), lazy_preprocessing=True
    )
    features, _ = SentenceTransformerTrainer.collect_features(None, collator(rows))
    return features


@pytest.fixture
def tiny_clip(tmp_path: Path):
    """Real local CLIP + processor, with random weights; no Hub downloads."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(17)
            yield make_tiny_clip(tmp_path)
    finally:
        torch.set_num_threads(previous_threads)


def make_tiny_clip(tmp_path: Path):
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "dog": 5}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=16,
    )
    common = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}
    config = CLIPConfig(
        text_config={
            **common,
            "vocab_size": 6,
            "max_position_embeddings": 16,
            "pad_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
        },
        vision_config={**common, "image_size": 16, "patch_size": 8},
        projection_dim=8,
    )
    folder = tmp_path / "model"
    CLIPModel(config).save_pretrained(folder)
    CLIPProcessor(
        image_processor=CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}),
        tokenizer=tokenizer,
    ).save_pretrained(folder)
    model = SentenceTransformer(
        modules=[Transformer(str(folder), model_kwargs={"attn_implementation": "eager"})], device="cpu"
    )
    model.train()
    return model


@pytest.fixture
def raw_columns(tmp_path: Path):
    rows = []
    for i in range(5):
        path = tmp_path / f"image-{i}.png"
        with Image.new("RGB", (24 + i, 20 + i), color=(30 + 40 * i, 200 - 30 * i, 80 + 20 * i)) as image:
            image.save(path)
        rows.append({"query": "a dog " * (i + 1), "image": str(path)})
    return [[row["query"] for row in rows], [{"image": row["image"]} for row in rows]]


def loss_and_gradients(model, loss_fn, inputs):
    model.zero_grad(set_to_none=True)
    loss = loss_fn(inputs, None)
    loss.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    assert gradients and any(grad.count_nonzero() for grad in gradients.values())
    assert all(torch.isfinite(grad).all() for grad in gradients.values())
    return loss.detach(), gradients


@pytest.mark.parametrize("mini_batch_size", [1, 2, 8])
def test_native_lazy_loss_matches_eager(tiny_clip, raw_columns, mini_batch_size):
    from sentence_transformers import SentenceTransformerTrainer

    rows = [dict(zip(("query", "image"), values)) for values in zip(*raw_columns)]
    collator = BaseDataCollator(tiny_clip.preprocess, lazy_preprocessing=True, prompts={"query": "a "})
    with patch.object(tiny_clip, "preprocess", wraps=tiny_clip.preprocess) as preprocess:
        features, labels = SentenceTransformerTrainer.collect_features(None, collator(rows))
        preprocess.assert_not_called()
        loss_fn = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=mini_batch_size)
        actual_loss, actual_gradients = loss_and_gradients(tiny_clip, loss_fn, features)
        assert all(len(call.args[0]) <= mini_batch_size for call in preprocess.call_args_list)
    expected_loss, expected_gradients = loss_and_gradients(
        tiny_clip,
        loss_fn,
        [tiny_clip.preprocess(column, prompt="a " if i == 0 else None) for i, column in enumerate(raw_columns)],
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_trainer_lazy_train_evaluate_reload(tiny_clip, raw_columns, tmp_path, lazy, device):
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    dataset = Dataset.from_dict({"query": raw_columns[0], "image": raw_columns[1]})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "output"),
        use_cpu=device == "cpu",
        lazy_preprocessing=lazy,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        max_steps=1,
        learning_rate=1e-3,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_pin_memory=False,
        prompts={"query": "a "},
    )
    loss_fn = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)
    trainer = SentenceTransformerTrainer(
        model=tiny_clip, args=args, train_dataset=dataset, eval_dataset=dataset, loss=loss_fn
    )
    before = {name: p.detach().clone() for name, p in tiny_clip.named_parameters()}
    original_preprocess = tiny_clip.preprocess
    calls = []
    refs = []

    def preprocess(inputs, **kwargs):
        calls.append((len(inputs), kwargs))
        return original_preprocess(inputs, **kwargs)

    def observe_inputs(module, inputs):
        tensors = [v for v in inputs[0].values() if isinstance(v, torch.Tensor)]
        assert all(t.device.type == device for t in tensors)
        refs.extend(weakref.ref(t) for t in tensors)

    tiny_clip.preprocess = preprocess
    trainer.data_collator.preprocess_fn = preprocess
    hook = tiny_clip.register_forward_pre_hook(observe_inputs)
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        if lazy:
            assert not calls
            assert sorted(item["image"] for item in batch["image_raw_inputs"]) == sorted(
                item["image"] for item in raw_columns[1]
            )
        del batch
        calls.clear()
        result = trainer.train()
        assert math.isfinite(result.training_loss)
        assert any(not torch.equal(before[name].to(p.device), p) for name, p in tiny_clip.named_parameters())
        assert [size for size, _ in calls] == ([2, 2, 1] * 4 if lazy else [5, 5])
        assert all(ref() is None for ref in refs)
        calls.clear()
        assert math.isfinite(trainer.evaluate()["eval_loss"])
        assert [size for size, _ in calls] == ([2, 2, 1] * 2 if lazy else [5, 5])
    finally:
        hook.remove()
        tiny_clip.preprocess = original_preprocess
    trainer.save_model(str(tmp_path / "saved"))
    reloaded = SentenceTransformer(str(tmp_path / "saved"), device=device)
    for column in raw_columns:
        torch.testing.assert_close(tiny_clip.encode(column), reloaded.encode(column))


@pytest.mark.parametrize("mini_batch_size", [1, 2, 5, 8])
@pytest.mark.parametrize("reference", ["cached", "plain"])
def test_lazy_matches_eager_loss_and_gradients(tiny_clip, raw_columns, mini_batch_size, reference):
    eager = (
        CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=mini_batch_size)
        if reference == "cached"
        else MultipleNegativesRankingLoss(tiny_clip)
    )
    expected_loss, expected_gradients = loss_and_gradients(
        tiny_clip, eager, [preprocess_minibatch(tiny_clip, column) for column in raw_columns]
    )
    actual_loss, actual_gradients = loss_and_gradients(
        tiny_clip,
        CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=mini_batch_size),
        raw_features(raw_columns),
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in expected_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name], rtol=2e-4, atol=2e-5, msg=name)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_lazy_limits_preprocessing_and_replays_dropout(tiny_clip, raw_columns, device):
    model = tiny_clip.to(device)
    # Make the existing CLIP attention dropout non-zero in both encoders.
    for module in model.modules():
        if hasattr(module, "dropout") and isinstance(module.dropout, float):
            module.dropout = 0.25
    assert any(getattr(module, "dropout", None) == 0.25 for module in model.modules())
    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    calls, embeddings, tensor_refs = [], [], []
    original = model.preprocess

    def recording_preprocess(inputs, **kwargs):
        calls.append(len(inputs))
        return original(inputs, **kwargs)

    def record_inputs(module, args):
        # Observe the actual sliced model inputs, after device transfer. Tracking
        # only the processor's CPU tensors would miss retained CUDA inputs.
        for value in args[0].values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == device
                tensor_refs.append(weakref.ref(value))

    def record_forward(module, args, output):
        assert output["sentence_embedding"].device.type == device
        embeddings.append(output["sentence_embedding"].detach().cpu().clone())

    with (
        patch.object(model, "preprocess", side_effect=recording_preprocess),
        model.register_forward_pre_hook(record_inputs),
        model.register_forward_hook(record_forward),
    ):
        loss = loss_fn(raw_features(raw_columns), None)
        first_pass_count = len(raw_columns) * math.ceil(len(raw_columns[0]) / 2)
        assert len(calls) == first_pass_count
        assert calls == [2, 2, 1, 2, 2, 1]
        assert all(ref() is None for ref in tensor_refs)
        loss.backward()
        assert calls == [2, 2, 1, 2, 2, 1] * 2
        assert all(ref() is None for ref in tensor_refs)
        for first, replay in zip(embeddings[:first_pass_count], embeddings[first_pass_count:]):
            torch.testing.assert_close(first, replay, rtol=0, atol=0)


def test_no_grad_only_preprocesses_once(tiny_clip, raw_columns):
    with torch.no_grad(), patch.object(tiny_clip, "preprocess", wraps=tiny_clip.preprocess) as preprocess:
        actual = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)(raw_features(raw_columns), None)
        assert preprocess.call_count == 6
        assert not actual.requires_grad
        expected = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)(
            [preprocess_minibatch(tiny_clip, column) for column in raw_columns], None
        )
    torch.testing.assert_close(actual, expected)


def test_collator_does_not_open_images():
    with patch.object(Image, "open", side_effect=AssertionError("Unexpected image decoding")):
        features = raw_features([["a dog"], [{"image": "not-opened.png"}]])
        assert features[1]["raw_inputs"] == [{"image": "not-opened.png"}]


def test_lazy_optimizer_step_and_save_reload(tiny_clip, raw_columns, tmp_path):
    before = {name: p.detach().clone() for name, p in tiny_clip.named_parameters()}
    optimizer = torch.optim.SGD(tiny_clip.parameters(), lr=1e-3)
    loss = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)(raw_features(raw_columns), None)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(p, before[name]) for name, p in tiny_clip.named_parameters())
    folder = tmp_path / "trained"
    tiny_clip.save_pretrained(str(folder))
    reloaded = SentenceTransformer(str(folder), device="cpu", local_files_only=True)
    tiny_clip.eval()
    reloaded.eval()
    with torch.no_grad():
        for column in raw_columns:
            expected = tiny_clip(preprocess_minibatch(tiny_clip, column))["sentence_embedding"]
            actual = reloaded(preprocess_minibatch(reloaded, column))["sentence_embedding"]
            assert actual.shape == (5, 8) and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected)


def test_trainer_accumulation_and_partial_eval_match_eager(tiny_clip, raw_columns, tmp_path):
    import copy

    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainingArguments

    dataset = Dataset.from_dict({"query": raw_columns[0], "image": raw_columns[1]})
    results = []
    for lazy in (False, True):
        model = copy.deepcopy(tiny_clip)
        trainer = SentenceTransformerTrainer(
            model=model,
            args=SentenceTransformerTrainingArguments(
                output_dir=str(tmp_path / str(lazy)),
                use_cpu=True,
                lazy_preprocessing=lazy,
                per_device_train_batch_size=3,
                per_device_eval_batch_size=3,
                gradient_accumulation_steps=2,
                max_steps=1,
                learning_rate=1e-3,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=2),
        )
        training_loss = trainer.train().training_loss
        eval_loss = trainer.evaluate()["eval_loss"]
        results.append((training_loss, eval_loss, model.state_dict()))
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-5)
    assert results[0][1] == pytest.approx(results[1][1], rel=1e-5)
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name], results[1][2][name], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("configuration", ["plain_loss", "eager_collator", "wrapped_loss"])
def test_trainer_rejects_incompatible_lazy_configuration(tiny_clip, tmp_path, configuration):
    from sentence_transformers import SentenceTransformerTrainingArguments

    loss = (
        MultipleNegativesRankingLoss(tiny_clip)
        if configuration == "plain_loss"
        else CachedMultipleNegativesRankingLoss(tiny_clip)
    )
    if configuration == "wrapped_loss":
        from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss

        loss = MatryoshkaLoss(tiny_clip, loss, matryoshka_dims=[8, 4])
    with pytest.raises(ValueError, match="lazy_preprocessing"):
        SentenceTransformerTrainer(
            model=tiny_clip,
            args=SentenceTransformerTrainingArguments(
                output_dir=str(tmp_path), use_cpu=True, lazy_preprocessing=True, report_to=[]
            ),
            loss=loss,
            data_collator=BaseDataCollator(tiny_clip.preprocess) if configuration == "eager_collator" else None,
        )


def test_lazy_router_preserves_prompt_and_task(tiny_clip, raw_columns):
    from sentence_transformers.base.modules import Router

    model = SentenceTransformer(
        modules=[Router({"query": list(tiny_clip.children()), "document": list(tiny_clip.children())})], device="cpu"
    )
    rows = [dict(zip(("query", "image"), values), dataset_name="pairs") for values in zip(*raw_columns)]
    options = {
        "prompts": {"pairs": {"query": "a "}},
        "router_mapping": {"pairs": {"query": "query", "image": "document"}},
        "max_length": {"query": 8},
    }
    loss_fn = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    results = []
    for lazy in (False, True):
        collator = BaseDataCollator(model.preprocess, lazy_preprocessing=lazy, **options)
        features, labels = SentenceTransformerTrainer.collect_features(None, collator(rows))
        results.append(loss_and_gradients(model, loss_fn, features))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1].keys() == results[1][1].keys()
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("loss_dict", [False, True])
def test_custom_lazy_collator_checks_every_loss(tiny_clip, tmp_path, loss_dict):
    from sentence_transformers import SentenceTransformerTrainingArguments

    loss = MultipleNegativesRankingLoss(tiny_clip)
    if loss_dict:
        loss = {"supported": CachedMultipleNegativesRankingLoss(tiny_clip), "unsupported": loss}
    with pytest.raises(ValueError, match="lazy_preprocessing currently requires"):
        SentenceTransformerTrainer(
            model=tiny_clip,
            args=SentenceTransformerTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
            data_collator=BaseDataCollator(tiny_clip.preprocess, lazy_preprocessing=True),
            loss=loss,
        )


def test_lazy_rejects_custom_preprocess_that_would_be_ignored(tiny_clip, tmp_path):
    from sentence_transformers import SentenceTransformerTrainingArguments

    def custom_preprocess(inputs, **kwargs):
        return tiny_clip.preprocess(inputs, processing_kwargs={"text": {"max_length": 2}}, **kwargs)

    with pytest.raises(ValueError, match="preprocess_fn"):
        SentenceTransformerTrainer(
            model=tiny_clip,
            args=SentenceTransformerTrainingArguments(
                output_dir=str(tmp_path), use_cpu=True, lazy_preprocessing=True, report_to=[]
            ),
            loss=CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2),
            data_collator=BaseDataCollator(custom_preprocess, lazy_preprocessing=True),
        )


def test_lazy_streaming_train_and_evaluate(tiny_clip, raw_columns, tmp_path):
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainingArguments

    dataset = Dataset.from_dict({"query": raw_columns[0], "image": raw_columns[1]}).to_iterable_dataset()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        lazy_preprocessing=True,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        max_steps=1,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    trainer = SentenceTransformerTrainer(
        model=tiny_clip,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2),
    )
    assert math.isfinite(trainer.train().training_loss)
    assert math.isfinite(trainer.evaluate()["eval_loss"])


def test_lazy_fixed_length_left_padding_matches_eager(tiny_clip, raw_columns):
    # Dynamic left padding can change absolute token positions when the batch is
    # split. A fixed padding width preserves the eager inputs for these models.
    tiny_clip[0].processor.tokenizer.padding_side = "left"
    tiny_clip[0].processing_kwargs = {"text": {"padding": "max_length", "max_length": 16}}
    loss_fn = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)
    expected_loss, expected_grads = loss_and_gradients(
        tiny_clip, loss_fn, [tiny_clip.preprocess(column) for column in raw_columns]
    )
    actual_loss, actual_grads = loss_and_gradients(tiny_clip, loss_fn, raw_features(raw_columns))
    torch.testing.assert_close(actual_loss, expected_loss)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=1e-4, atol=1e-5)


def test_eager_raw_inputs_metadata_does_not_enable_lazy(tiny_clip):
    from sentence_transformers.base.losses.gradcache import _get_batch_size, _minibatch_ranges

    features = tiny_clip.preprocess(["a", "a dog"])
    features["raw_inputs"] = ["custom metadata"]
    assert _get_batch_size(features) == 2
    assert _minibatch_ranges(features, mini_batch_size=2, mini_batch_num_tokens=100) == [(0, 2)]
    loss_fn = CachedMultipleNegativesRankingLoss(tiny_clip, mini_batch_size=2)
    with patch.object(tiny_clip, "preprocess", side_effect=AssertionError("Eager inputs were preprocessed again")):
        embeddings, _ = loss_fn.embed_minibatch(features, 0, 2, with_grad=False, copy_random_state=False)
    assert embeddings.shape == (2, 8)
