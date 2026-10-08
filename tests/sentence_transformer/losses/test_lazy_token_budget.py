from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import (
    AutoModel,
    DataCollatorWithFlattening,
    PreTrainedTokenizerFast,
    Qwen2_5_VLProcessor,
    Qwen2VLConfig,
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
    Qwen3VLProcessor,
)

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.data_collator import BaseDataCollator
from sentence_transformers.base.modules import Transformer
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Pooling

CHAT_TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' }}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}"""


def make_tokenizer() -> PreTrainedTokenizerFast:
    tokens = (
        "[PAD] [UNK] <|im_start|> <|im_end|> <|vision_start|> <|vision_end|> <|image_pad|> "
        "<|video_pad|> user system assistant query document look at this red blue one two three four five six "
        "seven eight nine ten"
    ).split()
    backend = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<|im_start|> $A", special_tokens=[("<|im_start|>", tokens.index("<|im_start|>"))]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"],
        model_max_length=64,
    )


@pytest.fixture(scope="module")
def tiny_qwen_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("tiny-qwen2-vl")
    tokenizer = make_tokenizer()
    vision_kwargs = {
        "size": {"shortest_edge": 16, "longest_edge": 64},
        "patch_size": 2,
        "temporal_patch_size": 1,
        "merge_size": 2,
    }
    processor = Qwen2VLProcessor(
        image_processor=Qwen2VLImageProcessor(**vision_kwargs),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(**vision_kwargs),
        chat_template=CHAT_TEMPLATE,
    )
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "attention_dropout": 0.2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000, "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
        },
        image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    with torch.random.fork_rng():
        torch.manual_seed(17)
        AutoModel.from_config(config).save_pretrained(folder)
    processor.save_pretrained(folder)
    return folder


@pytest.fixture
def tiny_qwen(tiny_qwen_path: Path, monkeypatch: pytest.MonkeyPatch) -> SentenceTransformer:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    transformer = Transformer(str(tiny_qwen_path), model_kwargs={"attn_implementation": "eager"}, unpad_inputs=False)
    model = SentenceTransformer(modules=[transformer, Pooling(16)], device="cpu")
    model.train()
    return model


@pytest.fixture
def image_paths(tmp_path: Path) -> list[str]:
    paths = []
    for index, size in enumerate([(15, 17), (16, 16), (32, 16), (16, 32)]):
        path = tmp_path / f"image-{index}.{'jpg' if index == 0 else 'png'}"
        image = Image.new("RGB", size, color=(40 + index * 30, 90, 160))
        if index == 0:
            exif = Image.Exif()
            exif[274] = 6  # 90-degree orientation; token count is invariant to swapping height/width.
            image.save(path, exif=exif)
        else:
            image.save(path)
        paths.append(str(path))
    return paths


def raw_features(columns):
    rows = [dict(zip(("query", "document"), values)) for values in zip(*columns)]
    collator = BaseDataCollator(
        lambda *args, **kwargs: pytest.fail("Unexpected preprocessing"), lazy_preprocessing=True
    )
    features, _ = SentenceTransformerTrainer.collect_features(None, collator(rows))
    return features


def message(*content):
    return [{"role": "user", "content": list(content)}]


def spy_tokenizer_calls(tokenizer):
    calls = []
    original_call = type(tokenizer).__call__

    def recording_call(instance, *args, **kwargs):
        output = original_call(instance, *args, **kwargs)
        if instance is tokenizer:
            calls.append((args[0] if args else kwargs.get("text"), kwargs, output))
        return output

    return calls, patch.object(type(tokenizer), "__call__", recording_call)


def loss_and_gradients(model, loss_fn, features):
    model.zero_grad(set_to_none=True)
    with torch.random.fork_rng():
        torch.manual_seed(29)
        loss = loss_fn(features, None)
        loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients.values())
    return loss.detach(), gradients


@pytest.mark.parametrize("processor_class", [Qwen2VLProcessor, Qwen2_5_VLProcessor, Qwen3VLProcessor])
def test_token_lengths_match_preprocess_with_sizes_images_prompt_and_truncation(
    tiny_qwen, image_paths, processor_class
):
    # Exercise each real processor's rendering/tokenization, without loading additional model weights.
    processor = tiny_qwen[0].processor
    tiny_qwen[0].processor = processor_class(
        image_processor=processor.image_processor,
        tokenizer=processor.tokenizer,
        video_processor=processor.video_processor,
        chat_template=processor.chat_template,
    )
    pil_image = Image.new("RGB", (17, 15), "red")
    inputs = [
        message({"type": "text", "text": "one two three four five six"}),
        message({"type": "image", "image": image_paths[0]}, {"type": "text", "text": "look at this"}),
        message(
            {"type": "image", "image": image_paths[1]},
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "red blue one two three four five six seven eight nine ten"},
        ),
    ]
    processing_kwargs = {"text": {"max_length": 28, "truncation": True}}
    lengths = tiny_qwen[0]._get_token_lengths(inputs, prompt="query ", processing_kwargs=processing_kwargs)
    features = tiny_qwen.preprocess(inputs, prompt="query ", processing_kwargs=processing_kwargs)
    assert lengths == features["attention_mask"].sum(dim=-1).tolist()
    assert lengths[-1] == 28
    default_lengths = tiny_qwen[0]._get_token_lengths(inputs, prompt="query ")
    assert default_lengths[-1] > lengths[-1]

    image_override = {"image": {"min_pixels": 16, "max_pixels": 16}}
    overridden = tiny_qwen[0]._get_token_lengths(inputs, processing_kwargs=image_override)
    expected = tiny_qwen.preprocess(inputs, processing_kwargs=image_override)["attention_mask"].sum(dim=-1).tolist()
    assert overridden == expected
    assert overridden != tiny_qwen[0]._get_token_lengths(inputs)
    assert tiny_qwen[0]._get_token_lengths(inputs, prompt="query ") == default_lengths


def test_token_lengths_preserve_text_flattening(tiny_qwen):
    inputs = ["one", "one two three four"]
    expected = tiny_qwen.preprocess(inputs, prompt="query ")["attention_mask"].sum(dim=-1).tolist()
    transformer = tiny_qwen[0]
    # Exercise preprocessing metadata on CPU; no FlashAttention kernel or forward pass is needed.
    transformer.can_flatten_inputs = True
    transformer._flatten_position_offset = 0
    transformer.data_collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True, return_seq_idx=True)
    features = tiny_qwen.preprocess(inputs, prompt="query ")
    assert features["cu_seq_lens_q"].diff().tolist() == expected
    assert transformer._get_token_lengths(inputs, prompt="query ") == expected


def test_text_token_lengths_are_ragged_without_attention_mask(tiny_qwen):
    inputs = ["one", "one two three four five six seven eight"]
    transformer = tiny_qwen[0]
    expected = transformer._get_token_lengths(inputs)
    calls, tokenizer_spy = spy_tokenizer_calls(transformer.tokenizer)
    processing_kwargs = {"text": {"return_attention_mask": False}, "common": {"return_tensors": None}}

    with tokenizer_spy:
        lengths = transformer._get_token_lengths(inputs, processing_kwargs=processing_kwargs)

    assert lengths == expected
    assert calls
    _, kwargs, output = calls[-1]
    assert kwargs["padding"] is False and kwargs.get("return_tensors") is None
    assert list(map(len, output["input_ids"])) == lengths
    assert len(set(map(len, output["input_ids"]))) > 1


def test_qwen2_token_lengths_only_read_image_metadata(tiny_qwen, image_paths):
    inputs = [
        message({"type": "image", "image": image_paths[0]}),
        message({"type": "image", "image": image_paths[1]}, {"type": "image", "image": image_paths[2]}),
    ]
    image_processor = tiny_qwen[0].processor.image_processor
    with (
        patch.object(image_processor, "preprocess", side_effect=AssertionError("image pixels were processed")),
        patch.object(Image.Image, "load", side_effect=AssertionError("image pixels were decoded")),
        patch.object(Image.Image, "resize", side_effect=AssertionError("image pixels were resized")),
    ):
        lengths = tiny_qwen[0]._get_token_lengths(inputs)
    expected = tiny_qwen.preprocess(inputs)["attention_mask"].sum(dim=-1).tolist()
    assert lengths == expected


def test_image_token_lengths_use_compact_formula_for_huge_patch_counts(tiny_qwen, image_paths):
    transformer = tiny_qwen[0]
    processor = transformer.processor
    image_processor = processor.image_processor
    inputs = [message({"type": "image", "image": image_paths[1]}, {"type": "text", "text": "one"})]
    processing_kwargs = {"text": {"truncation": False}}
    with patch.object(image_processor, "get_number_of_image_patches", return_value=4):
        one_image_token_length = transformer._get_token_lengths(inputs, processing_kwargs=processing_kwargs)[0]

    calls, tokenizer_spy = spy_tokenizer_calls(processor.tokenizer)
    with (
        tokenizer_spy,
        patch.object(image_processor, "get_number_of_image_patches", return_value=4_000_000),
        patch.object(Image.Image, "load", side_effect=AssertionError("image pixels were decoded")),
        patch.object(Image.Image, "resize", side_effect=AssertionError("image pixels were resized")),
    ):
        huge_length = transformer._get_token_lengths(inputs, processing_kwargs=processing_kwargs)[0]

    assert huge_length == one_image_token_length + 999_999
    assert calls
    for rendered, kwargs, output in calls:
        rendered = [rendered] if isinstance(rendered, str) else rendered
        assert all(text.count(processor.image_token) == 1 and len(text) < 200 for text in rendered)
        assert kwargs["padding"] is False and kwargs.get("return_tensors") is None
        assert max(map(len, output["input_ids"])) < 20


def test_image_token_lengths_match_safe_truncation_and_reject_partial_span(tiny_qwen, image_paths):
    transformer = tiny_qwen[0]
    tokenizer = transformer.tokenizer
    image_token_id = transformer.processor.image_token_id
    inputs = [
        message(
            {"type": "text", "text": "one two three four"},
            {"type": "image", "image": image_paths[2]},
            {"type": "text", "text": "five six seven eight nine ten"},
        )
    ]
    full_ids = tiny_qwen.preprocess(inputs, processing_kwargs={"text": {"truncation": False}})["input_ids"][0]
    image_positions = (full_ids == image_token_id).nonzero().flatten().tolist()
    assert len(image_positions) > 1

    max_lengths = {
        "right": image_positions[-1] + 2,
        "left": len(full_ids) - image_positions[0] + 1,
    }
    for side, max_length in max_lengths.items():
        with patch.object(tokenizer, "truncation_side", side):
            processing_kwargs = {"text": {"truncation": True, "max_length": max_length}}
            expected = tiny_qwen.preprocess(inputs, processing_kwargs=processing_kwargs)
            assert transformer._get_token_lengths(inputs, processing_kwargs=processing_kwargs) == [
                int(expected["attention_mask"].sum())
            ]

    partial_lengths = {
        "right": image_positions[0] + 1,
        "left": len(full_ids) - image_positions[-1],
    }
    for side, max_length in partial_lengths.items():
        with patch.object(tokenizer, "truncation_side", side):
            partial = {"text": {"truncation": True, "max_length": max_length}}
            with pytest.raises(ValueError, match=r"(?i)image.*token"):
                tiny_qwen.preprocess(inputs, processing_kwargs=partial)
            with pytest.raises(ValueError, match=r"(?i)image.*token"):
                transformer._get_token_lengths(inputs, processing_kwargs=partial)


def test_token_lengths_reuse_processor_bos_behavior_without_mutation(tiny_qwen, image_paths):
    transformer = tiny_qwen[0]
    processor = transformer.processor
    process_images = processor._process_images
    fetch_images = processor.image_processor.fetch_images
    processor_config = copy.deepcopy(processor.to_dict())
    image_processor_config = copy.deepcopy(processor.image_processor.to_dict())
    tokenizer_config = copy.deepcopy(processor.tokenizer.init_kwargs)
    post_processor_state = processor.tokenizer.backend_tokenizer.post_processor.__getstate__()
    inputs = [message({"type": "image", "image": image_paths[0]})]

    lengths = transformer._get_token_lengths(inputs)

    assert transformer.processor is processor
    assert processor._process_images.__self__ is process_images.__self__
    assert processor._process_images.__func__ is process_images.__func__
    assert processor.image_processor.fetch_images.__self__ is fetch_images.__self__
    assert processor.image_processor.fetch_images.__func__ is fetch_images.__func__
    assert processor.to_dict() == processor_config
    assert processor.image_processor.to_dict() == image_processor_config
    assert processor.tokenizer.init_kwargs == tokenizer_config
    assert processor.tokenizer.backend_tokenizer.post_processor.__getstate__() == post_processor_state

    rendered = processor.apply_chat_template(inputs, tokenize=False)
    bos_token_id = processor.tokenizer.bos_token_id
    # A bare tokenizer call prepends BOS although the rendered chat already starts with BOS.
    assert processor.tokenizer(rendered[0])["input_ids"][:2] == [bos_token_id, bos_token_id]
    # ProcessorMixin detects that rendered BOS and disables tokenizer special-token insertion.
    processed = processor.apply_chat_template(inputs, tokenize=True, return_dict=True, return_tensors="pt")
    assert processed["input_ids"][0, :2].tolist() != [bos_token_id, bos_token_id]
    assert lengths == processed["attention_mask"].sum(dim=-1).tolist()

    features = tiny_qwen.preprocess(inputs)
    assert "pixel_values" in features and features["pixel_values"].numel() > 0


@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_image_token_lengths_preserve_tokenizer_added_special_tokens(tiny_qwen, image_paths, truncation_side):
    processor = tiny_qwen[0].processor
    tokenizer = processor.tokenizer
    # Without an initial BOS in the rendered chat, the real processor adds tokenizer BOS/EOS.
    processor.chat_template = "look " + processor.chat_template
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="<|im_start|> $A <|im_end|>",
        special_tokens=[("<|im_start|>", tokenizer.bos_token_id), ("<|im_end|>", tokenizer.eos_token_id)],
    )
    tokenizer.truncation_side = truncation_side
    inputs = [
        message(
            {"type": "text", "text": "one " * 8},
            {"type": "image", "image": image_paths[2]},
            {"type": "text", "text": "two " * 8},
        )
    ]
    full_length = tiny_qwen.preprocess(inputs)["attention_mask"].sum().item()
    options = {"text": {"truncation": True, "max_length": full_length - 2}}
    expected = tiny_qwen.preprocess(inputs, processing_kwargs=options)["attention_mask"].sum(dim=-1).tolist()
    assert tiny_qwen[0]._get_token_lengths(inputs, processing_kwargs=options) == expected


def test_token_lengths_preserve_disabled_truncation_with_padding(tiny_qwen):
    inputs = ["one two three four five", "one"]
    options = {"text": {"padding": True, "truncation": None, "max_length": 4}}
    expected = tiny_qwen.preprocess(inputs, processing_kwargs=options)["attention_mask"].sum(dim=-1).tolist()
    assert expected[0] > 4
    assert tiny_qwen[0]._get_token_lengths(inputs, processing_kwargs=options) == expected


def test_lazy_token_budget_matches_eager_loss_gradients_and_replays_boundaries(tiny_qwen, image_paths):
    columns = [
        ["one", "one two", "one two three four", "one two three four five six seven"],
        [{"image": path} for path in image_paths],
    ]
    budget = 12
    lazy_features = raw_features(columns)
    loss_fn = CachedMultipleNegativesRankingLoss(tiny_qwen, mini_batch_num_tokens=budget)
    eager_features = [tiny_qwen.preprocess(column) for column in columns]
    ranges = [loss_fn._get_minibatch_ranges(feature) for feature in lazy_features]
    assert ranges == [loss_fn._get_minibatch_ranges(feature) for feature in eager_features]
    assert len(ranges[1]) > 1
    assert len({end - begin for column_ranges in ranges for begin, end in column_ranges}) > 1

    calls = []
    original_preprocess = tiny_qwen.preprocess

    def recording_preprocess(inputs, **kwargs):
        calls.append(tuple(map(str, inputs)))
        return original_preprocess(inputs, **kwargs)

    with patch.object(tiny_qwen, "preprocess", side_effect=recording_preprocess):
        lazy_loss, lazy_gradients = loss_and_gradients(tiny_qwen, loss_fn, lazy_features)
    calls_per_pass = sum(map(len, ranges))
    assert calls[:calls_per_pass] == calls[calls_per_pass:] and len(calls) == calls_per_pass * 2

    eager_loss, eager_gradients = loss_and_gradients(tiny_qwen, loss_fn, eager_features)
    torch.testing.assert_close(lazy_loss, eager_loss, rtol=1e-5, atol=1e-6)
    assert lazy_gradients.keys() == eager_gradients.keys()
    for name in lazy_gradients:
        torch.testing.assert_close(lazy_gradients[name], eager_gradients[name], rtol=2e-4, atol=2e-5, msg=name)


def test_trainer_lazy_token_budget_train_and_evaluate(tiny_qwen, image_paths, tmp_path):
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_dict(
        {
            "query": ["one", "one two", "one two three", "one two three four"],
            "document": [{"image": path} for path in image_paths],
        }
    )
    trainer = SentenceTransformerTrainer(
        model=tiny_qwen,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / "output"),
            use_cpu=True,
            lazy_preprocessing=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            max_steps=1,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            dataloader_pin_memory=False,
            prompts={"query": "query "},
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(tiny_qwen, mini_batch_num_tokens=12),
    )
    assert math.isfinite(trainer.train().training_loss)
    assert math.isfinite(trainer.evaluate()["eval_loss"])


@pytest.mark.parametrize(
    "processing_kwargs,match",
    [
        ({"image": {"do_resize": False}}, "do_resize"),
        ({"image": {"size": {"shortest_edge": 20, "longest_edge": 40}}}, "size"),
    ],
)
def test_qwen2_token_lengths_reject_unsupported_image_options(tiny_qwen, image_paths, processing_kwargs, match):
    with pytest.raises(ValueError, match=match):
        tiny_qwen[0]._get_token_lengths([{"image": image_paths[0]}], processing_kwargs=processing_kwargs)
