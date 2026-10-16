from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertLMHeadModel,
    BertModel,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer


@pytest.fixture
def local_tsdae(tmp_path):
    def build(*, padding_side="right", pad_is_eos=False, decoder_type="bert"):
        decoder_path = tmp_path / f"decoder-{padding_side}-{pad_is_eos}-{decoder_type}"
        model_path = tmp_path / f"bert-{padding_side}-{pad_is_eos}"
        vocab = {
            token: index
            for index, token in enumerate(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "alpha", "beta"]
            )
        }
        tokenizer = BertTokenizer(
            vocab=vocab,
            padding_side=padding_side,
            eos_token="[SEP]",
            pad_token="[SEP]" if pad_is_eos else "[PAD]",
        )
        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        with torch.random.fork_rng():
            torch.manual_seed(42)
            BertModel(config).save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            if decoder_type == "bert":
                decoder_config = BertConfig(**config.to_dict())
                decoder_config.is_decoder = True
                decoder_config.add_cross_attention = True
                BertLMHeadModel(decoder_config).save_pretrained(decoder_path)
                tokenizer.save_pretrained(decoder_path)
            else:
                decoder_vocab = {character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))}
                decoder_vocab.update({"<unk>": 256, "<bos>": 257, "<eos>": 258})
                backend = Tokenizer(BPE(decoder_vocab, merges=[], unk_token="<unk>"))
                backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
                backend.decoder = ByteLevelDecoder()
                backend.post_processor = TemplateProcessing(
                    single="<bos> $A <eos>",
                    special_tokens=[("<bos>", 257), ("<eos>", 258)],
                )
                decoder_tokenizer = GPT2TokenizerFast(
                    tokenizer_object=backend,
                    unk_token="<unk>",
                    bos_token="<bos>",
                    eos_token="<eos>",
                    padding_side=padding_side,
                )
                decoder_tokenizer.save_pretrained(decoder_path)
                GPT2LMHeadModel(
                    GPT2Config(
                        vocab_size=len(decoder_vocab),
                        n_embd=16,
                        n_layer=1,
                        n_head=2,
                        n_positions=32,
                        bos_token_id=257,
                        eos_token_id=258,
                        add_cross_attention=True,
                        resid_pdrop=0.0,
                        embd_pdrop=0.0,
                        attn_pdrop=0.0,
                    )
                ).save_pretrained(decoder_path)
            model = SentenceTransformer(modules=[Transformer(str(model_path)), Pooling(16)], device="cpu")
            model.model_card_data.generate_widget_examples = False
            loss = DenoisingAutoEncoderLoss(
                model,
                decoder_name_or_path=str(decoder_path),
                tie_encoder_decoder=False,
            )
        loss.eval()
        return loss

    return build


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_tsdae_decoder_receives_target_attention_mask(local_tsdae, padding_side):
    loss = local_tsdae(padding_side=padding_side)
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    target = loss.encoder.preprocess(["alpha", "alpha beta world"])
    calls = []
    handle = loss.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs),
        with_kwargs=True,
    )
    try:
        value = loss([source, target], labels=None)
    finally:
        handle.remove()
    assert torch.isfinite(value)
    if padding_side == "left":
        torch.testing.assert_close(calls[0]["attention_mask"], target["attention_mask"][:, :-1])
    else:
        assert calls[0]["attention_mask"] is None


def test_tsdae_left_padded_source_receives_reconstruction_gradients(local_tsdae):
    loss = local_tsdae(padding_side="left")
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    # Keep only the shorter row, whose first token is padding.
    source = {key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in source.items()}
    assert source["attention_mask"][0, 0] == 0
    target = loss.encoder.preprocess(["alpha beta"])
    pooled = []

    def retain_pool(module, args, output):
        pooled.append(output["sentence_embedding"])
        pooled[-1].retain_grad()

    handle = loss.encoder.register_forward_hook(retain_pool)
    try:
        loss([source, target], labels=None).backward()
    finally:
        handle.remove()
    assert pooled[0].grad is not None
    assert pooled[0].grad.abs().sum() > 0
    encoder_grads = [parameter.grad for parameter in loss.encoder.parameters() if parameter.grad is not None]
    assert all(torch.isfinite(gradient).all() for gradient in encoder_grads)
    assert sum(gradient.abs().sum() for gradient in encoder_grads) > 0


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("pad_is_eos", [False, True], ids=["distinct-pad", "pad-equals-eos"])
def test_tsdae_reconstruction_matches_masked_next_token_cross_entropy(local_tsdae, padding_side, pad_is_eos):
    loss = local_tsdae(padding_side=padding_side, pad_is_eos=pad_is_eos)
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    target = loss.encoder.preprocess(["alpha", "alpha beta world"])
    original_target = {key: value.clone() for key, value in target.items() if isinstance(value, torch.Tensor)}
    actual = loss([source, target], labels=None)
    reps = loss.encoder(source)["sentence_embedding"]
    input_mask = target["attention_mask"][:, :-1]
    logits = loss.decoder(
        input_ids=target["input_ids"][:, :-1],
        attention_mask=input_mask,
        encoder_hidden_states=reps[:, None],
        encoder_attention_mask=None,
        use_cache=False,
    ).logits
    # Enumerate real next-token pairs independently of the loss's vectorized ignore mask.
    positions = [
        (row, column)
        for row in range(len(target["input_ids"]))
        for column in range(1, target["input_ids"].shape[1])
        if target["attention_mask"][row, column] and target["attention_mask"][row, column - 1]
    ]
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[row, column - 1] for row, column in positions]),
        torch.stack([target["input_ids"][row, column] for row, column in positions]),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for key, original in original_target.items():
        torch.testing.assert_close(target[key], original)
    actual.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in loss.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("target_texts", [["alpha", "alpha beta world"], ["", ""]], ids=["ordinary", "eos-only"])
def test_tsdae_gpt2_retokenization_learns_eos_when_it_is_also_padding(local_tsdae, padding_side, target_texts):
    loss = local_tsdae(padding_side=padding_side, decoder_type="gpt2")
    assert loss.need_retokenization
    assert loss.tokenizer_decoder.pad_token_id == loss.tokenizer_decoder.eos_token_id
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    target = loss.encoder.preprocess(target_texts)
    retokenized = loss.retokenize(target)
    assert loss.tokenizer_decoder.batch_decode(retokenized["input_ids"], skip_special_tokens=True) == target_texts
    original_target = {key: value.clone() for key, value in target.items() if isinstance(value, torch.Tensor)}
    actual = loss([source, target], labels=None)
    reps = loss.encoder(source)["sentence_embedding"]
    logits = loss.decoder(
        input_ids=retokenized["input_ids"][:, :-1],
        attention_mask=retokenized["attention_mask"][:, :-1],
        encoder_hidden_states=reps[:, None],
        encoder_attention_mask=None,
        use_cache=False,
    ).logits
    positions = [
        (row, column)
        for row in range(len(retokenized["input_ids"]))
        for column in range(1, retokenized["input_ids"].shape[1])
        if retokenized["attention_mask"][row, column] and retokenized["attention_mask"][row, column - 1]
    ]
    labels = torch.stack([retokenized["input_ids"][row, column] for row, column in positions])
    assert (labels == loss.tokenizer_decoder.eos_token_id).sum() == len(source["input_ids"])
    if any(target_texts):
        assert (labels != loss.tokenizer_decoder.eos_token_id).any()
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[row, column - 1] for row, column in positions]),
        labels,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual_grad = torch.autograd.grad(actual, loss.decoder.get_output_embeddings().weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, loss.decoder.get_output_embeddings().weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual_grad).all()
    for key, original in original_target.items():
        torch.testing.assert_close(target[key], original)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_tsdae_ignores_masked_target_token_ids(local_tsdae, padding_side):
    loss = local_tsdae(padding_side=padding_side)
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    target = loss.encoder.preprocess(["alpha", "alpha beta world"])
    changed_target = {key: value.clone() for key, value in target.items() if isinstance(value, torch.Tensor)}
    changed_target["input_ids"][~target["attention_mask"].bool()] = loss.tokenizer_decoder.convert_tokens_to_ids(
        "world"
    )
    actual = loss([source, target], labels=None)
    changed = loss([source, changed_target], labels=None)
    torch.testing.assert_close(changed, actual, rtol=1e-6, atol=1e-6)


def test_tsdae_masked_reconstruction_runs_through_public_trainer(local_tsdae, tmp_path):
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    loss = local_tsdae(padding_side="left", decoder_type="gpt2")
    before = loss.encoder.transformers_model.get_input_embeddings().weight.detach().clone()
    trainer = SentenceTransformerTrainer(
        model=loss.encoder,
        loss=loss,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / "train"),
            per_device_train_batch_size=2,
            max_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            use_cpu=True,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
        ),
        train_dataset=Dataset.from_dict(
            {"noisy": ["hello", "hello world alpha"], "original": ["alpha", "alpha beta world"]}
        ),
    )
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    after = loss.encoder.transformers_model.get_input_embeddings().weight.detach()
    assert not torch.equal(before, after)
    assert all(torch.isfinite(parameter).all() for parameter in loss.parameters())


@pytest.mark.parametrize(
    "target_texts", [["alpha beta", "beta world"], ["alpha", "alpha beta world"]], ids=["unpadded", "padded"]
)
@pytest.mark.parametrize("pad_is_eos", [False, True], ids=["distinct-pad", "shared-eos-pad"])
def test_tsdae_preserves_public_target_preprocessing_without_attention_mask(local_tsdae, target_texts, pad_is_eos):
    loss = local_tsdae(pad_is_eos=pad_is_eos)
    options = {"text": {"return_attention_mask": False}}
    target = loss.encoder.preprocess(target_texts, processing_kwargs=options)
    assert "attention_mask" not in target
    original_keys = set(target)
    original_ids = target["input_ids"].clone()
    source = loss.encoder.preprocess(["hello", "hello world alpha"])
    actual = loss([source, target], labels=None)
    # With no mask, token IDs cannot disambiguate active EOS and padding. Preserve the old route.
    logits = loss.decoder(
        input_ids=target["input_ids"][:, :-1],
        attention_mask=None,
        encoder_hidden_states=loss.encoder(source)["sentence_embedding"][:, None],
        encoder_attention_mask=None,
        use_cache=False,
    ).logits
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target["input_ids"][:, 1:].reshape(-1),
        ignore_index=loss.tokenizer_decoder.pad_token_id,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.isfinite(actual)
    torch.testing.assert_close(target["input_ids"], original_ids)
    assert set(target) == original_keys
    assert options == {"text": {"return_attention_mask": False}}
    assert "attention_mask" in loss.encoder.preprocess(target_texts)
    actual_grad = torch.autograd.grad(actual, loss.decoder.get_output_embeddings().weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, loss.decoder.get_output_embeddings().weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_tsdae_uses_actual_mask_for_public_per_call_left_padding(local_tsdae):
    loss = local_tsdae()
    options = {"text": {"padding_side": "left"}}
    source = loss.encoder.preprocess(["hello", "hello world alpha"], processing_kwargs=options)
    target = loss.encoder.preprocess(["alpha", "alpha beta world"], processing_kwargs=options)
    assert loss.tokenizer_encoder.padding_side == loss.tokenizer_decoder.padding_side == "right"
    assert source["attention_mask"][0, 0] == target["attention_mask"][0, 0] == 0
    actual = loss([source, target], labels=None)
    logits = loss.decoder(
        input_ids=target["input_ids"][:, :-1],
        attention_mask=target["attention_mask"][:, :-1],
        encoder_hidden_states=loss.encoder(source)["sentence_embedding"][:, None],
        encoder_attention_mask=None,
        use_cache=False,
    ).logits
    positions = [
        (row, column)
        for row in range(len(target["input_ids"]))
        for column in range(1, target["input_ids"].shape[1])
        if target["attention_mask"][row, column] and target["attention_mask"][row, column - 1]
    ]
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[row, column - 1] for row, column in positions]),
        torch.stack([target["input_ids"][row, column] for row, column in positions]),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert options == {"text": {"padding_side": "left"}}
    assert loss.tokenizer_encoder.padding_side == "right"


def test_tsdae_masks_internal_target_attention_gaps(local_tsdae):
    loss = local_tsdae()
    source = loss.encoder.preprocess(["hello world alpha"])
    target = loss.encoder.preprocess(["alpha beta world hello"])
    target["attention_mask"][0, 2] = 0
    actual = loss([source, target], labels=None)
    logits = loss.decoder(
        input_ids=target["input_ids"][:, :-1],
        attention_mask=target["attention_mask"][:, :-1],
        encoder_hidden_states=loss.encoder(source)["sentence_embedding"][:, None],
        encoder_attention_mask=None,
        use_cache=False,
    ).logits
    columns = [
        column
        for column in range(1, target["input_ids"].shape[1])
        if target["attention_mask"][0, column] and target["attention_mask"][0, column - 1]
    ]
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, column - 1] for column in columns]),
        torch.stack([target["input_ids"][0, column] for column in columns]),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
