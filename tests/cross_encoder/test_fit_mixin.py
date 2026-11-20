from __future__ import annotations

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.base.evaluation import BaseEvaluator


@pytest.mark.parametrize("dict_scores", [False, True])
def test_old_fit_callback_receives_primary_score(tmp_path, dict_scores: bool) -> None:
    checkpoint = tmp_path / "checkpoint"
    BertForSequenceClassification(
        BertConfig(
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_labels=1,
        )
    ).save_pretrained(checkpoint)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", model_max_length=16
    ).save_pretrained(checkpoint)
    model = CrossEncoder(str(checkpoint), device="cpu")
    model.model_card_data.generate_widget_examples = False

    class ScoreEvaluator(BaseEvaluator):
        def __init__(self):
            super().__init__()
            self.primary_metric = "score"

        def __call__(self, model, output_path=None, epoch=-1, steps=-1):
            return {"score": float(steps), "other": 100.0} if dict_scores else float(steps)

    callback_results = []

    def callback(score: float, epoch: int, steps: int) -> None:
        callback_results.append((f"{score:.1f}", epoch, steps))

    output_path = tmp_path / "output"
    model.old_fit(
        DataLoader([InputExample(texts=["a", "b"], label=1.0)] * 2, batch_size=1),
        evaluator=ScoreEvaluator(),
        epochs=1,
        evaluation_steps=1,
        warmup_steps=0,
        output_path=str(output_path),
        callback=callback,
        show_progress_bar=False,
    )

    assert callback_results == [("1.0", 0, 1), ("2.0", 0, 2), ("-1.0", 0, -1)]
    assert model.best_score == 2.0
    assert (output_path / "config.json").exists()
