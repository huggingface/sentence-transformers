from __future__ import annotations

import torch

from sentence_transformers.base.modules import Module


class LogitScore(Module):
    """Converts language model logits into a relevance score for reranking.

    Extracts the logits at the last token position and computes a score based on specific
    vocabulary token IDs. If only ``true_token_id`` is provided, the score is the logit for
    that token. If ``false_token_id`` is also provided, the score is the log-odds:
    ``logit[true_token_id] - logit[false_token_id]``.

    This module is used as the post-processing step in a :class:`~sentence_transformers.cross_encoder.model.CrossEncoder`
    backed by a causal language model (e.g. Qwen, Llama).

    Args:
        true_token_id: Vocabulary ID of the token representing a positive/relevant match
            (e.g. ``"yes"`` or ``"1"``).
        false_token_id: Vocabulary ID of the token representing a negative/irrelevant match
            (e.g. ``"no"`` or ``"0"``). If ``None``, the score is the raw logit for
            ``true_token_id`` only.
        module_input_name: The key in the features dictionary to read logits from.
            Defaults to ``"causal_logits"``.
    """

    config_keys = ["true_token_id", "false_token_id", "module_input_name"]

    def __init__(
        self,
        true_token_id: int,
        false_token_id: int | None = None,
        module_input_name: str = "causal_logits",
    ):
        super().__init__()
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.module_input_name = module_input_name

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # Left padding is enforced by Transformer, so the last position is always a real token.
        # With logits_to_keep=1, causal_logits has shape (batch_size, 1, vocab_size), which we
        # convert to (batch_size, vocab_size).
        logits = features[self.module_input_name][:, -1]

        if self.false_token_id is None:
            scores = logits[:, self.true_token_id]
        else:
            scores = logits[:, self.true_token_id] - logits[:, self.false_token_id]

        features["scores"] = scores.unsqueeze(1)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
