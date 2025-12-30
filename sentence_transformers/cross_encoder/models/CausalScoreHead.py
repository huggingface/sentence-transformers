from __future__ import annotations

import torch

from sentence_transformers.base.models import Module


class CausalScoreHead(Module):
    # TODO: Documentation
    config_keys = ["true_token_id", "false_token_id"]

    def __init__(self, true_token_id: int, false_token_id: int | None = None):
        super().__init__()
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.num_labels = 1

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        is_padding_left = torch.all(features["attention_mask"][:, -1]).item()

        logits = features["causal_logits"]
        token_indices = (
            torch.tensor([self.true_token_id])
            if self.false_token_id is None
            else torch.tensor([self.true_token_id, self.false_token_id])
        )
        # Get logits for the true/false token(s)
        if is_padding_left:
            logits = logits[:, -1, token_indices]
        else:
            last_token_indices = features["attention_mask"].sum(1) - 1
            logits = logits[:, :, token_indices]
            logits = logits[torch.arange(logits.size(0)), last_token_indices, :]

        # If only true_token_id is provided, return its logit as score. Otherwise, return the difference between true
        # and false logits.
        if self.false_token_id is None:
            scores = logits[:, 0]
        else:
            scores = logits[:, 0] - logits[:, 1]
        # TODO: Not sure if this unsqueeze is necessary
        # scores = scores.unsqueeze(1)
        features["scores"] = scores
        return features

    def save(self, output_path) -> None:
        self.save_config(output_path)
