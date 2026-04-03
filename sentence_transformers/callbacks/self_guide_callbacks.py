from __future__ import annotations

import logging

from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class SelfGuideWarmupCallback(TrainerCallback):
    """Disables self-guide false-negative filtering during warmup, then enables it.

    This callback holds a reference to a loss module that has a ``self_guide_filtering_active``
    attribute and toggles it from ``False`` to ``True`` once the warmup phase is over.

    Args:
        loss: A loss module with a ``self_guide_filtering_active`` boolean attribute.
        warmup_ratio: Fraction of total training steps during which filtering is disabled.
            Defaults to 0.2 (i.e., filtering starts after the first 20% of training).
    """

    def __init__(self, loss, warmup_ratio: float = 0.2) -> None:
        super().__init__()
        self.loss = loss
        self.warmup_ratio = warmup_ratio
        self.warmup_steps: int | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.warmup_steps = int(state.max_steps * self.warmup_ratio)
        self.loss.self_guide_filtering_active = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.warmup_steps is None:
            return
        should_be_active = state.global_step >= self.warmup_steps
        if should_be_active != self.loss.self_guide_filtering_active:
            self.loss.self_guide_filtering_active = should_be_active

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs is not None:
            logs["self_guide_filtering_active"] = self.loss.self_guide_filtering_active
