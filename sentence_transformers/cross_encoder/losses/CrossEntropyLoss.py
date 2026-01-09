from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder.model import CrossEncoder


class CrossEntropyLoss(nn.Module):
    def __init__(self, model: CrossEncoder, activation_fn: nn.Module = nn.Identity(), **kwargs) -> None:
        """
        Computes the Cross Entropy Loss for a CrossEncoder model. This loss is used to train a model to predict the
        correct class label for a given pair of sentences. The number of classes should be equal to the number of model
        output labels.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss. Defaults to :class:`~torch.nn.Identity`.
            **kwargs: Additional keyword arguments passed to the underlying :class:`torch.nn.CrossEntropyLoss`.

        References:
            - :class:`torch.nn.CrossEntropyLoss`
            - `Cross Encoder > Training Examples > Natural Language Inference <../../../examples/cross_encoder/training/nli/README.html>`_

        Requirements:
            1. Your model can be initialized with `num_labels > 1` to predict multiple classes.
            2. The number of dataset classes should be equal to the number of model output labels (`model.num_labels`).

        Inputs:
            +-------------------------------------------------+--------+-------------------------------+
            | Texts                                           | Labels | Number of Model Output Labels |
            +=================================================+========+===============================+
            | (sentence_A, sentence_B) pairs                  | class  | `num_classes`                 |
            +-------------------------------------------------+--------+-------------------------------+

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base", num_labels=2)
                train_dataset = Dataset.from_dict({
                    "sentence1": ["How can I be a good geologist?", "What is the capital of France?"],
                    "sentence2": ["What should I do to be a great geologist?", "What is the capital of Germany?"],
                    "label": [1, 0],  # 1: duplicate, 0: not duplicate
                })
                loss = losses.CrossEntropyLoss(model)

                trainer = CrossEncoderTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn
        self.ce_loss = nn.CrossEntropyLoss(**kwargs)

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"CrossEntropyLoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]
        logits = self.activation_fn(logits)
        loss = self.ce_loss(logits, labels)
        return loss
