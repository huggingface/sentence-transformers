from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder.model import CrossEncoder
from sentence_transformers.util import fullname


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module = nn.Identity(),
        pos_weight: Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Computes the Binary Cross Entropy Loss for a CrossEncoder model. This loss is used to train a model to predict
        a high logit for positive pairs and a low logit for negative pairs. The model should be initialized with
        ``num_labels = 1`` (a.k.a. the default) to predict one class.

        It has been used to train many of the strong `CrossEncoder MS MARCO Reranker models <https://huggingface.co/models?author=cross-encoder&search=marco>`_.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.model.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss. Defaults to :class:`~torch.nn.Identity`.
            pos_weight (Tensor, optional): A weight of positive examples. Must be a :class:`torch.Tensor` like ``torch.tensor(4)`` for a weight of 4. Defaults to None.
            **kwargs: Additional keyword arguments passed to the underlying :class:`torch.nn.BCEWithLogitsLoss`.

        References:
            - :class:`torch.nn.BCEWithLogitsLoss`
            - `Cross Encoder > Training Examples > Semantic Textual Similarity <../../../examples/cross_encoder/training/sts/README.html>`_
            - `Cross Encoder > Training Examples > Quora Duplicate Questions <../../../examples/cross_encoder/training/quora_duplicate_questions/README.html>`_
            - `Cross Encoder > Training Examples > MS MARCO <../../../examples/cross_encoder/training/ms_marco/README.html>`_
            - `Cross Encoder > Training Examples > Rerankers <../../../examples/cross_encoder/training/rerankers/README.html>`_

        Requirements:
            1. Your model must be initialized with `num_labels = 1` (a.k.a. the default) to predict one class.

        Inputs:
            +-------------------------------------------------+----------------------------------------+-------------------------------+
            | Texts                                           | Labels                                 | Number of Model Output Labels |
            +=================================================+========================================+===============================+
            | (anchor, positive/negative) pairs               | 1 if positive, 0 if negative           | 1                             |
            +-------------------------------------------------+----------------------------------------+-------------------------------+
            | (sentence_A, sentence_B) pairs                  | float similarity score between 0 and 1 | 1                             |
            +-------------------------------------------------+----------------------------------------+-------------------------------+

        Recommendations:
            - Use :class:`~sentence_transformers.util.hard_negatives.mine_hard_negatives` with ``output_format="labeled-pair"``
              to convert question-answer pairs to the ``(anchor, positive/negative) pairs`` format with labels as 1 or 0,
              using hard negatives.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                model = CrossEncoder("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What are pandas?"],
                    "response": ["Pandas are a kind of bear.", "Pandas are a kind of fish."],
                    "label": [1, 0],
                })
                loss = losses.BinaryCrossEntropyLoss(model)

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
        self.pos_weight = pos_weight
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, **kwargs)

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 1 output label, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        # inputs being a list of lists might be a problem - we might need to update that to a dictionary like ST/SE
        if len(inputs) != 2:
            raise ValueError(
                f"BinaryCrossEntropyLoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        """
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        breakpoint()
        logits = self.model(**tokens)[0].view(-1)
        """
        # NOTE: I don't think I can preserve backwards compatibility: the `self.model` call just has different
        # tokenizer (manageable), inputs (manageable), and outputs (not manageable) now.
        tokens = self.model.preprocess(pairs)
        tokens.to(self.model.device)
        outputs = self.model(tokens)
        logits = outputs["scores"].view(-1)
        logits = self.activation_fn(logits)
        loss = self.bce_with_logits_loss(logits, labels.float())
        return loss

    def get_config_dict(self):
        return {
            "activation_fn": fullname(self.activation_fn),
            "pos_weight": self.pos_weight if self.pos_weight is None else self.pos_weight.item(),
        }
