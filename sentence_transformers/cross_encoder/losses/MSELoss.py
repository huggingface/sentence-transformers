from __future__ import annotations

from torch import Tensor, nn

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.util import fullname


class MSELoss(nn.Module):
    def __init__(self, model: CrossEncoder, activation_fn: nn.Module = nn.Identity(), **kwargs) -> None:
        """
        Computes the MSE loss between the computed query-passage score and a target query-passage score. This loss
        is used to distill a cross-encoder model from a teacher cross-encoder model or gold labels.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss.
            **kwargs: Additional keyword arguments passed to the underlying :class:`torch.nn.MSELoss`.

        .. note::

            Be mindful of the magnitude of both the labels and what the model produces. If the teacher model produces
            logits with Sigmoid to bound them to [0, 1], then you may wish to use a Sigmoid activation function in the loss.

        References:
            - Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation: https://huggingface.co/papers/2010.02666
            - `Cross Encoder > Training Examples > Distillation <../../../examples/cross_encoder/training/distillation/README.html>`_

        Requirements:
            1. Your model must be initialized with `num_labels = 1` (a.k.a. the default) to predict one class.
            2. Usually uses a finetuned CrossEncoder teacher M in a knowledge distillation setup.

        Inputs:
            +-----------------------------------------+-----------------------------+-------------------------------+
            | Texts                                   | Labels                      | Number of Model Output Labels |
            +=========================================+=============================+===============================+
            | (sentence_A, sentence_B) pairs          | similarity score            | 1                             |
            +-----------------------------------------+-----------------------------+-------------------------------+

        Relations:
            - :class:`MarginMSELoss` is similar to this loss, but with a margin through a negative pair.

        Example:
            ::

                from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, losses
                from datasets import Dataset

                student_model = CrossEncoder("microsoft/mpnet-base")
                teacher_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2")
                train_dataset = Dataset.from_dict({
                    "query": ["What are pandas?", "What is the capital of France?"],
                    "answer": ["Pandas are a kind of bear.", "The capital of France is Paris."],
                })

                def compute_labels(batch):
                    return {
                        "label": teacher_model.predict(list(zip(batch["query"], batch["answer"])))
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.MSELoss(student_model)

                trainer = CrossEncoderTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn
        self.loss_fct = nn.MSELoss(**kwargs)

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
        if len(inputs) != 2:
            raise ValueError(
                f"MSELoss expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0].view(-1)
        logits = self.activation_fn(logits)
        loss = self.loss_fct(logits, labels.float())
        return loss

    def get_config_dict(self):
        return {
            "activation_fn": fullname(self.activation_fn),
        }
