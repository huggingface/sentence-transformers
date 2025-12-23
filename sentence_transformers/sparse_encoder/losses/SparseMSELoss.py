from __future__ import annotations

from sentence_transformers.sentence_transformer.losses.MSELoss import MSELoss
from sentence_transformers.sparse_encoder.model import SparseEncoder


class SparseMSELoss(MSELoss):
    def __init__(self, model: SparseEncoder) -> None:
        """
        Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
        is used when extending sentence embeddings to new languages as described in our publication
        Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation.

        Args:
            model: SparseEncoder

        Requirements:
            1. Usually uses a finetuned teacher M in a knowledge distillation setup

        Inputs:
            +-----------------------------------------+-----------------------------+
            | Texts                                   | Labels                      |
            +=========================================+=============================+
            | sentence                                | model sentence embeddings   |
            +-----------------------------------------+-----------------------------+
            | sentence_1, sentence_2, ..., sentence_N | model sentence embeddings   |
            +-----------------------------------------+-----------------------------+

        Relations:
            - :class:`SparseMarginMSELoss` is equivalent to this loss, but with a margin through a negative pair.

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "english": ["The first sentence", "The second sentence", "The third sentence", "The fourth sentence"],
                        "french": ["La première phrase", "La deuxième phrase", "La troisième phrase", "La quatrième phrase"],
                    }
                )


                def compute_labels(batch):
                    return {"label": teacher_model.encode(batch["english"], convert_to_sparse_tensor=False)}


                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SparseMSELoss(student_model)

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()

        """
        return super().__init__(model)
