from __future__ import annotations

from collections.abc import Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn

from sentence_transformers.models.Module import Module
from sentence_transformers.util import fullname, import_from_string


class Dense(Module):
    """
    Feed-forward function with activation function.

    This layer takes a fixed-sized sentence embedding and passes it through a feed-forward layer. Can be used to generate deep averaging networks (DAN).

    Args:
        in_features: Size of the input dimension
        out_features: Output size
        bias: Add a bias vector
        activation_function: Pytorch activation function applied on
            output
        init_weight: Initial value for the matrix of the linear layer
        init_bias: Initial value for the bias of the linear layer
    """

    config_keys: list[str] = [
        "in_features",
        "out_features",
        "bias",
        "activation_function",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: Callable[[Tensor], Tensor] | None = nn.Tanh(),
        init_weight: Tensor | None = None,
        init_bias: Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = nn.Identity() if activation_function is None else activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: dict[str, Tensor]):
        features.update({"sentence_embedding": self.activation_function(self.linear(features["sentence_embedding"]))})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_function": fullname(self.activation_function),
        }

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    def __repr__(self):
        return f"Dense({self.get_config_dict()})"

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        config["activation_function"] = import_from_string(config["activation_function"])()
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
