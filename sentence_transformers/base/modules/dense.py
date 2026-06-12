from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from torch import Tensor, nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util import fullname, import_from_string

logger = logging.getLogger(__name__)


class Dense(Module):
    """Applies a linear transformation with an optional activation function.

    Passes the embedding through a feed-forward layer (``nn.Linear`` + activation), useful for
    dimensionality reduction or projecting embeddings into a different space.

    Args:
        in_features: Size of the input dimension.
        out_features: Size of the output dimension.
        bias: Whether to include a bias vector in the linear layer.
        activation_function: Activation function applied after the linear layer.
            If ``None``, uses ``nn.Identity()``. Defaults to ``nn.Tanh()``.
        init_weight: Initial value for the weight matrix of the linear layer.
        init_bias: Initial value for the bias vector of the linear layer.
        module_input_name: The key in the features dictionary to read the input from.
            Defaults to ``"sentence_embedding"``.
        module_output_name: The key in the features dictionary to store the output in.
            If ``None``, uses the same key as ``module_input_name``.
        use_residual: If True, add a residual connection from the input to the projected output.
            When ``in_features != out_features``, the residual is projected through a separate
            ``nn.Linear(in_features, out_features, bias=False)``. Defaults to False.
    """

    config_keys: list[str] = [
        "in_features",
        "out_features",
        "bias",
        "activation_function",
        "module_input_name",
        "module_output_name",
        "use_residual",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function: Callable[[Tensor], Tensor] | None = nn.Tanh(),
        init_weight: Tensor | None = None,
        init_bias: Tensor | None = None,
        module_input_name: str = "sentence_embedding",
        module_output_name: str | None = None,
        use_residual: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = nn.Identity() if activation_function is None else activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.module_input_name = module_input_name
        self.module_output_name = module_output_name if module_output_name is not None else module_input_name
        self.use_residual = use_residual
        if use_residual and in_features != out_features:
            self.residual = nn.Linear(in_features, out_features, bias=False)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None and bias:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: dict[str, Tensor]):
        x = features[self.module_input_name]
        out = self.activation_function(self.linear(x))
        if self.use_residual:
            residual = x if self.in_features == self.out_features else self.residual(x)
            out = out + residual
        features[self.module_output_name] = out
        return features

    def get_embedding_dimension(self) -> int:
        return self.out_features

    def get_config_dict(self):
        config = super().get_config_dict()
        config["activation_function"] = fullname(self.activation_function)
        return config

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
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
        if "activation_function" in config:
            if trust_remote_code or config["activation_function"].startswith("torch."):
                config["activation_function"] = import_from_string(config["activation_function"])()
            else:
                logger.warning(
                    f"Activation function path '{config['activation_function']}' is not trusted, "
                    "falling back to the default activation function (Tanh). "
                    "Please load the model with `trust_remote_code=True` to allow loading custom activation "
                    "functions via the configuration."
                )
                del config["activation_function"]
        # Drop config keys this constructor doesn't accept, so a newer/foreign save (e.g. a PyLate `Dense`)
        # with an extra parameter loads instead of crashing in `cls(**config)`. No-op for well-formed ST saves.
        accepted_params = set(inspect.signature(cls).parameters)
        unexpected_keys = set(config) - accepted_params
        if unexpected_keys:
            logger.warning(
                f"Ignoring unrecognized {cls.__name__} config key(s) {sorted(unexpected_keys)} from "
                f"{model_name_or_path!r}: not constructor parameters of {cls.__name__}."
            )
            config = {key: value for key, value in config.items() if key in accepted_params}
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
