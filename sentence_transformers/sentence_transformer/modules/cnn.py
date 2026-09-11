from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs


class CNN(Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings.

    Convolutions start at each sequence's first unmasked token. Outputs retain
    the common length of all branches, with aligned masks and prompt boundaries.
    """

    config_keys: list[str] = ["in_embedding_dimension", "out_channels", "kernel_sizes", "stride_sizes"]
    config_file_name: str = "cnn_config.json"
    config_key_renames = {"in_word_embedding_dimension": "in_embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    def __init__(
        self,
        in_embedding_dimension: int,
        out_channels: int = 256,
        kernel_sizes: list[int] = [1, 3, 5],
        stride_sizes: list[int] | None = None,
    ):
        nn.Module.__init__(self)
        self.in_embedding_dimension = in_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        if stride_sizes is None:
            stride_sizes = [1] * len(kernel_sizes)
        self.stride_sizes = stride_sizes

        self.embeddings_dimension = out_channels * len(kernel_sizes)
        self.convs = nn.ModuleList()

        for kernel_size, stride in zip(kernel_sizes, stride_sizes):
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(
                in_channels=in_embedding_dimension,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_size,
            )
            self.convs.append(conv)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        input_mask = features.get("attention_mask")
        if input_mask is not None:
            input_mask = input_mask.to(device=token_embeddings.device)
            # Anchor each row at its first token, independently of batch padding.
            positions = torch.arange(token_embeddings.size(1), device=token_embeddings.device)
            positions = positions.unsqueeze(0) + input_mask.to(torch.int32).argmax(dim=1, keepdim=True)
            indices = positions.clamp(max=token_embeddings.size(1) - 1)
            token_embeddings = token_embeddings.gather(1, indices.unsqueeze(-1).expand_as(token_embeddings))
            input_mask = input_mask.gather(1, indices).masked_fill(positions >= input_mask.size(1), 0)
            token_embeddings.masked_fill_(input_mask.unsqueeze(-1) == 0, 0)
        elif "sentence_lengths" in features:
            positions = torch.arange(token_embeddings.size(1), device=token_embeddings.device)
            input_mask = positions.unsqueeze(0) < features["sentence_lengths"].to(token_embeddings.device).unsqueeze(1)
            token_embeddings = token_embeddings.masked_fill(~input_mask.unsqueeze(-1), 0)

        token_embeddings = token_embeddings.transpose(1, -1)
        vectors = [conv(token_embeddings) for conv in self.convs]
        output_length = min(vector.size(-1) for vector in vectors)
        out = torch.cat([vector[:, :, :output_length] for vector in vectors], dim=1).transpose(1, -1)

        # Do not replace the encoder's input mask when losses replay its features.
        features = features.copy()
        if input_mask is not None:
            attention_mask = None
            geometries = {(conv.stride[0], conv.kernel_size[0] % 2 == 0) for conv in self.convs}
            for stride, even_kernel in geometries:
                branch_mask = input_mask[:, ::stride][:, :output_length]
                if even_kernel:
                    branch_mask = branch_mask.masked_fill(input_mask[:, 1::stride][:, :output_length] == 0, 0)
                attention_mask = (
                    branch_mask if attention_mask is None else attention_mask.masked_fill(branch_mask == 0, 0)
                )
            features["attention_mask"] = attention_mask
            if "sentence_lengths" in features:
                features["sentence_lengths"] = attention_mask.sum(dim=1).to(features["sentence_lengths"])

        if "prompt_length" in features:
            # Exclude an output if any concatenated branch's anchor is in the prompt.
            stride = min(conv.stride[0] for conv in self.convs)
            prompt_length = (features["prompt_length"] + stride - 1) // stride
            features["prompt_length"] = (
                prompt_length.clamp(max=output_length)
                if isinstance(prompt_length, torch.Tensor)
                else min(prompt_length, output_length)
            )

        features["token_embeddings"] = out
        return features

    def get_embedding_dimension(self) -> int:
        return self.embeddings_dimension

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
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
