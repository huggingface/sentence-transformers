from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.models.Module import Module


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class SparseAutoEncoder(Module):
    """
    This module implements the Sparse AutoEncoder architecture based on the paper:
    Beyond Matryoshka: Revisiting Sparse Coding for Adaptive Representation, https://huggingface.co/papers/2503.01776

    This module transforms dense embeddings into sparse representations by:

    1. Applying a multi-layer feed-forward network
    2. Applying top-k sparsification to keep only the largest values
    3. Supporting auxiliary losses for training stability (via k_aux parameter)

    Args:
        input_dim: Dimension of the input embeddings.
        hidden_dim: Dimension of the hidden layers. Defaults to 512.
        k: Number of top values to keep in the final sparse representation. Defaults to 8.
        k_aux: Number of top values to keep for auxiliary loss calculation. Defaults to 512.
        normalize: Whether to apply layer normalization to the input embeddings. Defaults to False.
        dead_threshold: Threshold for dead neurons. Neurons with non-zero activations below this threshold are considered dead. Defaults to 30.
    """

    config_keys = ["input_dim", "hidden_dim", "k", "k_aux", "normalize", "dead_threshold"]

    forward_kwargs = {"max_active_dims"}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        k: int = 8,
        k_aux: int = 512,
        normalize: bool = False,
        dead_threshold: int = 30,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dead_threshold = dead_threshold
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder: nn.Module = nn.Linear(input_dim, hidden_dim, bias=False)
        self.latent_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder: TiedTranspose = TiedTranspose(self.encoder)
        self.k = k
        self.k_aux = k_aux
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, input_dim])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, hidden_dim])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(x, self.encoder.weight, self.latent_bias)
        return latents_pre_act

    def LN(self, x: torch.Tensor, eps: float = 1e-5):
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = self.LN(x)
        return x, dict(mu=mu, std=std)

    def top_k(self, x: torch.Tensor, k: int | None = None, compute_aux: bool = True) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, input_dim])
        :return: autoencoder latents (shape: [batch, hidden_dim])
        """
        if k is None:
            k = self.k
        topk = torch.topk(x, k=k, dim=-1)
        z_topk = torch.zeros_like(x)
        z_topk.scatter_(-1, topk.indices, topk.values)
        latents_k = F.relu(z_topk)
        ## set num nonzero stat ##
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            topk.indices.reshape(-1),
            (topk.values > 1e-5).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1
        ## end stats ##

        latents_auxk = None
        if self.k_aux and compute_aux:
            aux_topk = torch.topk(
                input=self.auxk_mask_fn(x),
                k=self.k_aux,
            )
            z_auxk = torch.zeros_like(x)
            z_auxk.scatter_(-1, aux_topk.indices, aux_topk.values)
            latents_auxk = F.relu(z_auxk)
        return latents_k, latents_auxk

    def decode(self, latents: torch.Tensor, info=None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, hidden_dim])
        :return: reconstructed data (shape: [batch, n_inputs])
        """

        ret = self.decoder(latents) + self.pre_bias

        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(
        self, features: dict[str, torch.Tensor], max_active_dims: int | None = None
    ) -> dict[str, torch.Tensor]:
        k = max_active_dims if max_active_dims is not None else self.k
        x = features["sentence_embedding"]

        # If the model is in inference mode, we don't need to e.g. compute the 4k, auxk, or apply the decoder
        if torch.is_inference_mode_enabled():
            x, info = self.preprocess(x)
            latents_pre_act = self.encode_pre_act(x)
            latents_k, _ = self.top_k(latents_pre_act, k, compute_aux=False)
            features["sentence_embedding"] = latents_k
            return features

        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)

        latents_k, latents_auxk = self.top_k(latents_pre_act, k)
        latents_4k, _ = self.top_k(latents_pre_act, 4 * k)

        recons_k = self.decode(latents_k, info)
        recons_4k = self.decode(latents_4k, info)

        recons_aux = self.decode(latents_auxk, info)

        # Update the features dictionary
        features.update(
            {
                "sentence_embedding_backbone": x,
                "sentence_embedding_encoded": latents_pre_act,
                "sentence_embedding_encoded_4k": latents_4k,
                "auxiliary_embedding": latents_auxk,
                "decoded_embedding_k": recons_k,
                "decoded_embedding_4k": recons_4k,
                "decoded_embedding_aux": recons_aux,
                "decoded_embedding_k_pre_bias": recons_k - self.pre_bias,
            }
        )
        features["sentence_embedding"] = latents_k
        return features

    def save(self, output_path, safe_serialization: bool = True) -> None:
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

    def __repr__(self):
        return f"SparseAutoEncoder({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """
        Get the dimension of the sentence embedding. Warning: the number of non-zero elements in the embedding is only k out of the hidden_dim.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.hidden_dim
