from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from sentence_transformers.backend.utils import save_or_push_to_hub_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.onnxruntime.configuration import OptimizationConfig
    except ImportError:
        pass


def export_optimized_onnx_model(
    model: SentenceTransformer | SparseEncoder | CrossEncoder,
    optimization_config: OptimizationConfig | Literal["O1", "O2", "O3", "O4"],
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
) -> None:
    """
    Export an optimized ONNX model from a SentenceTransformer, SparseEncoder, or CrossEncoder model.

    The O1-O4 optimization levels are defined by Optimum and are documented here:
    https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/optimization

    The optimization levels are:

    - O1: basic general optimizations.
    - O2: basic and extended general optimizations, transformers-specific fusions.
    - O3: same as O2 with GELU approximation.
    - O4: same as O3 with mixed precision (fp16, GPU-only)

    See the following pages for more information & benchmarks:

    - `Sentence Transformer > Usage > Speeding up Inference <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>`_
    - `Cross Encoder > Usage > Speeding up Inference <https://sbert.net/docs/cross_encoder/usage/efficiency.html>`_

    Args:
        model (SentenceTransformer | SparseEncoder | CrossEncoder): The SentenceTransformer, SparseEncoder,
            or CrossEncoder model to be optimized. Must be loaded with `backend="onnx"`.
        optimization_config (OptimizationConfig | Literal["O1", "O2", "O3", "O4"]): The optimization configuration or level.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the optimized model will be saved.
        push_to_hub (bool, optional): Whether to push the optimized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str | None, optional): The suffix to add to the optimized model file name. Defaults to None.

    Raises:
        ImportError: If the required packages `optimum` and `onnxruntime` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer, SparseEncoder, or CrossEncoder model loaded with `backend="onnx"`.
        ValueError: If the provided optimization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.onnxruntime import ORTModel, ORTOptimizer
        from optimum.onnxruntime.configuration import AutoOptimizationConfig
    except ImportError:
        raise ImportError(
            "Please install Optimum and ONNX Runtime to use this function. "
            "You can install them with pip: `pip install sentence-transformers[onnx]` "
            "or `pip install sentence-transformers[onnx-gpu]`"
        )

    viable_st_model = (
        isinstance(model, SentenceTransformer)
        and len(model)
        and hasattr(model[0], "auto_model")
        and isinstance(model[0].auto_model, ORTModel)
    )
    viable_se_model = (
        isinstance(model, SparseEncoder)
        and len(model)
        and hasattr(model[0], "auto_model")
        and isinstance(model[0].auto_model, ORTModel)
    )
    viable_ce_model = isinstance(model, CrossEncoder) and isinstance(model.model, ORTModel)
    if not (viable_st_model or viable_ce_model or viable_se_model):
        raise ValueError(
            'The model must be a Transformer-based SentenceTransformer, SparseEncoder, or CrossEncoder model loaded with `backend="onnx"`.'
        )

    if viable_st_model or viable_se_model:
        ort_model: ORTModel = model[0].auto_model
    else:
        ort_model: ORTModel = model.model
    optimizer = ORTOptimizer.from_pretrained(ort_model)

    if isinstance(optimization_config, str):
        if optimization_config not in AutoOptimizationConfig._LEVELS:
            raise ValueError(
                "optimization_config must be an OptimizationConfig instance or one of 'O1', 'O2', 'O3', 'O4'."
            )

        file_suffix = file_suffix or optimization_config
        optimization_config = getattr(AutoOptimizationConfig, optimization_config)()

    if file_suffix is None:
        file_suffix = "optimized"

    save_or_push_to_hub_model(
        export_function=lambda save_dir: optimizer.optimize(optimization_config, save_dir, file_suffix=file_suffix),
        export_function_name="export_optimized_onnx_model",
        config=optimization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="onnx",
        model=model,
    )
