from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from sentence_transformers.backend.utils import save_or_push_to_hub_model
from sentence_transformers.util import disable_datasets_caching, is_datasets_available

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.intel import OVQuantizationConfig
    except ImportError:
        pass
    try:
        from optimum.onnxruntime.configuration import QuantizationConfig
    except ImportError:
        pass


def export_dynamic_quantized_onnx_model(
    model: SentenceTransformer | SparseEncoder | CrossEncoder,
    quantization_config: QuantizationConfig | Literal["arm64", "avx2", "avx512", "avx512_vnni"],
    model_name_or_path: str,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str | None = None,
) -> None:
    """
    Export a quantized ONNX model from a SentenceTransformer, SparseEncoder, or CrossEncoder model.

    This function applies dynamic quantization, i.e. without a calibration dataset.
    Each of the default quantization configurations quantize the model to int8, allowing
    for faster inference on CPUs, but are likely slower on GPUs.

    See the following pages for more information & benchmarks:

    - `Sentence Transformer > Usage > Speeding up Inference <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>`_
    - `Cross Encoder > Usage > Speeding up Inference <https://sbert.net/docs/cross_encoder/usage/efficiency.html>`_

    Args:
        model (SentenceTransformer | SparseEncoder | CrossEncoder): The SentenceTransformer, SparseEncoder,
            or CrossEncoder model to be quantized. Must be loaded with `backend="onnx"`.
        quantization_config (QuantizationConfig): The quantization configuration.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str | None, optional): The suffix to add to the quantized model file name. Defaults to None.

    Raises:
        ImportError: If the required packages `optimum` and `onnxruntime` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer, SparseEncoder, or CrossEncoder
            model loaded with `backend="onnx"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.onnxruntime import ORTModel, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
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
    quantizer = ORTQuantizer.from_pretrained(ort_model)

    if isinstance(quantization_config, str):
        if quantization_config not in ["arm64", "avx2", "avx512", "avx512_vnni"]:
            raise ValueError(
                "quantization_config must be an QuantizationConfig instance or one of 'arm64', 'avx2', 'avx512', or 'avx512_vnni'."
            )

        quantization_config_name = quantization_config[:]
        quantization_config = getattr(AutoQuantizationConfig, quantization_config)(is_static=False)
        file_suffix = file_suffix or f"{quantization_config.weights_dtype.name.lower()}_{quantization_config_name}"

    if file_suffix is None:
        file_suffix = f"{quantization_config.weights_dtype.name.lower()}_quantized"

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(quantization_config, save_dir, file_suffix=file_suffix),
        export_function_name="export_dynamic_quantized_onnx_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="onnx",
        model=model,
    )


def export_static_quantized_openvino_model(
    model: SentenceTransformer | SparseEncoder | CrossEncoder,
    quantization_config: OVQuantizationConfig | dict | None,
    model_name_or_path: str,
    dataset_name: str | None = None,
    dataset_config_name: str | None = None,
    dataset_split: str | None = None,
    column_name: str | None = None,
    push_to_hub: bool = False,
    create_pr: bool = False,
    file_suffix: str = "qint8_quantized",
) -> None:
    """
    Export a quantized OpenVINO model from a SentenceTransformer, SparseEncoder, or CrossEncoder model.

    This function applies Post-Training Static Quantization (PTQ) using a calibration dataset, which calibrates
    quantization constants without requiring model retraining. Each default quantization configuration converts
    the model to int8 precision, enabling faster inference while maintaining accuracy.

    See the following pages for more information & benchmarks:

    - `Sentence Transformer > Usage > Speeding up Inference <https://sbert.net/docs/sentence_transformer/usage/efficiency.html>`_
    - `Cross Encoder > Usage > Speeding up Inference <https://sbert.net/docs/cross_encoder/usage/efficiency.html>`_

    Args:
        model (SentenceTransformer | SparseEncoder | CrossEncoder): The SentenceTransformer, SparseEncoder,
            or CrossEncoder model to be quantized. Must be loaded with `backend="openvino"`.
        quantization_config (OVQuantizationConfig | dict | None): The quantization configuration. If None, default values are used.
        model_name_or_path (str): The path or Hugging Face Hub repository name where the quantized model will be saved.
        dataset_name(str, optional): The name of the dataset to load for calibration.
            If not specified, the `sst2` subset of the `glue` dataset will be used by default.
        dataset_config_name (str, optional): The specific configuration of the dataset to load.
        dataset_split (str, optional): The split of the dataset to load (e.g., 'train', 'test'). Defaults to None.
        column_name (str, optional): The column name in the dataset to use for calibration. Defaults to None.
        push_to_hub (bool, optional): Whether to push the quantized model to the Hugging Face Hub. Defaults to False.
        create_pr (bool, optional): Whether to create a pull request when pushing to the Hugging Face Hub. Defaults to False.
        file_suffix (str, optional): The suffix to add to the quantized model file name. Defaults to `qint8_quantized`.

    Raises:
        ImportError: If the required packages `optimum` and `openvino` are not installed.
        ValueError: If the provided model is not a valid SentenceTransformer, SparseEncoder, or CrossEncoder model
            loaded with `backend="openvino"`.
        ValueError: If the provided quantization_config is not valid.

    Returns:
        None
    """
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder

    try:
        from optimum.intel.openvino import (
            OVConfig,
            OVQuantizationConfig,
            OVQuantizer,
        )
        from optimum.intel.openvino.modeling import OVModel
    except ImportError:
        raise ImportError(
            "Please install datasets, optimum-intel and openvino to use this function. "
            "You can install them with pip: `pip install datasets sentence-transformers[openvino]`"
        )
    if not is_datasets_available():
        raise ImportError(
            "Please install datasets to use this function. You can install it with pip: `pip install datasets`"
        )

    viable_st_model = (
        isinstance(model, SentenceTransformer)
        and len(model)
        and hasattr(model[0], "auto_model")
        and isinstance(model[0].auto_model, OVModel)
    )
    viable_se_model = (
        isinstance(model, SparseEncoder)
        and len(model)
        and hasattr(model[0], "auto_model")
        and isinstance(model[0].auto_model, OVModel)
    )
    viable_ce_model = isinstance(model, CrossEncoder) and isinstance(model.model, OVModel)
    if not (viable_st_model or viable_ce_model or viable_se_model):
        raise ValueError(
            'The model must be a Transformer-based SentenceTransformer, SparseEncoder, or CrossEncoder model loaded with `backend="openvino"`.'
        )

    if viable_st_model or viable_se_model:
        ov_model: OVModel = model[0].auto_model
    else:
        ov_model: OVModel = model.model

    if quantization_config is None:
        quantization_config = OVQuantizationConfig()

    ov_config = OVConfig(quantization_config=quantization_config)
    quantizer = OVQuantizer.from_pretrained(ov_model)

    if any(param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]) and not all(
        param is not None for param in [dataset_name, dataset_config_name, dataset_split, column_name]
    ):
        raise ValueError(
            "Either specify all of `dataset_name`, `dataset_config_name`, `dataset_split`, and `column_name`, or leave them all unspecified."
        )

    def preprocess_function(examples):
        return model.tokenizer(examples, padding="max_length", max_length=384, truncation=True)

    dataset_name = dataset_name if dataset_name is not None else "glue"
    dataset_config_name = dataset_config_name if dataset_config_name is not None else "sst2"
    dataset_split = dataset_split if dataset_split is not None else "train"
    column_name = column_name if column_name is not None else "sentence"
    with disable_datasets_caching():
        calibration_dataset = quantizer.get_calibration_dataset(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            preprocess_function=lambda examples: preprocess_function(examples[column_name]),
            num_samples=quantization_config.num_samples if quantization_config is not None else 300,
            dataset_split=dataset_split,
        )

    save_or_push_to_hub_model(
        export_function=lambda save_dir: quantizer.quantize(
            calibration_dataset, save_directory=save_dir, ov_config=ov_config
        ),
        export_function_name="export_static_quantized_openvino_model",
        config=quantization_config,
        model_name_or_path=model_name_or_path,
        push_to_hub=push_to_hub,
        create_pr=create_pr,
        file_suffix=file_suffix,
        backend="openvino",
        model=model,
    )
