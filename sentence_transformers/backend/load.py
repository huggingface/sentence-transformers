from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers.configuration_utils import PretrainedConfig

from sentence_transformers.backend.utils import _save_pretrained_wrapper, backend_should_export, backend_warn_to_save

logger = logging.getLogger(__name__)


def load_onnx_model(model_name_or_path: str, config: PretrainedConfig, task_name: str, **model_kwargs):
    """
    Load and perhaps export an ONNX model using the Optimum library.

    Args:
        model_name_or_path (str): The model name on Hugging Face (e.g. 'naver/splade-cocondenser-ensembledistil')
            or the path to a local model directory.
        config (PretrainedConfig): The model configuration.
        task_name (str): The task name for the model (e.g. 'feature-extraction', 'fill-mask', 'sequence-classification').
        model_kwargs (dict): Additional keyword arguments for the model loading.
    """
    try:
        import onnxruntime as ort
        from optimum.onnxruntime import (
            ONNX_WEIGHTS_NAME,
            ORTModelForFeatureExtraction,
            ORTModelForMaskedLM,
            ORTModelForSequenceClassification,
        )

        # Map task names to their corresponding model classes
        task_to_model_mapping = {
            "feature-extraction": ORTModelForFeatureExtraction,
            "fill-mask": ORTModelForMaskedLM,
            "sequence-classification": ORTModelForSequenceClassification,
        }

        # Get the appropriate model class based on the task name
        if task_name not in task_to_model_mapping:
            supported_tasks = ", ".join(task_to_model_mapping.keys())
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {supported_tasks}")

        model_cls = task_to_model_mapping[task_name]
    except ModuleNotFoundError:
        raise Exception(
            "Using the ONNX backend requires installing Optimum and ONNX Runtime. "
            "You can install them with pip: `pip install sentence-transformers[onnx]` "
            "or `pip install sentence-transformers[onnx-gpu]`"
        )

    # Default to the highest priority available provider if not specified
    # E.g. Tensorrt > CUDA > CPU
    model_kwargs["provider"] = model_kwargs.pop("provider", ort.get_available_providers()[0])

    load_path = Path(model_name_or_path)
    is_local = load_path.exists()
    backend_name = "ONNX"
    target_file_glob = "*.onnx"

    # Determine whether the model should be exported or whether we can load it directly
    export, model_kwargs = backend_should_export(
        load_path, is_local, model_kwargs, ONNX_WEIGHTS_NAME, target_file_glob, backend_name
    )

    # If we're exporting, then there's no need for a file_name to load the model from
    if export:
        model_kwargs.pop("file_name", None)

    # Either load an exported model, or export the model to ONNX
    model = model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        export=export,
        **model_kwargs,
    )

    # Wrap the save_pretrained method to save the model in the correct subfolder
    model._save_pretrained = _save_pretrained_wrapper(model._save_pretrained, subfolder="onnx")

    # Warn the user to save the model if they haven't already
    if export:
        backend_warn_to_save(model_name_or_path, is_local, backend_name)

    return model


def load_openvino_model(model_name_or_path: str, config: PretrainedConfig, task_name: str, **model_kwargs):
    """
    Load and perhaps export an OpenVINO model using the Optimum library.

    Args:
        model_name_or_path (str): The model name on Hugging Face (e.g. 'naver/splade-cocondenser-ensembledistil')
            or the path to a local model directory.
        config (PretrainedConfig): The model configuration.
        task_name (str): The task name for the model (e.g. 'feature-extraction', 'fill-mask', 'sequence-classification').
        model_kwargs (dict): Additional keyword arguments for the model loading.
    """
    try:
        from optimum.intel.openvino import (
            OV_XML_FILE_NAME,
            OVModelForFeatureExtraction,
            OVModelForMaskedLM,
            OVModelForSequenceClassification,
        )

        # Map task names to their corresponding model classes
        task_to_model_mapping = {
            "feature-extraction": OVModelForFeatureExtraction,
            "fill-mask": OVModelForMaskedLM,
            "sequence-classification": OVModelForSequenceClassification,
        }

        # Get the appropriate model class based on the task name
        if task_name not in task_to_model_mapping:
            supported_tasks = ", ".join(task_to_model_mapping.keys())
            raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {supported_tasks}")

        model_cls = task_to_model_mapping[task_name]
    except ModuleNotFoundError:
        raise Exception(
            "Using the OpenVINO backend requires installing Optimum and OpenVINO. "
            "You can install them with pip: `pip install sentence-transformers[openvino]`"
        )

    load_path = Path(model_name_or_path)
    is_local = load_path.exists()
    backend_name = "OpenVINO"
    target_file_glob = "openvino*.xml"

    # Determine whether the model should be exported or whether we can load it directly
    export, model_kwargs = backend_should_export(
        load_path, is_local, model_kwargs, OV_XML_FILE_NAME, target_file_glob, backend_name
    )

    # If we're exporting, then there's no need for a file_name to load the model from
    if export:
        model_kwargs.pop("file_name", None)

    # ov_config can be either a dictionary, or point to a json file with an OpenVINO config,
    # at which point we load the config dict from the file
    if "ov_config" in model_kwargs:
        ov_config = model_kwargs["ov_config"]
        if not isinstance(ov_config, dict):
            if not Path(ov_config).exists():
                raise ValueError(
                    "ov_config should be a dictionary or a path to a .json file containing an OpenVINO config"
                )
            with open(ov_config, encoding="utf-8") as f:
                model_kwargs["ov_config"] = json.load(f)
    else:
        model_kwargs["ov_config"] = {}

    # Either load an exported model, or export the model to OpenVINO
    model = model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        export=export,
        **model_kwargs,
    )

    # Wrap the save_pretrained method to save the model in the correct subfolder
    model._save_pretrained = _save_pretrained_wrapper(model._save_pretrained, subfolder="openvino")

    # Warn the user to save the model if they haven't already
    if export:
        backend_warn_to_save(model_name_or_path, is_local, backend_name)

    return model
