from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Literal, TypedDict, get_args, get_type_hints

import torch
from packaging.version import parse as parse_version
from tokenizers.normalizers import Lowercase, Sequence
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    BlenderbotConfig,
    BlenderbotSmallConfig,
    FeatureExtractionMixin,
    ImageProcessingMixin,
    LongT5Config,
    M2M100Config,
    MarianConfig,
    MT5Config,
    PegasusConfig,
    PegasusXConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    ProphetNetConfig,
    SwitchTransformersConfig,
    T5Config,
    UdopConfig,
    UMT5Config,
    WhisperConfig,
)
from transformers import __version__ as transformers_version
from transformers.utils import ModelOutput
from transformers.utils import logging as transformers_logging
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

from sentence_transformers.backend import load_onnx_model, load_openvino_model
from sentence_transformers.base.modality import InputFormatter, format_modality
from sentence_transformers.base.modality_types import (
    MODALITY_TO_PROCESSOR_ARG,
    MessageInput,
    Modality,
    PairInput,
    SingleInput,
)
from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.util.decorators import transformer_kwargs_decorator
from sentence_transformers.util.environment import suggest_extra_on_exception

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    from transformers import BaseVideoProcessor
except ImportError:

    class BaseVideoProcessor:
        pass


try:
    from transformers import T5Gemma2Config, T5Gemma2TextConfig
except ImportError:

    class T5Gemma2Config:
        pass

    class T5Gemma2TextConfig:
        pass


try:
    from transformers import T5GemmaConfig
except ImportError:

    class T5GemmaConfig:
        pass


try:
    from transformers import MoonshineConfig
except ImportError:

    class MoonshineConfig:
        pass


try:
    from transformers import TimmWrapperConfig
except ImportError:

    class TimmWrapperConfig:
        pass


if TYPE_CHECKING and is_peft_available():
    from peft import PeftConfig

logger = transformers_logging.get_logger(__name__)

_TRANSFORMERS_PROCESSOR_SUPPORTS_MODALITY_KWARGS = parse_version(transformers_version) > parse_version("4.56.1")
_TRANSFORMERS_APPLY_CHAT_TEMPLATE_RECOMMENDS_PROCESSOR_KWARGS = parse_version(transformers_version) >= parse_version(
    "5.4.0.dev0"
)

TransformerTask = Literal[
    "feature-extraction", "sequence-classification", "text-generation", "any-to-any", "fill-mask"
]


class _ModalityParamsRequired(TypedDict):
    method: str
    method_output_name: str | None


class ModalityParams(_ModalityParamsRequired, total=False):
    """Parameters for a single modality entry in the modality config.

    The ``format`` key is only used for the ``"message"`` modality and controls how
    message content is structured: ``"structured"`` (list of typed dicts) or ``"flat"``
    (direct string value).
    """

    format: Literal["structured", "flat"]


ModalityConfig = dict[Modality, ModalityParams]

TRANSFORMER_TASK_TO_AUTO_MODEL: dict[TransformerTask, Any] = {
    "feature-extraction": AutoModel,  # Used by SentenceTransformer, also covers "image-feature-extraction"
    "sequence-classification": AutoModelForSequenceClassification,  # Used by CrossEncoder
    "text-generation": AutoModelForCausalLM,  # Used by CrossEncoder
    "fill-mask": AutoModelForMaskedLM,  # Used by SparseEncoder
}

try:
    from transformers import AutoModelForMultimodalLM

    TRANSFORMER_TASK_TO_AUTO_MODEL["any-to-any"] = (
        AutoModelForMultimodalLM  # Used by CrossEncoder, also covers "image-text-to-text"
    )
except ImportError:
    pass

# Default (modality_config, module_output_name) per transformer task.
# Used as the fallback when loading models saved before modality_config was introduced,
# and as the source of default method_output_name / module_output_name during modality inference.
TRANSFORMER_TASK_DEFAULTS: dict[TransformerTask, tuple[ModalityConfig, str]] = {
    "feature-extraction": (
        {"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
    ),
    "sequence-classification": (
        {"text": {"method": "forward", "method_output_name": "logits"}},
        "scores",
    ),
    "text-generation": (
        {"text": {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
    ),
    "any-to-any": (
        {"text": {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
    ),
    "fill-mask": (
        {"text": {"method": "forward", "method_output_name": "logits"}},
        "token_embeddings",
    ),
}

# Registry of encoder-decoder architectures whose encoder can be loaded standalone.
# Each entry maps a config class to (module_path, encoder_class_name) for lazy importing.
_ENCODER_ONLY_MODELS: list[tuple[type, str, str]] = [
    (T5Config, "transformers", "T5EncoderModel"),
    (MT5Config, "transformers", "MT5EncoderModel"),
    (UMT5Config, "transformers", "UMT5EncoderModel"),
    (UdopConfig, "transformers", "UdopEncoderModel"),
    (LongT5Config, "transformers", "LongT5EncoderModel"),
    (ProphetNetConfig, "transformers", "ProphetNetEncoder"),
    (SwitchTransformersConfig, "transformers", "SwitchTransformersEncoderModel"),
    (BlenderbotConfig, "transformers.models.blenderbot.modeling_blenderbot", "BlenderbotEncoder"),
    (
        BlenderbotSmallConfig,
        "transformers.models.blenderbot_small.modeling_blenderbot_small",
        "BlenderbotSmallEncoder",
    ),
    (M2M100Config, "transformers.models.m2m_100.modeling_m2m_100", "M2M100Encoder"),
    (PegasusConfig, "transformers.models.pegasus.modeling_pegasus", "PegasusEncoder"),
    (PegasusXConfig, "transformers.models.pegasus_x.modeling_pegasus_x", "PegasusXEncoder"),
    (MoonshineConfig, "transformers.models.moonshine.modeling_moonshine", "MoonshineEncoder"),
    (WhisperConfig, "transformers.models.whisper.modeling_whisper", "WhisperEncoder"),
    (MarianConfig, "transformers.models.marian.modeling_marian", "MarianEncoder"),
    # T5Gemma2TextConfig is for loading from an already encoder-only checkpoint;
    # loading the encoder from a full T5Gemma2Config is handled separately in _load_encoder_only_model.
    (T5Gemma2TextConfig, "transformers.models.t5gemma2.modeling_t5gemma2", "T5Gemma2Encoder"),
]

# Hard-coded modality configs for model types that can't be handled by the general inference path.
# Each entry maps a model_type string to (modality_config, module_output_name, validate_output_names).
# When validate_output_names is True, each modality's method_output_name is validated at runtime
# against the model method's return type via _infer_method_output_name (for transformers v4/v5 compat).
_AUDIO_MODALITY_CONFIG: tuple[ModalityConfig, str, bool] = (
    {
        "audio": {"method": "forward", "method_output_name": "last_hidden_state"},
        ("audio", "text"): {"method": "forward", "method_output_name": "last_hidden_state"},
    },
    "token_embeddings",
    False,
)

_FEATURE_EXTRACTION_EDGE_CASES: dict[str, tuple[ModalityConfig, str, bool]] = {
    # Models with custom get_*_features methods that need output name validation
    "blip": (
        {
            "text": {"method": "get_text_features", "method_output_name": "pooler_output"},
            "image": {"method": "get_image_features", "method_output_name": "pooler_output"},
            ("image", "text"): {"method": "get_multimodal_features", "method_output_name": "pooler_output"},
        },
        "sentence_embedding",
        True,
    ),
    "blip-2": (
        {
            "text": {"method": "get_text_features", "method_output_name": "last_hidden_state"},
            "image": {"method": "get_image_features", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        True,
    ),
    "sam3": (
        {
            "text": {"method": "get_text_features", "method_output_name": "last_hidden_state"},
            "image": {"method": "get_vision_features", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        True,
    ),
    "flava": (
        {
            "text": {"method": "get_text_features", "method_output_name": "pooler_output"},
            "image": {"method": "get_image_features", "method_output_name": "pooler_output"},
        },
        "token_embeddings",
        True,
    ),
    # Models supporting text+image without message format
    "git": (
        {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            ("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        False,
    ),
    "visual_bert": (
        {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            ("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        False,
    ),
    # Only combined (image, text) input is supported, no text-only nor image-only
    "kosmos-2": (
        {("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
        False,
    ),
    "grounding-dino": (
        {("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
        False,
    ),
    "paligemma": (
        {("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
        False,
    ),
    "vilt": (
        {("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"}},
        "token_embeddings",
        False,
    ),
    # Image+text supported without message format, plus image-only
    "layoutlmv3": (
        {
            "image": {"method": "forward", "method_output_name": "last_hidden_state"},
            ("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        False,
    ),
    # All modalities via forward
    "idefics": (
        {
            "text": {"method": "forward", "method_output_name": "last_hidden_state"},
            "image": {"method": "forward", "method_output_name": "last_hidden_state"},
            ("image", "text"): {"method": "forward", "method_output_name": "last_hidden_state"},
        },
        "token_embeddings",
        False,
    ),
    # Audio encoder models; we load only the encoder, so text decoding is not available
    "hubert": _AUDIO_MODALITY_CONFIG,
    "moonshine": _AUDIO_MODALITY_CONFIG,
    "sew": _AUDIO_MODALITY_CONFIG,
    "sew-d": _AUDIO_MODALITY_CONFIG,
    "unispeech-sat": _AUDIO_MODALITY_CONFIG,
    "unispeech": _AUDIO_MODALITY_CONFIG,
    "wav2vec2": _AUDIO_MODALITY_CONFIG,
    "wav2vec2-conformer": _AUDIO_MODALITY_CONFIG,
    "wavlm": _AUDIO_MODALITY_CONFIG,
    "whisper": _AUDIO_MODALITY_CONFIG,
    "voxtral_realtime": _AUDIO_MODALITY_CONFIG,
}

_FILL_MASK_EDGE_CASES: dict[str, tuple[ModalityConfig, str, bool]] = {
    # wav2vec2's forward outputs 'logits' rather than 'last_hidden_state' for fill-mask,
    # but accepts audio input rather than text
    "wav2vec2": (
        {
            "audio": {"method": "forward", "method_output_name": "logits"},
        },
        "token_embeddings",
        False,
    ),
}

_TEXT_GENERATION_EDGE_CASES = {
    # Models supporting text+image without message format, but no image-only
    "git": (
        {
            "text": {"method": "forward", "method_output_name": "logits"},
            ("image", "text"): {"method": "forward", "method_output_name": "logits"},
        },
        "causal_logits",
        False,
    ),
    # The Whisper decoder is text only
    "whisper": (
        {
            "text": {"method": "forward", "method_output_name": "logits"},
        },
        "causal_logits",
        False,
    ),
}

_ANY_TO_ANY_EDGE_CASES = {
    # Models supporting text+image without message format, but no image-only
    "git": (
        {
            "text": {"method": "forward", "method_output_name": "logits"},
            ("image", "text"): {"method": "forward", "method_output_name": "logits"},
        },
        "causal_logits",
        False,
    ),
    # Only combined (image, text) input is supported, no text-only nor image-only
    "blip": (
        {("image", "text"): {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
        False,
    ),
    "blip-2": (
        {("image", "text"): {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
        False,
    ),
    "kosmos-2": (
        {("image", "text"): {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
        False,
    ),
    "paligemma": (
        {("image", "text"): {"method": "forward", "method_output_name": "logits"}},
        "causal_logits",
        False,
    ),
    # Models supporting text+audio, but no text-only
    "voxtral_realtime": (
        {
            "audio": {"method": "forward", "method_output_name": "logits"},
            ("audio", "text"): {"method": "forward", "method_output_name": "logits"},
        },
        "causal_logits",
        False,
    ),
}

_EDGE_CASE_MODALITY_CONFIGS: dict[str, dict[str, tuple[ModalityConfig, str, bool]]] = {
    "feature-extraction": _FEATURE_EXTRACTION_EDGE_CASES,
    "text-generation": _TEXT_GENERATION_EDGE_CASES,
    "any-to-any": _ANY_TO_ANY_EDGE_CASES,
    "fill-mask": _FILL_MASK_EDGE_CASES,
}


def _has_lowercase(normalizer) -> bool:
    """Check whether a tokenizers normalizer (or sequence of normalizers) includes Lowercase."""
    if normalizer is None:
        return False
    if isinstance(normalizer, Lowercase):
        return True
    if isinstance(normalizer, Sequence):
        return any(isinstance(n, Lowercase) for n in normalizer)
    return False


@contextmanager
def set_temporary_class_attrs(cls, **overrides):
    originals = {name: getattr(cls, name, None) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(cls, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(cls, name, value)


class ProcessingKwargs(TypedDict, total=False):
    """Keyword arguments applied when *calling* the processor during preprocessing.

    Valid keys: ``"common"``, ``"text"``, ``"audio"``, ``"image"``, ``"video"``, ``"chat_template"``.
    Modality and ``"common"`` kwargs override built-in defaults. ``"chat_template"`` kwargs are
    forwarded to ``apply_chat_template``.
    """

    common: dict[str, Any]
    text: dict[str, Any]
    audio: dict[str, Any]
    image: dict[str, Any]
    video: dict[str, Any]
    chat_template: dict[str, Any]


def _count_media_per_sample(messages: list[list[dict[str, Any]]]) -> tuple[list[int], list[int]]:
    """Count images and videos per sample from the message structure.

    Some VLM processors flatten per-sample visual tokens into single tensors (e.g.
    ``pixel_values`` shape ``(total_visual_tokens, hidden_dim)``), losing the per-sample
    association. By counting before the processor call, we get reliable per-sample counts
    that downstream minibatching can use directly.
    """
    num_images: list[int] = []
    num_videos: list[int] = []
    for sample_messages in messages:
        img_count = 0
        vid_count = 0
        for msg in sample_messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "image":
                        img_count += 1
                    elif item_type == "video":
                        vid_count += 1
        num_images.append(img_count)
        num_videos.append(vid_count)
    return num_images, num_videos


class Transformer(InputModule):
    """Hugging Face AutoModel wrapper that handles loading, preprocessing, and inference.

    Loads the appropriate model class (e.g. BERT, RoBERTa, CLIP, Whisper) based on the model configuration
    and the specified ``transformer_task``. Supports text, image, audio, and video modalities depending on
    the underlying model. This module is typically the first module in a
    :class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`, :class:`~sentence_transformers.sparse_encoder.model.SparseEncoder`,
    or :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` pipeline.

    Args:
        model_name_or_path (str): Hugging Face model name or path to a local model directory.
        transformer_task (str, optional): The task determining which ``AutoModel``-like class to load.
            Supported values:

            - ``"feature-extraction"`` (default): :class:`~transformers.AutoModel`, e.g. used by
              :class:`~sentence_transformers.sentence_transformer.model.SentenceTransformer`.
            - ``"sequence-classification"``: :class:`~transformers.AutoModelForSequenceClassification`,
              e.g. used by :class:`~sentence_transformers.cross_encoder.model.CrossEncoder`.
            - ``"text-generation"``: :class:`~transformers.AutoModelForCausalLM`, e.g. used by generative
              :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models. Sets the ``tokenizer`` padding_side to "left".
            - ``"any-to-any"``: :class:`~transformers.AutoModelForMultimodalLM`, e.g. used by multimodal generative
              :class:`~sentence_transformers.cross_encoder.model.CrossEncoder` models (requires transformers v5+). Sets the
              ``tokenizer`` padding_side to "left".
            - ``"fill-mask"``: :class:`~transformers.AutoModelForMaskedLM`, e.g. used by
              :class:`~sentence_transformers.sparse_encoder.model.SparseEncoder`.

            Defaults to ``"feature-extraction"``.
        model_kwargs (dict[str, Any], optional): Keyword arguments forwarded to
            ``AutoModel.from_pretrained`` when loading the model. Particularly useful options include:

            - ``torch_dtype``: Override the default ``torch.dtype`` and load the model under a specific
              dtype. Can be ``torch.float16``, ``torch.bfloat16``, ``torch.float32``, or ``"auto"`` to
              use the dtype from the model's ``config.json``.
            - ``attn_implementation``: The attention implementation to use. For example ``"eager"``,
              ``"sdpa"``, or ``"flash_attention_2"``. If you ``pip install kernels``, then
              ``"flash_attention_2"`` should work without having to install ``flash_attn``. It is
              frequently the fastest option. Defaults to ``"sdpa"`` when available (torch>=2.1.1).
            - ``device_map``: Device map for model parallelism, e.g. ``"auto"``.
            - ``provider``: For ``backend="onnx"``, the ONNX execution provider
              (e.g. ``"CUDAExecutionProvider"``).
            - ``file_name``: For ``backend="onnx"`` or ``"openvino"``, the filename to load
              (e.g. for optimized or quantized models).
            - ``export``: For ``backend="onnx"`` or ``"openvino"``, whether to export the model to the
              backend format. Also set automatically if the exported file doesn't exist.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details. Defaults to None.
        processor_kwargs (dict[str, Any], optional): Keyword arguments forwarded to
            ``AutoProcessor.from_pretrained`` when loading the processor/tokenizer. See the
            `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details. Defaults to None.
        config_kwargs (dict[str, Any], optional): Keyword arguments forwarded to
            ``AutoConfig.from_pretrained`` when loading the config. See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details. Defaults to None.
        processing_kwargs (dict[str, dict[str, Any]], optional): Keyword arguments applied when *calling*
            the processor during preprocessing. This is a nested dict whose keys are modality names
            (``"text"``, ``"audio"``, ``"image"``, ``"video"``), ``"common"`` for kwargs shared across all
            modalities, or ``"chat_template"`` for kwargs forwarded to ``apply_chat_template`` (e.g.
            ``{"add_generation_prompt": True}``). Modality and common kwargs override the built-in defaults.
            Saved to and loaded from the model configuration file. Defaults to None.
        backend (str, optional): Backend used for model inference. Can be ``"torch"`` (default), ``"onnx"``,
            or ``"openvino"``. Defaults to ``"torch"``.
        modality_config (dict, optional): Custom modality configuration mapping modality names to method and
            output name dicts. When provided, ``module_output_name`` must also be set. The ``"message"``
            modality entry may include a ``"format"`` key (``"structured"``, ``"flat"``, or ``"auto"``)
            to control how chat-template inputs are formatted. Defaults to None.
        module_output_name (str, optional): The name of the output feature this module creates (e.g.
            ``"token_embeddings"``, ``"scores"``). Required when ``modality_config`` is provided.
            Defaults to None.
        unpad_inputs (bool, optional): Controls whether text-only inputs are concatenated without
            padding for faster inference using flash attention's variable-length functions. Non-text
            inputs (images, audio, video) are always padded normally. If ``None`` (default), unpadding
            is enabled automatically when all prerequisites are met (flash attention with variable-length
            support, ``"torch"`` backend, ``"feature-extraction"`` task). Set to ``False`` to force
            padding, which is needed for architectures that don't support unpadded inputs (e.g.
            ``qwen2_vl``). Set to ``True`` to request unpadding explicitly; a warning is logged if the
            prerequisites are not met. Defaults to None.
        max_seq_length (int, optional): Truncate any inputs longer than this value. Prefer setting
            ``model_max_length`` via ``processor_kwargs`` instead. Defaults to None.
        do_lower_case (bool, optional): If true, lowercases the input (independent of whether the model
            is cased or not). Rarely needed. Defaults to False.
        tokenizer_name_or_path (str, optional): Name or path of the tokenizer. When None,
            ``model_name_or_path`` is used. Deprecated. Defaults to None.
    """

    config_file_name: str = "sentence_bert_config.json"
    config_keys: list[str] = [
        "transformer_task",
        "modality_config",
        "module_output_name",
        "processing_kwargs",
        "unpad_inputs",
    ]
    save_in_root: bool = True

    @transformer_kwargs_decorator
    def __init__(
        self,
        model_name_or_path: str,
        *,
        transformer_task: TransformerTask = "feature-extraction",
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        processing_kwargs: ProcessingKwargs | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        modality_config: ModalityConfig | None = None,
        module_output_name: str | None = None,
        unpad_inputs: bool | None = None,
        max_seq_length: int | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
    ) -> None:
        super().__init__()
        if transformer_task not in TRANSFORMER_TASK_TO_AUTO_MODEL:
            if transformer_task == "any-to-any":
                raise ImportError(
                    "The 'any-to-any' transformer task requires transformers v5+. "
                    "Please upgrade transformers with `pip install transformers>=5.0.0`."
                )
            raise ValueError(
                f"Unsupported transformer_task '{transformer_task}'. Supported tasks are: {list(TRANSFORMER_TASK_TO_AUTO_MODEL.keys())}"
            )
        self.transformer_task: TransformerTask = transformer_task
        if model_kwargs is None:
            model_kwargs = {}
        if processor_kwargs is None:
            processor_kwargs = {}
        if config_kwargs is None:
            config_kwargs = {}
        self.processing_kwargs: ProcessingKwargs = processing_kwargs or {}
        valid_keys = {"common", "text", "audio", "image", "video", "chat_template"}
        unknown_keys = set(self.processing_kwargs) - valid_keys
        if unknown_keys:
            logger.warning(
                f"Unknown keys in `processing_kwargs`: {unknown_keys}. "
                f"Valid keys are: {sorted(valid_keys)}. "
                "Unknown keys will be ignored. Did you mean to nest them under "
                "'common' or a modality key ('text', 'audio', 'image', 'video')?"
            )
        self.backend = backend
        self.do_lower_case = do_lower_case
        self.track_media_counts = False
        self._prompt_length_mapping = {}
        self._method_signature_cache: dict[str, set[str]] = {}

        config, is_peft_model = self._load_config(model_name_or_path, backend, config_kwargs)

        if (
            transformer_task == "sequence-classification"
            and "num_labels" not in config_kwargs
            and (
                config.architectures is None
                or not any(arch.endswith("ForSequenceClassification") for arch in config.architectures)
            )
        ):
            # If we're loading a model for sequence-classification, but the base architecture is not for sequence-classification,
            # and num_labels is not specified, we default to 1 label for CrossEncoder-like behavior
            config.num_labels = 1

        self.model = self._load_model(
            model_name_or_path, transformer_task, config, backend, is_peft_model, **model_kwargs
        )

        # Start from the forward signature and add common parameter names as a safety net
        # for models that use **kwargs or a wrapper that hides them from the signature.
        self.model_forward_params = set(inspect.signature(self.model.forward).parameters) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
            "return_dict",
        }

        if max_seq_length is not None and "model_max_length" not in processor_kwargs:
            processor_kwargs["model_max_length"] = max_seq_length
        with suggest_extra_on_exception():
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
                **processor_kwargs,
            )

        # Cap the tokenizer model_max_length at the model's max_position_embeddings
        if self.tokenizer is not None:
            # NOTE: xlnet uses a hardcoded config.max_position_embeddings != -1 to denote no max_length
            if (
                "model_max_length" not in processor_kwargs
                and hasattr(self.config, "max_position_embeddings")
                and self.config.max_position_embeddings != -1
            ):
                self.tokenizer.model_max_length = min(
                    self.tokenizer.model_max_length, self.config.max_position_embeddings
                )

            if do_lower_case:
                # NOTE: All Transformers v5 tokenizers are fast tokenizers, but we keep the v4 branch for compatibility
                if self.tokenizer.is_fast:
                    normalizer = self.tokenizer.backend_tokenizer.normalizer
                    if not _has_lowercase(normalizer):
                        new_normalizers = [Lowercase()]
                        if isinstance(normalizer, Sequence):
                            new_normalizers += list(normalizer)
                        elif normalizer is not None:
                            new_normalizers.append(normalizer)
                        self.tokenizer.backend_tokenizer.normalizer = Sequence(new_normalizers)
                else:
                    # Some v4 Tokenizers have do_lower_case as property without a setter, and those often
                    # have a basic_tokenizer on which do_lower_case can be set.
                    try:
                        self.tokenizer.do_lower_case = do_lower_case
                    except AttributeError:
                        self.tokenizer.basic_tokenizer.do_lower_case = do_lower_case

        # Causal models require left padding so the last position is always a real token,
        # which is needed for logits_to_keep=1 and LogitScore.
        if self.transformer_task in ("text-generation", "any-to-any"):
            self.processor.padding_side = "left"
            if hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = "left"

        # Extract message format from modality_config if provided, otherwise let InputFormatter infer it
        if modality_config is not None and "message" in modality_config:
            message_format = modality_config["message"].get("format", "auto")
        else:
            message_format = "auto"
        self.input_formatter = InputFormatter(
            model_type=self.config.model_type, message_format=message_format, processor=self.processor
        )

        if modality_config is not None:
            self.modality_config = modality_config
            if module_output_name is None:
                raise ValueError(
                    "Loading the Transformer module with a custom modality_config requires also providing "
                    "module_output_name with the name of the output feature that this module should create, "
                    'for example "token_embeddings" or "sentence_embedding".'
                )
            self.module_output_name = module_output_name
            for modality_key, params in modality_config.items():
                if not isinstance(params, dict) or "method" not in params or "method_output_name" not in params:
                    raise ValueError(
                        f"Invalid modality_config entry for {modality_key!r}: each entry must be a dict with "
                        f"'method' and 'method_output_name' keys, but got {params!r}"
                    )
        else:
            self.modality_config, self.module_output_name = self.infer_modalities(self.model, self.processor)
        logger.debug(f"Active modality config: {self.modality_config}")
        self.input_formatter.supported_modalities = list(self.modality_config.keys())

        if tokenizer_name_or_path is not None:
            logger.warning(
                "The `tokenizer_name_or_path` argument is deprecated and will be removed in a future version. "
                "Please use the same path for the model and processor."
            )
            self.model.config.tokenizer_class = self.processor.__class__.__name__

        # Evaluate whether we can skip padding
        self.unpad_inputs = unpad_inputs

    @property
    def unpad_inputs(self) -> bool | None:
        """Whether text-only inputs are concatenated without padding for faster inference.

        Non-text inputs (images, audio, video) are always padded normally.
        ``None`` auto-detects, ``False`` forces padding, ``True`` requests unpadding.
        Re-evaluates on every assignment, so it can be changed after loading::

            model = SentenceTransformer("my-model", model_kwargs={"attn_implementation": "flash_attention_2"})
            model[0].unpad_inputs = False  # Force padding for models that need it
        """
        return self._unpad_inputs

    @unpad_inputs.setter
    def unpad_inputs(self, value: bool | None) -> None:
        self._unpad_inputs = value
        if value is False:
            self.can_flatten_inputs = False
        else:
            self.can_flatten_inputs = self._can_flatten_inputs()
            if value is True and not self.can_flatten_inputs:
                logger.warning(
                    "unpad_inputs=True was set, but the prerequisites for skipping padding are not met. "
                    "Falling back to padded inputs."
                )

    def _can_flatten_inputs(self) -> bool:
        """Determine whether text-only inputs can be flattened (concatenated without padding) for more efficient inference.

        When enabled, text-only inputs are concatenated into a single sequence and processed using flash
        attention's variable-length functions, eliminating padding overhead and significantly speeding up
        inference. Non-text inputs (images, audio, video) are always padded normally, even when this
        returns True.

        This requires:
        1. The ``"feature-extraction"`` task, as model heads (e.g. ``AutoModelForSequenceClassification``)
           are incompatible with flattened inputs.
        2. The ``"text"`` modality must be supported by the model.
        3. All modality call methods must be ``"forward"``; ``get_..._features`` methods apply heads
           that are incompatible with flattened inputs.
        4. The ``"torch"`` backend with an attention-interface-compatible model.
        5. Flash attention with variable-length function support.

        Note: Some architectures don't work with unpadded inputs (e.g. ``qwen2_vl``). Use
        ``unpad_inputs=False`` to disable this optimization for such models.

        Returns:
            bool: True if text-only inputs can be flattened for efficient inference.
        """
        if (
            self.transformer_task != "feature-extraction"
            or "text" not in self.modality_config
            or self.backend != "torch"
            or not getattr(self.model, "is_backend_compatible", lambda: False)()
            or any(params["method"] != "forward" for params in self.modality_config.values())
        ):
            return False

        try:
            from transformers import DataCollatorWithFlattening
            from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
            from transformers.utils.generic import is_flash_attention_requested
        except ImportError:
            logger.debug(
                "Consider upgrading to transformers >= 5.0.0 to skip padding for text-only inputs, "
                "which can significantly speed up processing."
            )
            return False

        attn_implementation = self.config._attn_implementation
        if not is_flash_attention_requested(requested_attention_implementation=attn_implementation):
            return False

        (_, flash_varlen_fn, *_), _ = lazy_import_flash_attention(attn_implementation)
        if flash_varlen_fn is None:
            return False

        logger.debug(
            "Using flattened inputs with flash attention variable-length functions to avoid padding overhead for text-only inputs."
        )
        self.data_collator = DataCollatorWithFlattening(
            return_seq_idx=True,  # Not always necessary, perhaps only Mamba/Bamba?
            return_flash_attn_kwargs=True,
            return_position_ids=True,  # Crucial for performance
        )
        # Ensure the flash attention keys reach the model through **kwargs. They
        # are not named parameters in the model's forward signature, but the model
        # passes them through to its attention layers via **kwargs.
        self.model_forward_params |= {
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
            "seq_idx",
        }
        return True

    @property
    def max_seq_length(self) -> int | None:
        """The maximum input sequence length. Reads from the tokenizer if available, otherwise
        falls back to ``max_position_embeddings`` from the model config."""
        if self.tokenizer is not None:
            return self.tokenizer.model_max_length

        # Get text config for multi-modal models that don't have a tokenizer
        if hasattr(self.model.config, "get_text_config"):
            text_config = self.model.config.get_text_config()
        else:
            text_config = self.model.config

        if hasattr(text_config, "max_position_embeddings"):
            return text_config.max_position_embeddings
        return None

    @max_seq_length.setter
    def max_seq_length(self, value: int | None) -> None:
        """Set the maximum input sequence length. Only effective when a tokenizer is available."""
        if self.tokenizer is not None:
            self.tokenizer.model_max_length = value

    @property
    def auto_model(self) -> PreTrainedModel:
        """The underlying transformer model."""
        return self.model

    @property
    def config(self) -> PretrainedConfig:
        """The underlying model configuration."""
        return self.model.config

    @property
    def modalities(self) -> list[Modality]:
        """The list of supported input modalities (e.g. ``"text"``, ``"image"``, ``("image", "text")``)."""
        return list(self.modality_config.keys())

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """The tokenizer, extracted from the processor. Returns ``None`` for non-text processors."""
        if isinstance(self.processor, PreTrainedTokenizerBase):
            return self.processor
        return getattr(self.processor, "tokenizer", None)

    def preprocess(
        self,
        inputs: list[SingleInput | PairInput],
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Preprocess inputs into model-ready features.

        Args:
            inputs: List of inputs. Can contain strings, dicts with modality keys, PIL images,
                or numpy/torch arrays for audio/video.
            prompt: Optional prompt to prepend to text inputs or inject as a system message.
            **kwargs: Additional keyword arguments forwarded to prompt length computation
                (e.g. ``task``). Only used when ``prompt`` is provided for text inputs.

        Returns:
            Dictionary containing preprocessed tensors with a ``modality`` key indicating the
            input type and optionally a ``prompt_length`` key for prompt-aware pooling.
        """
        if not inputs:
            return {}

        common_kwargs = {"return_tensors": "pt"}
        modality_kwargs = {
            "text": {"padding": True, "truncation": "longest_first"},
            "audio": {"padding": True},
            "image": {},
            "video": {},
        }
        if self.config.model_type == "whisper":
            # Whisper requires inputs to be exactly 30 seconds long, while its WhisperFeatureExtractor defaults to
            # padding=True (a.k.a. "longest"), instead of defaulting to the required "max_length".
            modality_kwargs["audio"]["padding"] = "max_length"

        # Apply user-configured processing_kwargs on top of the defaults
        if "common" in self.processing_kwargs:
            common_kwargs.update(self.processing_kwargs["common"])
        for modality_key in modality_kwargs:
            if overrides := self.processing_kwargs.get(modality_key):  # type: ignore[arg-type]
                modality_kwargs[modality_key].update(overrides)

        modality, processor_inputs, extra_modality_kwargs = self.input_formatter.parse_inputs(inputs)

        for modality_key, extra_kwargs in extra_modality_kwargs.items():
            modality_kwargs[modality_key].update(extra_kwargs)

        # Flatten inputs to avoid padding overhead when using flash attention variable-length functions.
        # Only safe for text-only inputs, since DataCollatorWithFlattening only handles input_ids/labels.
        should_flatten = self.can_flatten_inputs and (
            modality == "text"
            or (modality == "message" and self.input_formatter.is_text_only_messages(processor_inputs["message"]))
        )
        if should_flatten:
            del common_kwargs["return_tensors"]
            modality_kwargs["text"].pop("padding", None)
            modality_kwargs["text"]["return_attention_mask"] = False

        # Always convert to the message format if it's supported, since it's most flexible with e.g. defaults
        if "message" in self.modality_config and modality != "message":
            modality, processor_inputs = self.input_formatter.batch_to_message(modality, processor_inputs)
        elif modality not in self.modality_config:
            raise ValueError(
                f"Modality '{format_modality(modality)}' is not supported by this model. "
                f"Supported modalities: {', '.join(format_modality(m) for m in sorted(self.modality_config.keys(), key=str))}"
            )

        # Incorporate prompt into inputs if applicable
        prompt_length = None
        if prompt and modality == "message":
            processor_inputs["message"] = self.input_formatter.prepend_prompt_to_messages(
                processor_inputs["message"], prompt
            )
            # Models relying on the message format don't support excluding prompt tokens in mean pooling,
            # so we don't track prompt length.
        elif prompt and modality == "text":
            processor_inputs["text"] = self.input_formatter.prepend_prompt_to_texts(processor_inputs["text"], prompt)
            prompt_length = self._get_prompt_length(prompt, **kwargs)

        # Track per-sample image/video counts before the processor flattens them into single tensors.
        # Losses that minibatch VLM inputs (e.g. CachedMNRL) use these counts to slice visual tensors.
        # Only used if the Trainer updated track_media_counts to True.
        num_images_per_sample = None
        num_videos_per_sample = None
        if self.training and self.track_media_counts and modality == "message":
            num_images_per_sample, num_videos_per_sample = _count_media_per_sample(processor_inputs["message"])

        with suggest_extra_on_exception():
            processor_output = self._call_processor(modality, processor_inputs, modality_kwargs, common_kwargs)

        if num_images_per_sample is not None and "image_grid_thw" in processor_output:
            processor_output["num_images_per_sample"] = torch.tensor(num_images_per_sample, dtype=torch.long)
        if num_videos_per_sample is not None and "video_grid_thw" in processor_output:
            processor_output["num_videos_per_sample"] = torch.tensor(num_videos_per_sample, dtype=torch.long)

        if should_flatten:
            # DataCollatorWithFlattening expects list[dict], but the processor returns dict[str, list].
            per_sample = [dict(zip(processor_output, values)) for values in zip(*processor_output.values())]
            processor_output = self.data_collator(per_sample)
            processor_output.pop("labels", None)

        processor_output["modality"] = modality
        if prompt_length is not None:
            processor_output["prompt_length"] = prompt_length

        if self.transformer_task in ("text-generation", "any-to-any"):
            if self.processor.padding_side != "left":
                raise ValueError(
                    f"The processor padding side is {self.processor.padding_side!r}, but causal models require "
                    "left padding so that the last token position is always a real token. "
                    "This is needed for efficient logit computation (logits_to_keep=1) and for LogitScore. "
                    'Please set ``processing_kwargs={"padding_side": "left"}``.'
                )
            processor_output["logits_to_keep"] = 1

        return processor_output

    def forward(self, features: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward pass through the transformer model.

        Dispatches to the appropriate model method based on the ``modality`` key in ``features``
        and writes the result into ``features[self.module_output_name]``.

        Args:
            features: Input features dictionary produced by :meth:`preprocess`. Must contain the
                keys expected by the underlying model (e.g. ``input_ids``, ``pixel_values``, etc.).
                A ``modality`` key selects the modality config to use; defaults to ``"text"``
                when absent.
            **kwargs: Additional keyword arguments forwarded to the model method (override features).

        Returns:
            The updated ``features`` dict with the model output stored under ``self.module_output_name``
            (e.g. ``token_embeddings``, ``sentence_embedding``, ``scores``, or ``causal_logits``).
            May also include ``all_layer_embeddings`` if ``output_hidden_states`` is enabled.
        """

        modality_name: Modality = features.get("modality", "text")
        modality_params = self.modality_config[modality_name]
        method_name = modality_params["method"]
        method_output_name = modality_params["method_output_name"]
        if isinstance(method_output_name, str):
            method_output_name = (method_output_name,)
        elif isinstance(method_output_name, list):
            method_output_name = tuple(method_output_name)

        # kwargs override features
        all_kwargs = {**features, **kwargs, "return_dict": True}
        model_method = getattr(self.model, method_name, None)
        if model_method is None:
            raise ValueError(f"Model does not have the requested '{method_name}' method")

        if method_name == "forward":
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in self.model_forward_params}
        else:
            method_params = self._method_signature_cache.get(method_name)
            if method_params is None:
                method_params = set(inspect.signature(model_method).parameters)
                self._method_signature_cache[method_name] = method_params
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in method_params}

        # Auto-enable output_hidden_states when the output path traverses hidden_states
        if method_output_name is not None and "hidden_states" in method_output_name:
            filtered_kwargs["output_hidden_states"] = True

        model_output = model_method(**filtered_kwargs)

        embedding = model_output
        if method_output_name is not None:
            for output_key in method_output_name:
                try:
                    embedding = embedding[output_key]
                except (KeyError, TypeError):
                    # Some models (e.g. chinese_clip) only expose output fields via attribute access,
                    # not dictionary-style indexing. See https://github.com/huggingface/transformers/issues/44079
                    try:
                        embedding = getattr(embedding, output_key)
                    except AttributeError:
                        raise AttributeError(
                            f"Could not access output key {output_key!r} via indexing or attribute access "
                            f"on {type(embedding).__name__}."
                        )

        if embedding.ndim == 4:
            # Some image models return (batch_size, num_channels, height, width) instead of (batch_size, seq_len, hidden_size)
            # We flatten the height and width dimensions and transpose to get (batch_size, height*width, num_channels)
            # which a subsequent Pooling layer can handle to remove the height*width dimension
            embedding = embedding.flatten(2).transpose(1, 2)

        features[self.module_output_name] = embedding

        # If the AutoModel is wrapped with a PeftModel(ForFeatureExtraction), then it may have added virtual tokens
        # We need to extend the attention mask to include these virtual tokens, or the pooling will fail
        if "input_ids" in features and "attention_mask" in features and is_peft_available():
            from peft import PeftModel

            if isinstance(self.model, PeftModel) and self.model.active_peft_config.is_prompt_learning:
                batch_size = features["input_ids"].shape[0]
                attention_mask = features["attention_mask"]
                prefix_attention_mask = torch.ones(
                    batch_size, self.model.active_peft_config.num_virtual_tokens, device=attention_mask.device
                )
                features["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if (
            hasattr(self.model.config, "output_hidden_states")
            and self.model.config.output_hidden_states
            and "hidden_states" in model_output
        ):
            features["all_layer_embeddings"] = model_output["hidden_states"]

        return features

    def get_embedding_dimension(self) -> int:
        """Get the output embedding dimension from the transformer model.

        Returns:
            int: The hidden dimension size of the model's embeddings.

        Raises:
            ValueError: If the embedding dimension cannot be determined from the model config.
        """
        # Edge case for timm models
        if isinstance(self.model.config, TimmWrapperConfig):
            return self.model.config.num_features

        def get_hidden_size_from_config(config):
            # If we're directly outputting sentence embeddings from the transformer (e.g., using the pooler output),
            # then we should check for projection_dim first, as that's likely the dimension of the sentence embeddings
            # after a projection layer
            if hasattr(config, "projection_dim") and self.module_output_name == "sentence_embedding":
                return config.projection_dim

            if hasattr(config, "hidden_size"):
                return config.hidden_size
            for attr_name in ("neck_hidden_sizes", "hidden_sizes", "embed_dims"):
                if hasattr(config, attr_name):
                    value = getattr(config, attr_name)
                    if isinstance(value, list):
                        if value:
                            return value[-1]
                    else:
                        return value
            if hasattr(config, "hidden_dim"):
                return config.hidden_dim
            return None

        if (hidden_size := get_hidden_size_from_config(self.model.config)) is not None:
            return hidden_size

        # Text config hidden size has priority
        if hasattr(self.model.config, "text_config"):
            if (hidden_size := get_hidden_size_from_config(self.model.config.text_config)) is not None:
                return hidden_size

        # Afterwards we check all sub-configs
        if hasattr(self.model.config, "sub_configs"):
            for sub_config_name in self.model.config.sub_configs.keys():
                sub_config = getattr(self.model.config, sub_config_name)
                if (hidden_size := get_hidden_size_from_config(sub_config)) is not None:
                    return hidden_size

        raise ValueError(
            f"Could not determine embedding dimension from model config. Config type: {type(self.model.config).__name__}."
        )

    def _call_processor(
        self,
        modality: Modality,
        processor_inputs: dict[str, list],
        modality_kwargs: dict[str, dict[str, Any]],
        common_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the appropriate processor with the correct arguments.

        Dispatches based on the processor type and modality:

        1. **Message modality**: delegates to :meth:`_process_chat_messages`.
        2. **Multi-modal processor** (:class:`ProcessorMixin`): delegates to
           :meth:`_call_multimodal_processor`, which handles both legacy (flat kwargs) and
           transformers v5 (per-modality kwargs) calling conventions.
        3. **Single-modality processor** (tokenizer, feature extractor, image/video processor):
           delegates to :meth:`_call_single_modality_processor`, which matches the processor
           type and passes the primary input as a positional argument when available.

        Args:
            modality: The modality or tuple of modalities being processed.
            processor_inputs: Dictionary of processor argument names to lists of values.
            modality_kwargs: Per-modality configuration kwargs (keys: ``"text"``, ``"image"``,
                ``"audio"``, ``"video"``).
            common_kwargs: Common kwargs passed to all processor calls (e.g. ``padding``,
                ``return_tensors``).

        Returns:
            Processor output dictionary.
        """
        if modality == "message":
            return self._process_chat_messages(processor_inputs["message"], modality_kwargs, common_kwargs)

        if isinstance(self.processor, ProcessorMixin):
            return self._call_multimodal_processor(modality, processor_inputs, modality_kwargs, common_kwargs)

        return self._call_single_modality_processor(modality, processor_inputs, modality_kwargs, common_kwargs)

    def _call_multimodal_processor(
        self,
        modality: Modality,
        processor_inputs: dict[str, list],
        modality_kwargs: dict[str, dict[str, Any]],
        common_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a :class:`ProcessorMixin` processor, handling both legacy and v5 calling conventions."""
        # Convert modality keys to processor argument names (e.g., "image" -> "images")
        processor_inputs = {MODALITY_TO_PROCESSOR_ARG.get(key, key): value for key, value in processor_inputs.items()}

        # Some transformers processors are still outdated, and don't accept common_kwargs, etc.
        if (
            self.config.model_type in {"clipseg", "whisper", "sam3"}
            or not _TRANSFORMERS_PROCESSOR_SUPPORTS_MODALITY_KWARGS
        ):
            # Check against the only valid multimodal modality for these architectures
            if modality == ("audio", "text"):
                # Audio must have priority for whisper, to correctly set padding to max_length
                kwargs = {**modality_kwargs["text"], **modality_kwargs["audio"]}
            else:
                kwargs = modality_kwargs[modality]
            return self.processor(**processor_inputs, **kwargs, **common_kwargs)

        # This is the much cleaner transformers v5 approach
        return self.processor(
            **processor_inputs,
            text_kwargs=modality_kwargs["text"],
            images_kwargs=modality_kwargs["image"],
            audio_kwargs=modality_kwargs["audio"],
            videos_kwargs=modality_kwargs["video"],
            common_kwargs=common_kwargs,
        )

    def _call_single_modality_processor(
        self,
        modality: Modality,
        processor_inputs: dict[str, list],
        modality_kwargs: dict[str, dict[str, Any]],
        common_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Call a single-modality processor (tokenizer, feature extractor, image/video processor)."""
        # Check in order: text, audio, video, image (video before image due to inheritance)
        processor_type_checks = [
            ("text", PreTrainedTokenizerBase, modality_kwargs["text"]),
            ("audio", FeatureExtractionMixin, modality_kwargs["audio"]),
            ("video", BaseVideoProcessor, modality_kwargs["video"]),
            ("image", ImageProcessingMixin, modality_kwargs["image"]),
        ]

        for modality_type, processor_class, type_kwargs in processor_type_checks:
            if not isinstance(self.processor, processor_class):
                continue

            call_kwargs = {**type_kwargs, **common_kwargs}

            # Tokenizers and feature extractors expect the primary input as the first positional arg
            if modality_type in processor_inputs:
                primary_input = processor_inputs.pop(modality_type)
                return self.processor(primary_input, **processor_inputs, **call_kwargs)
            return self.processor(**processor_inputs, **call_kwargs)

        raise RuntimeError(
            f"Could not determine how to call processor of type {type(self.processor).__name__} "
            f"for modality '{format_modality(modality)}'"
        )

    def _process_chat_messages(
        self,
        messages: list[list[MessageInput]],
        modality_kwargs: dict[str, dict[str, Any]],
        common_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Process chat messages using the processor's chat template."""
        if "message" not in self.modality_config:
            raise ValueError(
                f"The model does not support 'message' modality, but the input looks like a chat message. "
                f"Supported modalities: {list(self.modality_config.keys())}"
            )

        # Ideally we'd use the same code path for both ProcessorMixin and Tokenizers, but the latter expects
        # the text kwargs to be passed at the top level instead of in a nested "text_kwargs" dict.
        chat_template_kwargs = self.processing_kwargs.get("chat_template", {})
        if isinstance(self.processor, ProcessorMixin):
            # Transformers v5.4.0 prefers us to pass processor_kwargs as a single dict, but there's still some top level
            # kwargs that need to be hoisted out for backwards compatibility.
            if _TRANSFORMERS_APPLY_CHAT_TEMPLATE_RECOMMENDS_PROCESSOR_KWARGS:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors=common_kwargs.get("return_tensors"),
                    load_audio_from_video=modality_kwargs["video"].get("load_audio_from_video", False),
                    processor_kwargs={
                        "text_kwargs": modality_kwargs["text"],
                        "images_kwargs": modality_kwargs["image"],
                        "audio_kwargs": modality_kwargs["audio"],
                        "videos_kwargs": modality_kwargs["video"],
                        "common_kwargs": common_kwargs,
                    },
                    **chat_template_kwargs,
                )
            else:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    text_kwargs=modality_kwargs["text"],
                    images_kwargs=modality_kwargs["image"],
                    audio_kwargs=modality_kwargs["audio"],
                    videos_kwargs=modality_kwargs["video"],
                    common_kwargs=common_kwargs,
                    **chat_template_kwargs,
                )

        # apply_chat_template expects padding/truncation/max_length/return_tensors as top-level kwargs,
        # not nested inside tokenizer_kwargs or common_kwargs, so we hoist them out.
        top_level_kwarg_names = {"padding", "truncation", "max_length", "return_tensors"}
        top_level_kwargs = {key: common_kwargs.pop(key) for key in top_level_kwarg_names & common_kwargs.keys()}
        top_level_kwargs |= {
            key: modality_kwargs["text"].pop(key) for key in top_level_kwarg_names & modality_kwargs["text"].keys()
        }

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            tokenizer_kwargs=modality_kwargs["text"],
            common_kwargs=common_kwargs,
            **top_level_kwargs,
            **chat_template_kwargs,
        )

    def _get_prompt_length(self, prompt: str, **kwargs) -> int | None:
        """Return the length of the prompt in tokens, excluding any trailing special token.

        Returns None if the processor does not produce ``input_ids``.
        """
        cache_key = (prompt, *sorted(kwargs.items()))
        if cache_key in self._prompt_length_mapping:
            return self._prompt_length_mapping[cache_key]

        tokenized_prompt = self.preprocess([prompt], **kwargs)
        if "input_ids" not in tokenized_prompt:
            self._prompt_length_mapping[cache_key] = None
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a trailing special token (EOS, SEP, etc.), exclude it from the prompt length
        tokenizer = self.tokenizer
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if tokenizer is not None and hasattr(tokenizer, "all_special_ids") and last_token in tokenizer.all_special_ids:
            prompt_length -= 1
        self._prompt_length_mapping[cache_key] = prompt_length
        return prompt_length

    def _load_config(
        self, model_name_or_path: str, backend: str, config_kwargs: dict[str, Any]
    ) -> tuple[PeftConfig | PretrainedConfig, bool]:
        """Loads the transformers or PEFT configuration

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            config_kwargs (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers config.

        Returns:
            tuple[PeftConfig | PretrainedConfig, bool]: The model configuration and a boolean indicating whether the model is a PEFT model.
        """
        adapter_config_file = find_adapter_config_file(
            model_name_or_path,
            cache_dir=config_kwargs.get("cache_dir"),
            token=config_kwargs.get("token"),
            revision=config_kwargs.get("revision"),
            subfolder=config_kwargs.get("subfolder", ""),
            local_files_only=config_kwargs.get("local_files_only", False),
        )
        if adapter_config_file is not None:
            if backend != "torch":
                # TODO: Consider following these steps automatically so we can load PEFT models with other backends
                raise ValueError(
                    "PEFT models can currently only be loaded with the `torch` backend. "
                    'To use other backends, load the model with `backend="torch"`, call `model.transformers_model.merge_and_unload()`, '
                    "save that model with `model.save_pretrained()` and then load the model with the desired backend."
                )
            if not is_peft_available():
                raise ImportError(
                    "Loading a PEFT model requires installing the `peft` package. You can install it via `pip install peft`."
                )
            from peft import PeftConfig

            return PeftConfig.from_pretrained(model_name_or_path, **config_kwargs), True

        return AutoConfig.from_pretrained(model_name_or_path, **config_kwargs), False

    def _load_model(
        self,
        model_name_or_path: str,
        transformer_task: Literal[
            "feature-extraction", "sequence-classification", "text-generation", "any-to-any", "fill-mask"
        ],
        config: PeftConfig | PretrainedConfig,
        backend: str,
        is_peft_model: bool,
        **model_kwargs,
    ) -> PreTrainedModel:
        """Loads the transformers or PEFT model into the `auto_model` attribute

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            config ("PeftConfig" | PretrainedConfig): The model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            is_peft_model (bool): Whether the model is a PEFT model.
            model_kwargs (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers model.
        """
        if backend == "torch":
            # When loading a PEFT model, we load the base model first. The revision
            # (e.g. "main") refers to the adapter checkpoint, not the base model, so
            # we must not pass it to the base model's from_pretrained.
            if is_peft_model:
                model_kwargs.pop("revision", None)

            if transformer_task == "feature-extraction":
                model = self._load_encoder_only_model(model_name_or_path, config, **model_kwargs)
                if model is not None:
                    return model

            model_cls = TRANSFORMER_TASK_TO_AUTO_MODEL[transformer_task]
            return model_cls.from_pretrained(model_name_or_path, config=config, **model_kwargs)
        elif backend == "onnx":
            return load_onnx_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_kwargs,
            )
        elif backend == "openvino":
            return load_openvino_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _load_encoder_only_model(
        self,
        model_name_or_path: str,
        config: PretrainedConfig,
        **model_kwargs,
    ) -> PreTrainedModel | None:
        """Load encoder-only variants for encoder-decoder architectures.

        Checks :data:`_ENCODER_ONLY_MODELS` for standard mappings and handles a few special cases
        (T5Gemma, T5Gemma2) that require extra configuration before loading.

        Returns the loaded model, or None if the config doesn't match any encoder-only architecture.
        """

        def _load_encoder(model_cls, load_config=None, **extra_class_attrs):
            with set_temporary_class_attrs(
                model_cls, _keys_to_ignore_on_load_unexpected=["decoder.*"], **extra_class_attrs
            ):
                return model_cls.from_pretrained(model_name_or_path, config=load_config or config, **model_kwargs)

        # Special cases that need extra handling before/during loading
        if isinstance(config, T5GemmaConfig):
            from transformers import T5GemmaEncoderModel

            config.is_encoder_decoder = False
            return _load_encoder(T5GemmaEncoderModel)

        if isinstance(config, T5Gemma2Config):
            from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder

            # T5Gemma2Encoder expects the encoder sub-config, not the full composite config
            return _load_encoder(T5Gemma2Encoder, load_config=config.encoder, base_model_prefix="model.encoder")

        # Standard encoder-only models from the registry
        for config_cls, module_path, class_name in _ENCODER_ONLY_MODELS:
            if isinstance(config, config_cls):
                encoder_cls = getattr(importlib.import_module(module_path), class_name)
                return _load_encoder(encoder_cls)

        return None

    def infer_modalities(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin
        | PreTrainedTokenizerBase
        | FeatureExtractionMixin
        | BaseVideoProcessor
        | ImageProcessingMixin,
    ) -> tuple[ModalityConfig, str]:
        """Infer the modality configuration and module output name from the model and processor.

        First checks :meth:`infer_modalities_edge_cases` for hard-coded overrides, then falls back
        to general inference based on the processor type and model forward signature.
        """
        default_modality_config, default_module_output_name = TRANSFORMER_TASK_DEFAULTS[self.transformer_task]
        default_method_output_name = default_modality_config["text"]["method_output_name"]

        if (result := self.infer_modalities_edge_cases(model, processor)) is not None:
            modality_config, module_output_name = result
            # Edge-case models may also support the message format via chat templates
            if hasattr(processor, "chat_template") and processor.chat_template is not None:
                if "message" not in modality_config:
                    modality_config["message"] = {
                        **modality_config.get(
                            "text", {"method": "forward", "method_output_name": default_method_output_name}
                        ),
                        "format": self.input_formatter.message_format,
                    }
            return modality_config, module_output_name

        modalities = self.infer_modalities_from_processor(processor)
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            modalities.append("message")

        # Inspect forward to see if it can be used for all modalities, or if we need modality-specific methods.
        # If we can't inspect the method return type, we assume it has the default output name.
        output_fields = self._get_method_output_fields(model.forward)
        if output_fields is None or default_method_output_name in output_fields:
            modality_config: ModalityConfig = {}
            for modality in modalities:
                entry = ModalityParams(method="forward", method_output_name=default_method_output_name)
                if modality == "message":
                    entry["format"] = self.input_formatter.message_format
                modality_config[modality] = entry
            return modality_config, default_module_output_name

        # For feature-extraction, if there's no 'last_hidden_state', we can check for modality-specific methods like get_..._features
        if self.transformer_task == "feature-extraction":
            modality_config: ModalityConfig = {}
            for modality in modalities:
                if modality == "message":
                    continue

                method_name = f"get_{modality}_features"
                if hasattr(model, method_name):
                    method = getattr(model, method_name)
                    method_output_fields = self._get_method_output_fields(method)
                    if method_output_fields and "pooler_output" in method_output_fields:
                        modality_config[modality] = {"method": method_name, "method_output_name": "pooler_output"}
                    else:
                        modality_config[modality] = {"method": method_name, "method_output_name": None}

            return modality_config, "sentence_embedding"

        return {
            modality: {"method": "forward", "method_output_name": default_method_output_name}
            for modality in modalities
        }, default_module_output_name

    def infer_modalities_edge_cases(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin | PreTrainedTokenizerBase | FeatureExtractionMixin | ImageProcessingMixin,
    ) -> tuple[ModalityConfig, str] | None:
        """Return a ``(modality_config, module_output_name)`` for model types that cannot be handled
        by the general :meth:`infer_modalities` inference path, or ``None`` to fall through.

        Looks up the model type in the task-specific edge case configs from
        :data:`_EDGE_CASE_MODALITY_CONFIGS`. For entries that require output name validation
        (transformers v4/v5 compat), resolves each modality's ``method_output_name`` against the
        actual model method via :meth:`_infer_method_output_name`.
        """
        task_edge_cases = _EDGE_CASE_MODALITY_CONFIGS.get(self.transformer_task)
        if task_edge_cases is None:
            return None

        entry = task_edge_cases.get(model.config.model_type)
        if entry is None:
            return None

        raw_config, module_output_name, validate_output_names = entry
        if not validate_output_names:
            return raw_config, module_output_name

        modality_config: ModalityConfig = {}
        for modality, params in raw_config.items():
            if not hasattr(model, params["method"]):
                logger.warning_once(
                    f"Model does not have method {params['method']!r} for modality {modality!r}. Skipping."
                )
                continue
            method = getattr(model, params["method"])
            modality_config[modality] = {
                "method": params["method"],
                "method_output_name": self._infer_method_output_name(params["method_output_name"], method),
            }
        return modality_config, module_output_name

    def infer_modalities_from_processor(
        self,
        processor: ProcessorMixin | PreTrainedTokenizerBase | FeatureExtractionMixin | ImageProcessingMixin,
    ) -> list[Modality]:
        """Determine which modalities the processor supports by inspecting its attributes or type."""
        processor_attribute_mapping: dict[str, Modality] = {
            "tokenizer": "text",
            "image_processor": "image",
            "feature_extractor": "audio",
            "video_processor": "video",
        }
        if isinstance(processor, ProcessorMixin):
            processor_attributes = self._get_processor_attributes() or {}
            return [
                modality_name
                for processor_attribute, modality_name in processor_attribute_mapping.items()
                if processor_attribute in processor_attributes
            ]

        modality_checks: dict[Modality, type] = {
            "text": PreTrainedTokenizerBase,
            "audio": FeatureExtractionMixin,
            "video": BaseVideoProcessor,
            "image": ImageProcessingMixin,
        }
        for modality_name, processor_class in modality_checks.items():
            if isinstance(processor, processor_class):
                return [modality_name]

        logger.warning(
            f"Could not determine modalities from processor of type {type(processor).__name__}. "
            "Returning an empty modality list."
        )
        return []

    def _get_processor_attributes(self) -> list[str] | None:
        """Get the attributes of the processor if available. Will be removed in the future as transformers v5
        becomes the minimum requirement.

        Returns:
            list[str] | None: The processor attribute names, or None if not available.
        """
        if hasattr(self.processor, "get_attributes"):  # Transformers v5+
            return self.processor.get_attributes()
        elif hasattr(self.processor, "attributes"):  # Transformers v4
            return self.processor.attributes
        return None

    @staticmethod
    def _get_method_output_fields(method: Callable) -> list[str] | None:
        """Extract the output field names from a method's return type annotation.

        Args:
            method (Callable): The method to inspect.

        Returns:
            list[str] | None: List of output field names, or None if not found.
        """

        def find_model_output_class(type_annotation):
            if isinstance(type_annotation, type) and issubclass(type_annotation, ModelOutput):
                return type_annotation
            for sub_annotation in get_args(type_annotation):
                if (result := find_model_output_class(sub_annotation)) is not None:
                    return result
            return None

        try:
            return_annotation = get_type_hints(method).get("return", None)
        except Exception:
            return None
        output_class = find_model_output_class(return_annotation)
        if output_class is None:
            return None
        return [field.name for field in fields(output_class)]

    @staticmethod
    def _infer_method_output_name(method_output_name: str, method: Callable) -> str | None:
        """Validate that ``method_output_name`` is present in the method's return type annotation.

        Returns the name if found, or ``None`` if the method's output type does not include it.
        Primarily needed for transformers v4 compatibility: v5 often allows ``pooler_output``
        from ``get_..._features`` methods, but v4 didn't use ``BaseModelOutputWithPooling`` yet.
        """
        output_fields = Transformer._get_method_output_fields(method) or []
        if method_output_name in output_fields:
            return method_output_name
        return None

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        """Save the model, processor, and module config to ``output_path``."""
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.processor.save_pretrained(output_path)
        self.save_config(output_path)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        # Loading arguments
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        # Module-specific arguments
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> Self:
        """Load a Transformer module from a pretrained model directory or Hugging Face model name."""
        init_kwargs = cls._load_init_kwargs(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            processor_kwargs=processor_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )
        return cls(model_name_or_path=model_name_or_path, **init_kwargs)

    @classmethod
    def _load_init_kwargs(
        cls,
        model_name_or_path: str,
        # Loading arguments
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        # Module-specific arguments
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``__init__`` by merging config file, hub kwargs, and caller overrides.

        Priority (highest to lowest): caller kwargs > hub kwargs > config file values.
        """
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_dir": cache_folder,  # Transformers uses `cache_dir` instead of `cache_folder`
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }

        # 3rd priority: config file
        # Config files may use the old key names (model_args, tokenizer_args, config_args) for backwards compat.
        # We normalize them to the new names (model_kwargs, processor_kwargs, config_kwargs) here so we don't
        # trigger deprecation warnings from the transformer_kwargs_decorator on Transformer.__init__.
        _OLD_TO_NEW = {
            "model_args": "model_kwargs",
            "tokenizer_args": "processor_kwargs",
            "config_args": "config_kwargs",
        }
        for old_name, new_name in _OLD_TO_NEW.items():
            if old_name in config:
                config[new_name] = config.pop(old_name)
            config.setdefault(new_name, {})

        # 2nd priority: hub_kwargs
        config["model_kwargs"].update(hub_kwargs)
        config["processor_kwargs"].update(hub_kwargs)
        config["config_kwargs"].update(hub_kwargs)

        # 1st priority: kwargs passed to SentenceTransformer
        if model_kwargs:
            config["model_kwargs"].update(model_kwargs)
        if processor_kwargs:
            config["processor_kwargs"].update(processor_kwargs)
        if config_kwargs:
            config["config_kwargs"].update(config_kwargs)

        return {**config, "backend": backend}

    @classmethod
    def load_config(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        config_filename: str | None = None,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> dict[str, Any]:
        """Load the module config, trying several legacy config filenames for backward compatibility.

        Handles deserialization of ``modality_config`` tuple keys (stored as comma-separated strings
        in JSON) and strips ``trust_remote_code`` from all sub-dicts for security.
        """
        config_filenames = (
            [config_filename]
            if config_filename
            else [
                "sentence_bert_config.json",
                "sentence_roberta_config.json",
                "sentence_distilbert_config.json",
                "sentence_camembert_config.json",
                "sentence_albert_config.json",
                "sentence_xlm-roberta_config.json",
                "sentence_xlnet_config.json",
            ]
        )
        for config_filename in config_filenames:
            config = super().load_config(
                model_name_or_path=model_name_or_path,
                subfolder=subfolder,
                config_filename=config_filename,
                token=token,
                cache_folder=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
            )
            if config:
                break

        # Don't allow configs to set trust_remote_code
        for key in (
            "model_args",
            "model_kwargs",
            "tokenizer_args",
            "processor_kwargs",
            "config_args",
            "config_kwargs",
        ):
            if key in config and "trust_remote_code" in config[key]:
                config[key].pop("trust_remote_code")

        if "modality_config" in config:
            # Deserialize modality_config keys: "+" separates tuple modalities (e.g. "image+text")
            valid_single_modalities = {"text", "image", "audio", "video", "message"}
            deserialized_modality_config = {}
            for modality_key, params in config["modality_config"].items():
                if "+" in modality_key:
                    parts = tuple(modality_key.split("+"))
                    invalid = [p for p in parts if p not in valid_single_modalities]
                    if invalid:
                        logger.warning(
                            f"Ignoring unknown modality components {invalid!r} in modality_config key {modality_key!r}."
                        )
                        continue
                    deserialized_modality_config[parts] = params
                else:
                    if modality_key not in valid_single_modalities:
                        logger.warning(f"Ignoring unknown modality key {modality_key!r} in modality_config.")
                        continue
                    deserialized_modality_config[modality_key] = params
            config["modality_config"] = deserialized_modality_config

        else:
            # This method is only called if this model has a modules.json, i.e. it's already been saved
            # with Sentence Transformers. So, if modality_config is not in the config, we can assume it
            # was saved with an older version where Transformer was text-only, and so we can set the
            # modality_config accordingly for backward compatibility. Otherwise, we might infer and use
            # the 'message' format and get different results than what previously worked.
            config["modality_config"], config["module_output_name"] = cls._get_default_modality_config(config)

        return config

    @staticmethod
    def _get_default_modality_config(config: dict[str, Any]) -> tuple[ModalityConfig, str]:
        """Get the default modality configuration for the current transformer task.

        Returns:
            tuple[ModalityConfig, str]: A tuple of (modality_config, module_output_name).
                The modality_config maps modality keys to dicts with 'method' and 'method_output_name'.
                The module_output_name is the name of the output feature this module creates.
        """
        return TRANSFORMER_TASK_DEFAULTS[config.get("transformer_task", "feature-extraction")]

    def get_config_dict(self) -> dict[str, Any]:
        """Return the config dict for serialization, with tuple modality keys joined as plus-separated strings."""
        config_dict = super().get_config_dict()
        config_dict["modality_config"] = {
            format_modality(modality): params for modality, params in self.modality_config.items()
        }
        if not self.processing_kwargs:
            config_dict.pop("processing_kwargs", None)
        if self.unpad_inputs is None:
            config_dict.pop("unpad_inputs", None)
        return config_dict

    def __repr__(self) -> str:
        return f"Transformer({dict(self.get_config_dict(), architecture=self.model.__class__.__name__)})"
