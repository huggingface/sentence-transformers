from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Literal, TypedDict, get_args, get_type_hints

import torch
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
    MoonshineConfig,
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
    TimmWrapperConfig,
    UdopConfig,
    UMT5Config,
    WhisperConfig,
)
from transformers.utils import ModelOutput
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file

from sentence_transformers.backend import load_onnx_model, load_openvino_model
from sentence_transformers.base.modality import InputFormatter, format_modality
from sentence_transformers.base.modality_types import (
    MODALITY_TO_PROCESSOR_ARG,
    MessageFormat,
    MessageInput,
    Modality,
    PairInput,
    SingleInput,
)
from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.util.decorators import transformer_kwargs_decorator

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


logger = logging.getLogger(__name__)


if TYPE_CHECKING and is_peft_available():
    from peft import PeftConfig

TransformerTask = Literal[
    "feature-extraction", "sequence-classification", "text-generation", "any-to-any", "fill-mask"
]


class ModalityParams(TypedDict):
    method: str
    method_output_name: str | None


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


class Transformer(InputModule):
    """Hugging Face AutoModel wrapper for generating embeddings, scores, or logits.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    TODO: Rewrite this docstring, needs to be more informational

    Args:
        model_name_or_path: Hugging Face model name or path
            (https://huggingface.co/models).
        transformer_task: The task to load the model for. Can be
            ``"feature-extraction"``, ``"sequence-classification"``,
            ``"text-generation"``, ``"any-to-any"``, or ``"fill-mask"``.
        model_kwargs: Keyword arguments forwarded to
            ``AutoModel.from_pretrained`` when loading the model.
        processor_kwargs: Keyword arguments forwarded to
            ``AutoProcessor.from_pretrained`` when loading the
            processor/tokenizer.
        config_kwargs: Keyword arguments forwarded to
            ``AutoConfig.from_pretrained`` when loading the config.
        processing_kwargs: Keyword arguments applied when *calling* the
            processor during preprocessing. This is a nested dict whose keys
            are modality names (``"text"``, ``"audio"``, ``"image"``,
            ``"video"``), ``"common"`` for kwargs shared across all
            modalities, or ``"chat_template"`` for kwargs forwarded to
            ``apply_chat_template`` (e.g.
            ``{"add_generation_prompt": True}``). Modality and common kwargs
            override the built-in defaults. Saved to and loaded from the
            model configuration file.
        backend: Backend used for model inference. Can be ``"torch"``,
            ``"onnx"``, or ``"openvino"``. Default is ``"torch"``.
        modality_config: Optional custom modality configuration mapping
            modality names to method and output name dicts.
        module_output_name: The name of the output feature this module
            creates. Required when ``modality_config`` is provided.
        message_format: How to handle message-format inputs. Default is
            ``"auto"``.
        max_seq_length: Truncate any inputs longer than this value. Prefer
            setting ``model_max_length`` via ``processor_kwargs`` instead.
        do_lower_case: If true, lowercases the input (independent of whether
            the model is cased or not). Rarely needed.
        tokenizer_name_or_path: Name or path of the tokenizer. When
            None, then ``model_name_or_path`` is used. Deprecated.
    """

    config_file_name: str = "sentence_bert_config.json"
    config_keys: list[str] = [
        "transformer_task",
        "modality_config",
        "module_output_name",
        "message_format",
        "processing_kwargs",
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
        processing_kwargs: dict[str, dict[str, Any]] | None = None,  # TODO: Warn if unused kwargs provided
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        modality_config: ModalityConfig | None = None,
        module_output_name: str | None = None,
        message_format: MessageFormat = "auto",
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
        self.processing_kwargs: dict[str, dict[str, Any]] = processing_kwargs or {}
        self.backend = backend
        self.message_format = message_format
        self.do_lower_case = do_lower_case
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
        }

        if max_seq_length is not None and "model_max_length" not in processor_kwargs:
            processor_kwargs["model_max_length"] = max_seq_length
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

        self.input_formatter = InputFormatter(
            model_type=self.config.model_type, message_format=self.message_format, processor=self.processor
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
        logger.info(f"Active modality config: {self.modality_config}")

        if tokenizer_name_or_path is not None:
            logger.warning(
                "The `tokenizer_name_or_path` argument is deprecated and will be removed in a future version. "
                "Please use the same path for the model and processor."
            )
            self.model.config.tokenizer_class = self.processor.__class__.__name__

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
            if modality_key in self.processing_kwargs:
                modality_kwargs[modality_key].update(self.processing_kwargs[modality_key])

        prompt_length = None

        modality, processor_inputs, extra_modality_kwargs = self.input_formatter.parse_inputs(inputs)

        for modality_key, extra_kwargs in extra_modality_kwargs.items():
            modality_kwargs[modality_key].update(extra_kwargs)

        # Always convert to the message format if it's supported, since it's most flexible with e.g. defaults
        if "message" in self.modality_config and modality != "message":
            modality, processor_inputs = self.input_formatter.batch_to_messages(modality, processor_inputs)
        elif modality not in self.modality_config:
            raise ValueError(
                f"Modality '{format_modality(modality)}' is not supported by this model. "
                f"Supported modalities: {', '.join(format_modality(m) for m in sorted(self.modality_config.keys(), key=str))}"
            )

        # Incorporate prompt into inputs if applicable
        if prompt and modality == "message":
            processor_inputs["message"] = self.input_formatter.prepend_prompt_to_messages(
                processor_inputs["message"], prompt
            )
            # Models relying on the message format don't support excluding prompt tokens in mean pooling,
            # so we don't track prompt length.
        elif prompt and modality == "text":
            processor_inputs["text"] = self.input_formatter.prepend_prompt_to_texts(processor_inputs["text"], prompt)
            prompt_length = self._get_prompt_length(prompt, **kwargs)

        processor_output = self._call_processor(modality, processor_inputs, modality_kwargs, common_kwargs)
        processor_output["modality"] = modality
        if prompt_length is not None:
            processor_output["prompt_length"] = prompt_length

        return processor_output

    def forward(self, features: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Forward pass through the transformer model.

        Dispatches to the appropriate model method based on the ``modality`` key in ``features``
        and writes the result into ``features[self.module_output_name]``.

        Args:
            features: Input features dictionary produced by :meth:`preprocess`. Must contain the
                keys expected by the underlying model (e.g. ``input_ids``, ``pixel_values``, etc.)
                and a ``modality`` key indicating which modality config to use.
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

        model_output = model_method(**filtered_kwargs)

        embedding = model_output
        if method_output_name is not None:
            for output_key in method_output_name:
                try:
                    embedding = embedding[output_key]
                except (KeyError, TypeError):
                    # Some models (e.g. chinese_clip) only expose output fields via attribute access,
                    # not dictionary-style indexing. See https://github.com/huggingface/transformers/issues/44079
                    embedding = getattr(embedding, output_key)

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

        Args:
            modality: The modality or tuple of modalities being processed
            processor_inputs: Dictionary of processor argument names to lists of values
            modality_kwargs: Configuration kwargs for each modality type
            common_kwargs: Common kwargs to pass to all processor calls

        Returns:
            Processor output dictionary
        """
        if modality == "message":
            return self._process_chat_messages(processor_inputs["message"], modality_kwargs, common_kwargs)

        # Multi-modal processor: pass modality-specific kwargs
        if isinstance(self.processor, ProcessorMixin):
            # Convert modality keys to processor argument names (e.g., "image" -> "images")
            processor_inputs = {
                MODALITY_TO_PROCESSOR_ARG.get(key, key): value for key, value in processor_inputs.items()
            }

            # Some transformers processors are still outdated, and don't accept common_kwargs, etc.
            if self.config.model_type in {"clipseg", "whisper", "sam3"}:
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

        # Single-modality processor: determine type and call appropriately
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
                    modality_config["message"] = modality_config.get(
                        "text", {"method": "forward", "method_output_name": default_method_output_name}
                    )
            return modality_config, module_output_name

        modalities = self.infer_modalities_from_processor(processor)
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            modalities.append("message")

        # Inspect forward to see if it can be used for all modalities, or if we need modality-specific methods.
        # If we can't inspect the method return type, we assume it has the default output name.
        output_fields = self._get_method_output_fields(model.forward)
        if output_fields is None or default_method_output_name in output_fields:
            return {
                modality: {"method": "forward", "method_output_name": default_method_output_name}
                for modality in modalities
            }, default_module_output_name

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
            deserialized_modality_config = {}
            for modality_key, params in config["modality_config"].items():
                if "+" in modality_key:
                    deserialized_modality_config[tuple(modality_key.split("+"))] = params
                else:
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
        return config_dict

    def __repr__(self) -> str:
        return f"Transformer({dict(self.get_config_dict(), architecture=self.model.__class__.__name__)})"
