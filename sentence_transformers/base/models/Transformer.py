from __future__ import annotations

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
from sentence_transformers.base.models.InputModule import InputModule
from sentence_transformers.base.models.modality_utils import (
    ArrayInputs,
    DictInputs,
    ImageInputs,
    InputFormatter,
    MessageFormat,
    Modality,
    PairStrInputs,
    StrInputs,
)

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

TransformerTask = Literal["feature-extraction", "sequence-classification", "text-generation", "fill-mask"]


class ModalityParams(TypedDict):
    method: str
    method_output_name: str | None


ModalityConfig = dict[Modality, ModalityParams]

TRANSFORMER_TASK_TO_AUTO_MODEL: dict[TransformerTask, Any] = {
    "feature-extraction": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "text-generation": AutoModelForCausalLM,
    "fill-mask": AutoModelForMaskedLM,
}

# Maps transformer tasks -> modalities -> methods -> model output fields -> module feature names
# Structure: {task: {modality: {method_name: {model_output_field: module_feature_name}}}}
# TODO: How about defaults? E.g. I want to support "image" for a model that traditionally only has ("text", "image")
# by defaulting to a "text" input like an empty string or just a "<image>" token?
INFER_MODALITY_CONFIG: dict[
    TransformerTask, dict[Modality | Literal["multimodal"], dict[str, dict[str | None, str]]]
] = {
    "feature-extraction": {
        "text": {
            "get_text_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "text_embeds": "sentence_embedding"},
        },
        "image": {
            "get_image_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "image_embeds": "sentence_embedding"},
        },
        "audio": {
            "get_audio_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "audio_embeds": "sentence_embedding"},
        },
        "video": {
            "get_video_features": {None: "sentence_embedding"},
            "forward": {"last_hidden_state": "token_embeddings", "video_embeds": "sentence_embedding"},
        },
        "multimodal": {"forward": {"last_hidden_state": "token_embeddings"}},
    },
    "sequence-classification": {
        "text": {"forward": {"logits": "scores"}},
        "image": {"forward": {"logits": "scores"}},
        "audio": {"forward": {"logits": "scores"}},
        "video": {"forward": {"logits": "scores"}},
        "multimodal": {"forward": {"logits": "scores"}},
    },
    "text-generation": {
        "text": {"forward": {"logits": "causal_logits"}},
        "image": {"forward": {"logits": "causal_logits"}},
        "audio": {"forward": {"logits": "causal_logits"}},
        "video": {"forward": {"logits": "causal_logits"}},
        "multimodal": {"forward": {"logits": "causal_logits"}},
    },
    "fill-mask": {
        "text": {"forward": {"logits": "token_embeddings"}},
        "image": {"forward": {"logits": "token_embeddings"}},
        "audio": {"forward": {"logits": "token_embeddings"}},
        "video": {"forward": {"logits": "token_embeddings"}},
        "multimodal": {"forward": {"logits": "token_embeddings"}},
    },
}

TRANSFORMER_TASK_TO_METHOD_OUTPUT_NAME = {
    "feature-extraction": "last_hidden_state",
    "sequence-classification": "logits",
    "text-generation": "logits",
    "fill-mask": "logits",
}

TRANSFORMER_TASK_TO_DEFAULT_MODULE_OUTPUT_NAME = {
    "feature-extraction": "token_embeddings",
    "sequence-classification": "scores",
    "text-generation": "causal_logits",
    "fill-mask": "token_embeddings",
}

DEFAULT_MODALITY_CONFIG_MODULE_OUTPUT_NAME: dict[TransformerTask, tuple[ModalityConfig, str]] = {
    "feature-extraction": (
        {
            "text": {
                "method": "forward",
                "method_output_name": "last_hidden_state",
            },
        },
        "token_embeddings",
    ),
    "sequence-classification": (
        {
            "text": {
                "method": "forward",
                "method_output_name": "logits",
            },
        },
        "scores",
    ),
    "text-generation": (
        {
            "text": {
                "method": "forward",
                "method_output_name": "logits",
            },
        },
        "causal_logits",
    ),
    "fill-mask": (
        {
            "text": {
                "method": "forward",
                "method_output_name": "logits",
            },
        },
        "token_embeddings",
    ),
}


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
    """Hugging Face AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    Args:
        model_name_or_path: Hugging Face models name
            (https://huggingface.co/models)
        max_seq_length: Truncate any inputs longer than max_seq_length
        model_args: Keyword arguments passed to the Hugging Face
            Transformers model
        tokenizer_args: Keyword arguments passed to the Hugging Face
            Transformers tokenizer
        config_args: Keyword arguments passed to the Hugging Face
            Transformers config
        cache_dir: Cache dir for Hugging Face Transformers to store/load
            models
        do_lower_case: If true, lowercases the input (independent if the
            model is cased or not)
        tokenizer_name_or_path: Name or path of the tokenizer. When
            None, then model_name_or_path is used
        backend: Backend used for model inference. Can be `torch`, `onnx`,
            or `openvino`. Default is `torch`.
    """

    config_file_name: str = "sentence_bert_config.json"
    # TODO: Could we get rid of "max_seq_length" and "do_lower_case" here? Or are they not saved?
    config_keys: list[str] = [
        "transformer_task",
        "modality_config",
        "module_output_name",
        "message_format",
    ]  # , "max_seq_length", "do_lower_case"]
    save_in_root: bool = True

    # TODO: Replace model_args with model_kwargs, perhaps replace tokenizer_args with processing_kwargs/processor_kwargs, config_args with config_kwargs?
    # TODO: Perhaps remove do_lower_case and put that in tokenizer_args?
    # TODO: Idem for max_seq_length?
    # TODO: Fully deprecate tokenizer_name_or_path? Nobody (should) load a model with a different processor than model_name_or_path
    def __init__(
        self,
        model_name_or_path: str,
        transformer_task: TransformerTask = "feature-extraction",
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str | None = None,
        backend: str = "torch",
        modality_config: ModalityConfig | None = None,
        module_output_name: str | None = None,
        message_format: MessageFormat = "auto",
    ) -> None:
        super().__init__()
        self.transformer_task: TransformerTask = transformer_task
        if transformer_task not in TRANSFORMER_TASK_TO_AUTO_MODEL:
            raise ValueError(
                f"Unsupported transformer_task '{transformer_task}'. Supported tasks are: {list(TRANSFORMER_TASK_TO_AUTO_MODEL.keys())}"
            )
        # TODO: Reorder the args in __init__ body?
        self.do_lower_case = do_lower_case
        self.backend = backend
        self.message_format = message_format
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}
        self._prompt_length_mapping = {}

        config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)

        if (
            transformer_task == "sequence-classification"
            and "num_labels" not in config_args
            and (
                config.architectures is None
                or not any([arch.endswith("ForSequenceClassification") for arch in config.architectures])
            )
        ):
            # If we're loading a model for sequence-classification, but the base architecture is not for sequence-classification,
            # and num_labels is not specified, we default to 1 label for CrossEncoder-like behavior
            config.num_labels = 1

        self.model = self._load_model(
            model_name_or_path, transformer_task, config, cache_dir, backend, is_peft_model, **model_args
        )

        # Get the signature of the auto_model's forward method to pass only the expected arguments from `features`,
        # plus some common values like "input_ids", "attention_mask", etc.
        # TODO: Cache (or only run) all signature calls like this
        model_forward_params = list(inspect.signature(self.model.forward).parameters)
        self.model_forward_params = set(model_forward_params) | {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "inputs_embeds",
        }

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # Shrink the tokenizer model_max_length if the model config has a smaller max_position_embeddings
        if self.tokenizer is not None:
            # NOTE: xlnet uses a hardcoded config.max_position_embeddings != -1 to denote no max_length
            if (
                "model_max_length" not in tokenizer_args
                and hasattr(self.config, "max_position_embeddings")
                and self.config.max_position_embeddings != -1
            ):
                self.tokenizer.model_max_length = min(
                    self.tokenizer.model_max_length, self.config.max_position_embeddings
                )

            # TODO: self.processor.is_fast might not work
            if do_lower_case:
                # TODO: Transformers v5 only has fast tokenizers
                if self.tokenizer.is_fast:

                    def has_lowercase(normalizer):
                        if normalizer is None:
                            return False
                        if isinstance(normalizer, Lowercase):
                            return True
                        if isinstance(normalizer, Sequence):
                            return any(isinstance(n, Lowercase) for n in normalizer)
                        return False

                    normalizer = self.tokenizer.backend_tokenizer.normalizer
                    if not has_lowercase(normalizer):
                        new_normalizers = [Lowercase()]
                        if isinstance(normalizer, Sequence):
                            new_normalizers += list(normalizer)
                        elif normalizer is not None:
                            new_normalizers.append(normalizer)
                        self.tokenizer.backend_tokenizer.normalizer = Sequence(new_normalizers)
                else:
                    self.processor.do_lower_case = do_lower_case

        # Create input formatter for handling input parsing and message format conversion
        self.input_formatter = InputFormatter(
            model_type=config.model_type, message_format=self.message_format, processor=self.processor
        )

        """
        # No max_seq_length set. Try to infer from model
        # TODO: self.processor.model_max_length might not work
        if max_seq_length is None:
            if (
                hasattr(self.model, "config")
                and hasattr(self.model.config, "max_position_embeddings")
                and hasattr(self.processor, "model_max_length")
            ):
                max_seq_length = min(self.model.config.max_position_embeddings, self.processor.model_max_length)

        self.max_seq_length = max_seq_length
        """

        if modality_config is not None:
            self.modality_config = modality_config
            if module_output_name is None:
                raise ValueError(
                    "Loading the Transformer module with a custom modality_config requires also providing "
                    "module_output_name with the name of the output feature that this module should create, "
                    'for example "token_embeddings" or "sentence_embedding".'
                )
            self.module_output_name = module_output_name
            # TODO: Check if modality_config has the correct format
        else:
            self.modality_config, self.module_output_name = self.infer_modalities(self.model, self.processor)
        logger.info(f"Inferred modalities: {self.modality_config}")

        # TODO: Do we need this? Perhaps even remove tokenizer_name_or_path?
        if tokenizer_name_or_path is not None:
            self.model.config.tokenizer_class = self.processor.__class__.__name__

    @property
    def max_seq_length(self) -> int | None:
        if self.tokenizer is not None:
            return self.tokenizer.model_max_length

        # Get text config, e.g. for multi-modal models
        try:
            text_config = self.model.config.get_text_config()
        except AttributeError:
            text_config = self.model.config

        if hasattr(text_config, "max_position_embeddings"):
            return text_config.max_position_embeddings
        return None

    @max_seq_length.setter
    def max_seq_length(self, value: int | None) -> None:
        if self.tokenizer is not None:
            self.tokenizer.model_max_length = value

    @property
    def auto_model(self) -> PreTrainedModel:
        return self.model

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def modalities(self) -> list[str]:
        """
        modalities = self.model.input_modalities
        if isinstance(modalities, str):
            modalities = [modalities]
        elif isinstance(modalities, tuple):
            modalities = list(modalities)
        if hasattr(self.processor, "chat_template") and self.processor.chat_template is not None:
            modalities.append("message")
        return modalities
        """
        return list(self.modality_config.keys())

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if isinstance(self.processor, PreTrainedTokenizerBase):
            return self.processor
        return getattr(self.processor, "tokenizer", None)

    def _load_config(
        self, model_name_or_path: str, cache_dir: str | None, backend: str, config_args: dict[str, Any]
    ) -> tuple[PeftConfig | PretrainedConfig, bool]:
        """Loads the transformers or PEFT configuration

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            config_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers config.

        Returns:
            tuple[PretrainedConfig, bool]: The model configuration and a boolean indicating whether the model is a PEFT model.
        """
        if (
            find_adapter_config_file(
                model_name_or_path,
                cache_dir=cache_dir,
                token=config_args.get("token"),
                revision=config_args.get("revision"),
                local_files_only=config_args.get("local_files_only", False),
            )
            is not None
        ):
            if not is_peft_available():
                raise Exception(
                    "Loading a PEFT model requires installing the `peft` package. You can install it via `pip install peft`."
                )
            if backend != "torch":
                # TODO: Consider following these steps automatically so we can load PEFT models with other backends
                raise ValueError(
                    "PEFT models can currently only be loaded with the `torch` backend. "
                    'To use other backends, load the model with `backend="torch"`, call `model.transformers_model.merge_and_unload()`, '
                    "save that model with `model.save_pretrained()` and then load the model with the desired backend."
                )
            from peft import PeftConfig

            return PeftConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), True

        return AutoConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), False

    def _load_model(
        self,
        model_name_or_path: str,
        transformer_task: Literal["feature-extraction", "sequence-classification", "text-generation", "fill-mask"],
        config: PeftConfig | PretrainedConfig,
        cache_dir: str,
        backend: str,
        is_peft_model: bool,
        **model_args,
    ) -> PreTrainedModel:
        """Loads the transformers or PEFT model into the `auto_model` attribute

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            config ("PeftConfig" | PretrainedConfig): The model configuration.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            is_peft_model (bool): Whether the model is a PEFT model.
            model_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers model.
        """
        if backend == "torch":
            # When loading a PEFT model, we need to load the base model first,
            # but some model_args are only for the adapter
            if is_peft_model:
                for adapter_only_kwarg in ["revision"]:
                    model_args.pop(adapter_only_kwarg, None)

            if transformer_task == "feature-extraction":
                if isinstance(config, T5Config):
                    # Loads the encoder model from T5
                    from transformers import T5EncoderModel

                    with set_temporary_class_attrs(T5EncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return T5EncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                elif isinstance(config, MT5Config):
                    # Loads the encoder model from mT5
                    from transformers import MT5EncoderModel

                    with set_temporary_class_attrs(MT5EncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return MT5EncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                elif isinstance(config, UMT5Config):
                    # Loads the encoder model from UMT5
                    from transformers import UMT5EncoderModel

                    with set_temporary_class_attrs(UMT5EncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return UMT5EncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                elif isinstance(config, UdopConfig):
                    from transformers import UdopEncoderModel

                    with set_temporary_class_attrs(UdopEncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return UdopEncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                elif isinstance(config, T5GemmaConfig):
                    # Loads the encoder model from T5Gemma
                    from transformers import T5GemmaEncoderModel

                    config.is_encoder_decoder = False
                    with set_temporary_class_attrs(
                        T5GemmaEncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]
                    ):
                        return T5GemmaEncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                elif isinstance(config, T5Gemma2Config):
                    # Loads the encoder part from T5Gemma2
                    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder

                    with set_temporary_class_attrs(
                        T5Gemma2Encoder,
                        base_model_prefix="model.encoder",
                        _keys_to_ignore_on_load_unexpected=["decoder.*"],
                    ):
                        return T5Gemma2Encoder.from_pretrained(
                            model_name_or_path, config=config.encoder, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, T5Gemma2TextConfig):
                    # This class is not currently registered in AutoModel
                    from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2Encoder

                    return T5Gemma2Encoder.from_pretrained(
                        model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                    )

                elif isinstance(config, BlenderbotConfig):
                    from transformers.models.blenderbot.modeling_blenderbot import BlenderbotEncoder

                    with set_temporary_class_attrs(
                        BlenderbotEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]
                    ):
                        return BlenderbotEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, BlenderbotSmallConfig):
                    from transformers.models.blenderbot_small.modeling_blenderbot_small import BlenderbotSmallEncoder

                    with set_temporary_class_attrs(
                        BlenderbotSmallEncoder,
                        _keys_to_ignore_on_load_unexpected=["decoder.*"],
                    ):
                        return BlenderbotSmallEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, LongT5Config):
                    from transformers import LongT5EncoderModel

                    with set_temporary_class_attrs(
                        LongT5EncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]
                    ):
                        return LongT5EncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, M2M100Config):
                    from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder

                    with set_temporary_class_attrs(M2M100Encoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return M2M100Encoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, PegasusConfig):
                    from transformers.models.pegasus.modeling_pegasus import PegasusEncoder

                    with set_temporary_class_attrs(PegasusEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return PegasusEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, PegasusXConfig):
                    from transformers.models.pegasus_x.modeling_pegasus_x import PegasusXEncoder

                    with set_temporary_class_attrs(PegasusXEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return PegasusXEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, ProphetNetConfig):
                    from transformers import ProphetNetEncoder

                    with set_temporary_class_attrs(
                        ProphetNetEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]
                    ):
                        return ProphetNetEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, SwitchTransformersConfig):
                    from transformers import SwitchTransformersEncoderModel

                    with set_temporary_class_attrs(
                        SwitchTransformersEncoderModel, _keys_to_ignore_on_load_unexpected=["decoder.*"]
                    ):
                        return SwitchTransformersEncoderModel.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, MoonshineConfig):
                    from transformers.models.moonshine.modeling_moonshine import MoonshineEncoder

                    with set_temporary_class_attrs(MoonshineEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return MoonshineEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

                elif isinstance(config, WhisperConfig):
                    from transformers.models.whisper.modeling_whisper import WhisperEncoder

                    # TODO: Check if base_model_prefix should also be overridden for other architectures
                    # TODO: Should I then also only use the FeatureExtractionMixin instead of the Processor?
                    # TODO: How to determine when to load only the encoder vs the encoder-decoder? Perhaps if
                    # auto_model.forward(...) fails if there's no decoder_input_ids provided.
                    with set_temporary_class_attrs(WhisperEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        model = WhisperEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )
                        # Transformers v5 uses ("text", "audio",) even for the encoder
                        model.input_modalities = "audio"
                        return model

                elif isinstance(config, MarianConfig):
                    from transformers.models.marian.modeling_marian import MarianEncoder

                    # TODO: Check if base_model_prefix should also be overridden for other architectures
                    with set_temporary_class_attrs(MarianEncoder, _keys_to_ignore_on_load_unexpected=["decoder.*"]):
                        return MarianEncoder.from_pretrained(
                            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                        )

            # TODO: What if transformer_task is something else?
            model_cls = TRANSFORMER_TASK_TO_AUTO_MODEL.get(transformer_task, AutoModel)
            return model_cls.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir, **model_args)
        elif backend == "onnx":
            return load_onnx_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_args,
            )
        elif backend == "openvino":
            return load_openvino_model(
                model_name_or_path=model_name_or_path,
                config=config,
                task_name=transformer_task,
                **model_args,
            )
        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")

    def _get_method_output_fields(self, method: Callable) -> list[str]:
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

        return_annotation = get_type_hints(method).get("return", None)
        output_class = find_model_output_class(return_annotation)
        if output_class is None:
            # raise ValueError("Could not determine ModelOutput subclass from method return annotation.")
            return None
        return [field.name for field in fields(output_class)]

    def infer_modalities_edge_cases(self, model: PreTrainedModel, processor) -> tuple[ModalityConfig, str] | None:
        # TODO: What if there's a model from here that also has a chat_template?
        # TODO: What if someone has a Blip2VisionModel or something? It would also match model.config.model_type,
        # but model.get_..._features won't exist.
        # Update: Blip2VisionModel seems to have a different config type
        def infer_method_output_name(method_output_name, method) -> str:
            # Primarily transformers v4 compatibility: v5 often allows for "pooler_output" outputs from get_..._features
            # methods, but v4 didn't use BaseModelOutputWithPooling for these methods yet
            output_fields = self._get_method_output_fields(method) or []
            if method_output_name in output_fields:
                return method_output_name
            return None

        match model.config.model_type:
            case "blip":
                # Custom case as it supports text+image with get_multimodal_features
                return {
                    "text": {
                        "method": "get_text_features",
                        "method_output_name": infer_method_output_name("pooler_output", model.get_text_features),
                    },
                    "image": {
                        "method": "get_image_features",
                        "method_output_name": infer_method_output_name("pooler_output", model.get_image_features),
                    },
                    ("text", "image"): {
                        "method": "get_multimodal_features",
                        "method_output_name": infer_method_output_name("pooler_output", model.get_multimodal_features),
                    },
                }, "sentence_embedding"
            case "blip-2":
                # Custom case because blip-2 doesn't expose pooler_output, but we can still get non-pooled embeddings
                # from the modality-specific methods, even if the forward method is used for multimodal input
                return {
                    "text": {
                        "method": "get_text_features",
                        "method_output_name": infer_method_output_name("last_hidden_state", model.get_text_features),
                    },
                    "image": {
                        "method": "get_image_features",
                        "method_output_name": infer_method_output_name("last_hidden_state", model.get_image_features),
                    },
                }, "token_embeddings"
            case "git" | "visual_bert":
                # Custom case because text+image is supported without the messages format
                # TODO: Should this be an automatic case?
                return {
                    "text": {"method": "forward", "method_output_name": "last_hidden_state"},
                    # "image": {"method": "forward", "method_output_name": "last_hidden_state"},  # TODO: I think git always requires text?
                    ("text", "image"): {"method": "forward", "method_output_name": "last_hidden_state"},
                }, "token_embeddings"
            case "kosmos-2" | "grounding-dino" | "paligemma" | "vilt":
                # Custom case because text+image is supported without the messages format, and text nor image aren't supported
                return {
                    ("text", "image"): {"method": "forward", "method_output_name": "last_hidden_state"},
                }, "token_embeddings"
            case "layoutlmv3":
                # Custom case because text+image is supported without the messages format, and image only is also supported
                return {
                    "image": {"method": "forward", "method_output_name": "last_hidden_state"},
                    ("text", "image"): {"method": "forward", "method_output_name": "last_hidden_state"},
                }, "token_embeddings"
            case "idefics":
                return {
                    "text": {"method": "forward", "method_output_name": "last_hidden_state"},
                    "image": {"method": "forward", "method_output_name": "last_hidden_state"},
                    ("text", "image"): {"method": "forward", "method_output_name": "last_hidden_state"},
                }, "token_embeddings"
            case (
                "hubert"
                | "moonshine"
                | "sew"
                | "sew-d"
                | "unispeech-sat"
                | "unispeech"
                | "wav2vec2"
                | "wav2vec2-conformer"
                | "wavlm"
                | "whisper"
            ):
                # "whisper" is maybe only Audio? The decoder is only for the decoder
                return {
                    "audio": {"method": "forward", "method_output_name": "last_hidden_state"},
                    ("text", "audio"): {"method": "forward", "method_output_name": "last_hidden_state"},
                }, "token_embeddings"
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
        if (result := self.infer_modalities_edge_cases(model, processor)) is not None:
            return result

        modalities = self.infer_modalities_from_processor(model, processor)
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            modalities.append("message")

        # if self.transformer_task == "feature-extraction":
        target_method_output_name = TRANSFORMER_TASK_TO_METHOD_OUTPUT_NAME[self.transformer_task]
        target_module_output_name = TRANSFORMER_TASK_TO_DEFAULT_MODULE_OUTPUT_NAME[self.transformer_task]

        # Let's inspect the forward to see if it can be used for all modalities, or if we need modality-specific methods
        # If we can't inspect the method return type, we assume it has a 'last_hidden_state'.
        output_fields = self._get_method_output_fields(model.forward)
        # if output_fields is None or target_method_output_name in output_fields:
        if target_method_output_name in output_fields:
            return {
                modality: {"method": "forward", "method_output_name": target_method_output_name}
                for modality in modalities
            }, target_module_output_name

        # For feature-extraction, if there's no 'last_hidden_state', we can check for modality-specific methods like get_..._features
        if self.transformer_task == "feature-extraction":
            modality_config: ModalityConfig = {}
            for modality in modalities:
                if modality == "message":
                    continue

                method_name = f"get_{modality}_features"
                if hasattr(model, method_name):
                    method = getattr(model, method_name)
                    output_fields = self._get_method_output_fields(method)
                    if "pooler_output" in output_fields:
                        modality_config[modality] = {"method": method_name, "method_output_name": "pooler_output"}
                    else:
                        modality_config[modality] = {"method": method_name, "method_output_name": None}

            return modality_config, "sentence_embedding"

        return {
            modality: {"method": "forward", "method_output_name": target_method_output_name} for modality in modalities
        }, target_module_output_name

    def infer_modalities_from_processor(self, model: PreTrainedModel, processor) -> list[Modality]:
        # Transformers v5+
        """
        if hasattr(model, "input_modalities"):
            modalities = model.input_modalities
            if isinstance(modalities, str):
                modalities = [modalities]
            elif isinstance(modalities, tuple):
                modalities = list(modalities)
            return modalities
        """

        # Transformers v4:
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

        # This should not be reached
        return []

    def _get_processor_attributes(self) -> dict[str, Any] | None:
        """Get the attributes of the processor if available. Will be removed in the future as transformers v5
        becomes the minimum requirement.

        Returns:
            dict[str, Any] | None: The attributes of the processor, or None if not available.
        """
        if hasattr(self.processor, "get_attributes"):  # Transformers v5+
            return self.processor.get_attributes()
        elif hasattr(self.processor, "attributes"):  # Transformers v4
            return self.processor.attributes
        return None

    def __repr__(self) -> str:
        return f"Transformer({dict(self.get_config_dict(), architecture=self.model.__class__.__name__)})"

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Forward pass through the transformer model.

        This method processes the input features through the underlying transformers model
        and returns the token embeddings along with any other relevant outputs.

        Notes:
            - Only passes arguments that are expected by the underlying transformer model

        Args:
            features (dict[str, torch.Tensor]): Input features dictionary containing at least
                'input_ids' and 'attention_mask'. May also contain other tensors required by
                the underlying transformer model.
            **kwargs: Additional keyword arguments to pass to the underlying transformer model.

        Returns:
            dict[str, torch.Tensor]: Updated features dictionary containing the input features, plus:
                - 'token_embeddings': Token-level embeddings from the transformer model
                - 'attention_mask': Possibly modified attention mask if using PeftModel with prompt learning
                - 'all_layer_embeddings': If the model outputs hidden states, contains embeddings from all layers
        """

        # TODO: Should we pass along the modality in 'features'?
        modality_name: Modality = features.get("modality", "text")
        modality_params = self.modality_config[modality_name]
        # TODO: Allow 'method' to be a tuple of methods to execute sequentially? A bit messy with the kwargs though
        method_name = modality_params["method"]
        method_output_name = modality_params["method_output_name"]
        if isinstance(method_output_name, str):
            method_output_name = (method_output_name,)

        # TODO: Does this prioritize features or kwargs?
        all_kwargs = {**features, **kwargs, "return_dict": True}
        model_method = getattr(self.model, method_name, None)
        if model_method is None:
            raise ValueError(f"Model does not have the requested '{method_name}' method")

        if method_name == "forward":
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in self.model_forward_params}
        else:
            signature = inspect.signature(model_method)
            filtered_kwargs = {key: value for key, value in all_kwargs.items() if key in signature.parameters}

        # TODO: I (re)moved return_dict=True, and I changed up **kwargs
        model_output = model_method(**filtered_kwargs)

        if method_output_name is None:
            embedding = model_output
        else:
            embedding = model_output
            for output_key in method_output_name:
                try:
                    embedding = embedding[output_key]
                except KeyError:
                    # It's possible that the requested key is not accessible via dictionary-style indexing,
                    # but only via attribute access (e.g. chinese_clip) had this issue
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

        # TODO: Check if this is still viable
        if (
            hasattr(self.model.config, "output_hidden_states")
            and self.model.config.output_hidden_states
            and "hidden_states" in model_output
        ):
            features["all_layer_embeddings"] = model_output["hidden_states"]

        return features

    def get_word_embedding_dimension(self) -> int:
        """Get the output embedding dimension from the transformer model.

        Returns:
            int: The hidden dimension size of the model's embeddings.

        Raises:
            ValueError: If the embedding dimension cannot be determined from the model config.
        """
        # Edge case for timm models
        if isinstance(self.model.config, TimmWrapperConfig):
            return self.model.config.num_features

        # Get text config, e.g. for multi-modal models
        """
        try:
            text_config = self.model.config.get_text_config()
        except AttributeError:
            text_config = self.model.config

        if hasattr(text_config, "hidden_size"):
            return text_config.hidden_size

        # Try hidden_sizes list (e.g., ResNet, some vision models)
        if hasattr(text_config, "hidden_sizes"):
            if isinstance(text_config.hidden_sizes, list):
                return text_config.hidden_sizes[-1]  # Use final layer dimension
            return text_config.hidden_sizes
        """

        def get_hidden_size_from_config(config):
            # If we're directly outputting sentence embeddings from the transformer (e.g., using the pooler output),
            # then we should check for projection_dim first, as that's likely the dimension of the sentence embeddings
            # after a projection layer
            if hasattr(config, "projection_dim") and self.module_output_name == "sentence_embedding":
                return config.projection_dim

            if hasattr(config, "hidden_size"):
                return config.hidden_size
            if hasattr(config, "neck_hidden_sizes"):
                if isinstance(config.neck_hidden_sizes, list):
                    return config.neck_hidden_sizes[-1]
                return config.neck_hidden_sizes  # TODO: Unsure if it's ever not a list
            if hasattr(config, "hidden_sizes"):
                if isinstance(config.hidden_sizes, list):
                    return config.hidden_sizes[-1]
                return config.hidden_sizes  # TODO: Unsure if it's ever not a list
            if hasattr(config, "hidden_dim"):
                return config.hidden_dim
            if hasattr(config, "embed_dims"):
                if isinstance(config.embed_dims, list):
                    return config.embed_dims[-1]
                return config.embed_dims  # TODO: Unsure if it's ever not a list
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
            f"Could not determine embedding dimension from model config. Config type: {type(self.model.config).__name__}. "
        )

    def preprocess(
        self,
        inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs]
        | StrInputs
        | PairStrInputs
        | DictInputs
        | ImageInputs
        | ArrayInputs,  # TODO: Rename to inputs?
        prompt: str | None = None,
        modality: str | tuple[str, ...] | None = None,
        padding: str | bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor | Any]:
        """Preprocesses inputs and maps tokens to token-ids.

        Args:
            texts: List of inputs which can be:
                - str: Text inputs
                - dict: Dictionary with modality keys (text, image, audio, video) or chat messages
                - PIL.Image.Image: Image inputs
                - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs

                If a single input is provided, it must be wrapped in a list.
            prompt: Optional system prompt to include in the input
            modality: Optional modality to use. If not provided, will be inferred from inputs.
            padding: Padding strategy for preprocessing

        Returns:
            Dictionary containing preprocessed inputs with 'modality' key indicating the input type
        """
        # Configuration for different modality types
        common_kwargs = {"return_tensors": "pt"}
        # TODO: We should likely set as little defaults as possible here, and also allow users to pass extra kwargs.
        modality_kwargs = {
            "text": {"padding": padding, "truncation": "longest_first"},
            "audio": {"padding": padding},
            "image": {},
            "video": {},
        }
        # Apply architecture-specific defaults
        # TODO: Do we want to have architecture-specific defaults here?
        if self.config.model_type == "whisper":
            # Whisper requires inputs to be exactly 30 seconds long, while its WhisperFeatureExtractor defaults to
            # padding=True (a.k.a. "longest"), instead of defaulting to the required "max_length".
            modality_kwargs["audio"]["padding"] = "max_length"

        prompt_length = None

        # Parse inputs using the input formatter which handles modality detection and message conversion
        # TODO: Having to pass the supports_message might mean that it's cleaner to pull that post-processing to here
        modality, processor_inputs, extra_modality_kwargs = self.input_formatter.parse_inputs(inputs)

        for modality_key, extra_kwargs in extra_modality_kwargs.items():
            modality_kwargs[modality_key].update(extra_kwargs)

        # Always convert to the message format if it's supported, since it's most flexible with e.g. defaults
        if "message" in self.modality_config and modality != "message":
            modality, processor_inputs = self.input_formatter.convert_to_message(modality, processor_inputs)
        elif modality not in self.modality_config:
            raise ValueError(
                f"Modality '{modality}' is not supported by this model. "
                f"Supported modalities: {sorted(self.modality_config.keys())}"
            )

        # Incorporate prompt into inputs if applicable
        if prompt:
            if modality == "message":
                processor_inputs["message"] = self.input_formatter.prepend_prompt_to_messages(
                    processor_inputs["message"], prompt
                )
                # No need to track prompt length for chat messages
            elif modality == "text":
                processor_inputs["text"] = self.input_formatter.prepend_prompt_to_texts(
                    processor_inputs["text"], prompt
                )
                prompt_length = self._get_prompt_length(prompt, **kwargs)

        processor_output = self._call_processor(modality, processor_inputs, modality_kwargs, common_kwargs)
        processor_output["modality"] = modality
        if prompt_length is not None:
            processor_output["prompt_length"] = prompt_length

        return processor_output

    def _get_prompt_length(self, prompt: str, **kwargs) -> int:
        """Return the length of the prompt in tokens, including the BOS token."""
        if (prompt, *kwargs.values()) in self._prompt_length_mapping:
            return self._prompt_length_mapping[(prompt, *kwargs.values())]

        tokenized_prompt = self.preprocess([prompt], modality="text", **kwargs)
        if "input_ids" not in tokenized_prompt:
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a special EOS token, we do not count it as part of the prompt length
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if hasattr(self.tokenizer, "all_special_ids") and last_token in self.tokenizer.all_special_ids:
            prompt_length -= 1
        self._prompt_length_mapping[(prompt, *kwargs.values())] = prompt_length
        return prompt_length

    def _process_chat_messages(
        self, messages: list[DictInputs], text_kwargs: dict[str, Any], common_kwargs: dict[str, Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Process chat messages using the processor's chat template."""
        processor_output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            **text_kwargs,
            **common_kwargs,
            # add_generation_prompt=True,  # Needed for Qwen3-VL-Embedding, but I can't hardcode this
        )

        if "message" not in self.modality_config:
            # TODO: This should be sooner right?
            raise ValueError(
                f"The model does not support 'message' modality, but the input looks like a chat message. "
                f"Supported modalities: {list(self.modality_config.keys())}"
            )

        return processor_output

    def _call_processor(
        self,
        modality: Modality,
        processor_inputs: dict[str, list],
        modality_kwargs: dict[str, dict],
        common_kwargs: dict[str, Any],
    ) -> dict[str, torch.Tensor | Any]:
        """Call the appropriate processor with the correct arguments.

        Args:
            modality: The modality or tuple of modalities being processed
            processor_inputs: Dictionary of processor argument names to lists of values
            modality_kwargs: Configuration kwargs for each modality type
            common_kwargs: Common kwargs to pass to all processor calls

        Returns:
            Processor output dictionary
        """
        # Handle chat/message format
        if modality == "message":
            return self._process_chat_messages(processor_inputs["message"], modality_kwargs["text"], common_kwargs)

        if isinstance(self.processor, ProcessorMixin):
            # Multi-modal processor: pass modality-specific kwargs
            # NOTE: "common_kwargs" is usually sometimes supported via **kwargs, but then we can't check with type inspect easily
            # so we might be forced to use try-except
            try:
                # This is the much cleaner transformers v5 approach
                return self.processor(
                    **processor_inputs,
                    text_kwargs=modality_kwargs["text"],
                    image_kwargs=modality_kwargs["image"],
                    audio_kwargs=modality_kwargs["audio"],
                    video_kwargs=modality_kwargs["video"],
                    common_kwargs=common_kwargs,  # TODO: Ideally we would pass common_kwargs, but for BC we'll stick with unpacking
                    **common_kwargs,
                )
            except Exception:
                # TODO: Ensure that modality is right, especially if we're allowing non-message multi-modal
                kwargs = {}
                if isinstance(modality, str):
                    kwargs.update(modality_kwargs[modality])
                else:
                    for mod in modality:
                        kwargs.update(modality_kwargs[mod])
                return self.processor(**processor_inputs, **kwargs, **common_kwargs)

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

            # Combine type-specific and common kwargs
            call_kwargs = {**type_kwargs, **common_kwargs}

            # If the modality type is in the inputs, extract it as primary argument
            if modality_type in processor_inputs:
                primary_input = processor_inputs.pop(modality_type)
                return self.processor(primary_input, **processor_inputs, **call_kwargs)
            else:
                return self.processor(**processor_inputs, **call_kwargs)

        raise RuntimeError(
            f"Could not determine how to call processor of type {type(self.processor).__name__} "
            f"for modality '{modality}'"
        )

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
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
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> Self:
        init_kwargs = cls._load_init_kwargs(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
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
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: str = "torch",
        **kwargs,
    ) -> dict[str, Any]:
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
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }

        # 3rd priority: config file
        if "model_args" not in config:
            config["model_args"] = {}
        if "tokenizer_args" not in config:
            config["tokenizer_args"] = {}
        if "config_args" not in config:
            config["config_args"] = {}

        # 2nd priority: hub_kwargs
        config["model_args"].update(hub_kwargs)
        config["tokenizer_args"].update(hub_kwargs)
        config["config_args"].update(hub_kwargs)

        # 1st priority: kwargs passed to SentenceTransformer
        if model_kwargs:
            config["model_args"].update(model_kwargs)
        if tokenizer_kwargs:
            config["tokenizer_args"].update(tokenizer_kwargs)
        if config_kwargs:
            config["config_args"].update(config_kwargs)

        return {**config, "cache_dir": cache_folder, "backend": backend}

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
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "tokenizer_args" in config and "trust_remote_code" in config["tokenizer_args"]:
            config["tokenizer_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")

        if "modality_config" in config:
            # Deserialize modality_config keys if they were serialized as comma-separated strings
            deserialized_modality_config = {}
            for modality_key, params in config["modality_config"].items():
                if "," in modality_key:
                    modality_tuple = tuple(modality_key.split(","))
                    deserialized_modality_config[modality_tuple] = params
                else:
                    deserialized_modality_config[modality_key] = params
            config["modality_config"] = deserialized_modality_config

        return config

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()

        def serialize_tuple_keys(key):
            if isinstance(key, tuple):
                return ",".join(key)
            return key

        config_dict["modality_config"] = {
            serialize_tuple_keys(modality): params for modality, params in self.modality_config.items()
        }
        return config_dict
