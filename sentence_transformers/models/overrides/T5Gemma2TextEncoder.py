import torch
import torch.nn as nn

from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import BaseModelOutput
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils.generic import OutputRecorder

from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2Config, T5Gemma2EncoderConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2SelfAttention, T5Gemma2EncoderLayer, T5Gemma2TextScaledWordEmbedding, T5Gemma2RMSNorm, T5Gemma2RotaryEmbedding, sliding_window_mask_function


class T5Gemma2PreTrainedModel(PreTrainedModel):
    config: T5Gemma2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "T5Gemma2EncoderLayer"
    ]
    _skip_keys_device_placement = ["past_key_values"]

    # Mask creation is incompatible
    # FA due to non-default creation / SWA
    _supports_flash_attn = False
    _supports_sdpa = True
    # Flex due to custom masks not compatible to be merged after creation
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": [T5Gemma2EncoderLayer],
        "attentions": [
            OutputRecorder(T5Gemma2SelfAttention, index=1, layer_name="self_attn"),
        ],
    }
    input_modalities = ("text")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        elif isinstance(module, T5Gemma2TextScaledWordEmbedding):
            init.zeros_(module.eoi_embedding)
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif "RMSNorm" in module.__class__.__name__:
            init.zeros_(module.weight)
        elif isinstance(module, T5Gemma2RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)

    def prepare_decoder_input_ids_from_labels(self, input_ids):
        """
        Shifts input_ids to the right, prepends the decoder_start_token_id, and handles
        pad_token_id replacement for labels that were -100.
        This is a common preparation step for decoder inputs in sequence-to-sequence models.
        """
        decoder_config = self.config.decoder
        decoder_start_token_id = decoder_config.bos_token_id
        pad_token_id = decoder_config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError("self.model.config.decoder.bos_token_id has to be defined. ")

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.decoder.pad_token_id has to be defined.")

        # Is this T5 specific?
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Gemma2TextEncoder(T5Gemma2PreTrainedModel):
    config: T5Gemma2EncoderConfig
    _can_record_outputs = {
        "attentions": T5Gemma2SelfAttention,
        "hidden_states": T5Gemma2EncoderLayer,
    }

    def __init__(
        self,
        config: T5Gemma2EncoderConfig,
        eoi_token_index: int = 256_000,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.text_config.vocab_size

        text_config = config.text_config

        self.embed_tokens = T5Gemma2TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            self.padding_idx,
            embed_scale=text_config.hidden_size**0.5,
            eoi_token_index=eoi_token_index,
        )
        self.norm = T5Gemma2RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList(
            [T5Gemma2EncoderLayer(text_config, layer_idx) for layer_idx in range(text_config.num_hidden_layers)]
        )
        self.dropout = nn.Dropout(text_config.dropout_rate)
        self.rotary_emb = T5Gemma2RotaryEmbedding(text_config)

        self.text_config = text_config

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        # Unused for processor compatibility kept in signature.
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        del token_type_ids
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # As we want to pass `past_key_values=None` explicitly everywhere, we need to pop them from kwargs if present
        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None:
            inputs_embeds = self.preprocess_image_features(
                pixel_values, input_ids=input_ids, inputs_embeds=inputs_embeds
            )

        if position_ids is None:
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if not isinstance(self_attn_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            self_attn_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_mask(
                    **mask_kwargs,
                    and_mask_function=sliding_window_mask_function(self.text_config.sliding_window, is_causal=False),
                ),
            }

        # input layer
        hidden_states = inputs_embeds

        # global and local position embeddings
        position_embeddings = {}
        for layer_type in self.text_config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # dropout
        hidden_states = self.dropout(hidden_states)

        for layer_module in self.layers[: self.text_config.num_hidden_layers]:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings[layer_module.attention_type],
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )