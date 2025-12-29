import torch
import torch.nn as nn


from transformers import  T5Gemma2Config, T5Gemma2EncoderConfig, T5Gemma2PreTrainedModel
from transformers.utils import can_return_tuple
from transformers.utils.generic import check_model_inputs
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.masking_utils import create_bidirectional_mask

from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2SelfAttention, T5Gemma2EncoderLayer, T5Gemma2TextScaledWordEmbedding, T5Gemma2RMSNorm, T5Gemma2RotaryEmbedding, sliding_window_mask_function

class T5Gemma2TextEncoderModel(T5Gemma2PreTrainedModel):

    def __init__(self, config: T5Gemma2Config):
        super().__init__(config)

        # setup encoder
        self.encoder = T5Gemma2TextEncoder(config.encoder, config.eoi_token_index)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.encoder.set_input_embeddings(new_embeddings)

    @can_return_tuple
    def forward(
        self,
        # encoder inputs
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Seq2SeqModelOutput:
        r"""
        decoder_position_ids (`torch.LongTensor` of shape `(batch_size, decoder_sequence_length)`, *optional*):
            Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0,
            config.decoder.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        """
        # encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return Seq2SeqModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            past_key_values=encoder_outputs.past_key_values,
        )



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
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # As we want to pass `past_key_values=None` explicitly everywhere, we need to pop them from kwargs if present
        kwargs.pop("past_key_values", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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