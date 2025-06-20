from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, ALL_ATTENTION_FUNCTIONS, logger
import types

def apply_activation_checkpointing(model):
    """
    use checkpoint in model self-attention
    """

    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(forward_ckpt, layer.self_attn)

    return model


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout,
    **kwargs,
):

    attn_output, attn_weights = _self_attn_ckpt(module, query, key, value, attention_mask, scaling, dropout)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def _self_attn(module, query, key, value, attention_mask, scaling, dropout):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output, attn_weights
    
def _self_attn_ckpt(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
):
    return checkpoint(
        _self_attn,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
    )

def forward_ckpt(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_value,
    cache_position,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights