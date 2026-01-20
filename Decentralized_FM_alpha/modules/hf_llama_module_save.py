"""
Llama Module for DejaVu - Data Collection Version

This module is used to collect training data for the sparse predictor.
It saves:
1. mlp_x: Input to MLP block (before RMSNorm) - used as query for predictor
2. mlp_label: Output of gate_proj * up_proj (activation pattern) - used as label
3. att_x: Input to attention block (before RMSNorm) - used as query for attention predictor  
4. att_label: Attention head importance scores - used as label

KEY DIFFERENCE from OPT:
- OPT uses ReLU, so label is simply (fc1_output > 0)
- Llama uses SiLU gating: gate_proj(x) * SiLU(up_proj(x)), so activation pattern is different
- We save the product of gate and up projections to analyze sparsity

IMPORTANT FOR DEJAVU HYPOTHESIS TESTING:
- This collects data to train predictors
- We expect predictors to perform poorly on Llama because embeddings change layer-to-layer
"""

from typing import List, Optional, Tuple, Union
import math
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.llama.configuration_llama import LlamaConfig


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """Make causal mask used for bi-directional self-attention."""
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    """Prepare the attention mask for Llama decoder."""
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, inputs_embeds.dtype, inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask


class LlamaEmbeddings(nn.Module):
    """Llama Embeddings - simpler than OPT since Llama uses RoPE (handled in attention)"""
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx, device=device,
        )

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)
        module = torch.nn.utils.skip_init(cls, config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(model_path, "pytorch_embs.pt")))
        except:
            print("Cannot load embeddings. The model is randomly initialized.")
        return module

    def forward(self, input_ids, past_layer=None, mask=None, **kwargs):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embed_tokens(input_ids)
        return hidden_states


class LlamaAttention(nn.Module):
    """Llama Attention with data collection for sparse predictor training"""
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, device="cpu"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device)
        
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        # Data collection variables (set in from_pretrained)
        self.fp_i = 0
        self.fp_label = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with attention head importance collection"""
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Collect attention head importance: norm of each head's output
        # This is used as label for attention sparsity predictor
        # Only collect during prefill (when q_len > 1), not during token generation
        if self.fp_label is not None and self.fp_i < self.fp_label.shape[0] and q_len > 1:
            attn_output_norm = attn_output.norm(dim=-1)  # [bsz, num_heads, seq_len]
            attn_output_norm = attn_output_norm.transpose(2, 1).reshape(-1, self.num_heads)
            num_tokens = attn_output_norm.size(0)
            
            begin, end = self.fp_i, min(self.fp_i + num_tokens, self.fp_label.shape[0])
            self.fp_label[begin:end] = attn_output_norm[:end-begin].detach().cpu().numpy()
            self.fp_i += num_tokens

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


class LlamaBlock(nn.Module):
    """
    Llama Decoder Layer with Data Collection
    
    Collects:
    1. MLP input (before RMSNorm) as query for predictor
    2. MLP activation pattern (gate * up) as label
    3. Attention input as query for attention predictor
    4. Attention head importance as label
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, device="cpu"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, device=device)
        
        # MLP - we create a simple module to hold the components
        # This matches HF state dict keys: mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False, device=device)
        self.mlp.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False, device=device)
        self.mlp.down_proj = nn.Linear(config.intermediate_size, self.hidden_size, bias=False, device=device)
        self.act_fn = nn.SiLU()
        
        # Data collection variables
        self.fp_i = 0
        self.fp_mlp_query = None
        self.fp_att_query = None
        self.fp_label = None

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None, data_path=None, num_samples=80000):
        """
        Load pretrained layer and setup data collection.
        
        Args:
            model_path: Path to converted Llama checkpoint
            config: LlamaConfig
            layer_index: Index of this layer
            data_path: Path to save collected data (e.g., "/path/to/llama_3b_c4")
            num_samples: Number of samples to collect (default 400000, same as OPT)
        """
        assert layer_index is not None
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)

        module = torch.nn.utils.skip_init(cls, config, layer_index).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(model_path, f"pytorch_{layer_index}.pt")))
        except Exception as e:
            print(f"Cannot load layer {layer_index}. Error: {e}")

        module.layer_index = layer_index
        module.self_attn.layer_index = layer_index
        module.fp_i = 0
        
        # Setup data collection paths
        if data_path is None:
            data_path = os.environ.get("DATA_PATH", "./data/llama_3b_c4")
        
        os.makedirs(data_path, exist_ok=True)
        print(f"[Layer {layer_index}] Setting up data collection in {data_path}")
        
        # MLP query: input to MLP block (before RMSNorm)
        module.fp_mlp_query = np.memmap(
            f"{data_path}/mlp_x_{layer_index}.mmap",
            dtype="float16", mode="w+",
            shape=(num_samples, config.hidden_size),
        )
        # Don't initialize - too large, would cause bus error
        
        # Attention query: input to attention block (before RMSNorm)
        module.fp_att_query = np.memmap(
            f"{data_path}/att_x_{layer_index}.mmap",
            dtype="float16", mode="w+",
            shape=(num_samples, config.hidden_size),
        )
        
        # MLP label: activation pattern from gated MLP
        # Note: Llama uses gate_proj and up_proj, so intermediate_size is the label dimension
        module.fp_label = np.memmap(
            f"{data_path}/mlp_label_{layer_index}.mmap",
            dtype="float16", mode="w+",
            shape=(num_samples, config.intermediate_size),
        )
        
        # Attention label: head importance scores
        module.self_attn.fp_i = 0
        module.self_attn.fp_label = np.memmap(
            f"{data_path}/att_label_{layer_index}.mmap",
            dtype="float16", mode="w+",
            shape=(num_samples, config.num_attention_heads),
        )

        return module

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass with data collection."""
        
        if layer_past is not None:
            past_length = layer_past[0].size(2)
        else:
            past_length = 0
            
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1) + past_length), dtype=torch.bool, device=x.device)
        
        attention_mask = _prepare_decoder_attention_mask(mask, x.shape[:2], x, past_length)
        
        if position_ids is None:
            seq_length = x.shape[1]
            position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0)

        hidden_states = x
        residual = hidden_states

        # Collect attention input query (before RMSNorm)
        # Only collect during prefill (when seq_len > 1), not during token generation
        seq_len = hidden_states.size(1)
        if self.fp_att_query is not None and self.fp_i < self.fp_att_query.shape[0] and seq_len > 1:
            # During prefill: hidden_states is [batch, seq_len, hidden]
            # Flatten and select valid positions based on mask
            _hidden = hidden_states.view(-1, hidden_states.size(-1))
            # Mask may be different shape, so just take all positions for prefill
            num_tokens = _hidden.size(0)
            begin, end = self.fp_i, min(self.fp_i + num_tokens, self.fp_att_query.shape[0])
            self.fp_att_query[begin:end] = _hidden[:end-begin].detach().cpu().numpy()

        # Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=layer_past,
            use_cache=True,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # Collect MLP input query (before RMSNorm)
        # Only collect during prefill (when seq_len > 1)
        residual = hidden_states
        if self.fp_mlp_query is not None and self.fp_i < self.fp_mlp_query.shape[0] and seq_len > 1:
            _hidden = hidden_states.view(-1, hidden_states.size(-1))
            num_tokens = _hidden.size(0)
            begin, end = self.fp_i, min(self.fp_i + num_tokens, self.fp_mlp_query.shape[0])
            self.fp_mlp_query[begin:end] = _hidden[:end-begin].detach().cpu().numpy()

        # MLP with data collection
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Compute MLP components separately for data collection
        # Use self.mlp.* to match HF state dict keys
        gate = self.mlp.gate_proj(hidden_states)
        up = self.mlp.up_proj(hidden_states)
        
        # The activation pattern: SiLU(gate) * up
        # This is what we use as label for the sparse predictor
        activation = self.act_fn(gate) * up
        
        # Collect MLP activation label
        # Only collect during prefill (when seq_len > 1)
        if self.fp_label is not None and self.fp_i < self.fp_label.shape[0] and seq_len > 1:
            # For Llama, we record the magnitude of activation (similar to how OPT records fc1 > 0)
            # But since SiLU doesn't create strict zeros, we record the actual values
            _label = activation.view(-1, activation.size(-1))
            num_tokens = _label.size(0)
            begin, end = self.fp_i, min(self.fp_i + num_tokens, self.fp_label.shape[0])
            self.fp_label[begin:end] = _label[:end-begin].detach().cpu().numpy()
            self.fp_i += num_tokens
            
            # Print progress occasionally
            if self.layer_idx == 0 and self.fp_i % 10000 < num_tokens:
                print(f"[Data Collection] Layer 0: {self.fp_i}/{self.fp_label.shape[0]} samples collected")
        
        hidden_states = self.mlp.down_proj(activation)
        hidden_states = residual + hidden_states

        return hidden_states, present


class LlamaLMHead(nn.Module):
    """Llama Language Model Head"""
    def __init__(self, config: LlamaConfig, device="cpu"):
        super().__init__()
        self.config = config
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)
        module = torch.nn.utils.skip_init(cls, config).eval()
        try:
            module.load_state_dict(torch.load(os.path.join(model_path, "pytorch_lm_head.pt")))
        except:
            print("Cannot load LM head. The model is randomly initialized.")
        return module

    def forward(self, x, input_ids=None):
        x = self.norm(x)
        x = self.lm_head(x)
        return x
