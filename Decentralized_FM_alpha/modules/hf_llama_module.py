"""
Llama Module for DejaVu Sparse Inference

This file adapts the DejaVu framework for Llama 3.2 3B model.

KEY ARCHITECTURAL DIFFERENCES from OPT:
1. Normalization: RMSNorm (not LayerNorm)
2. Positional Encoding: Rotary Position Embeddings (RoPE) instead of learned absolute
3. Activation: SiLU (Swish) instead of ReLU
4. MLP: gate_proj + up_proj → SiLU → down_proj (gated MLP)
5. Attention: Grouped Query Attention (GQA) with num_key_value_heads

IMPORTANT FOR DEJAVU HYPOTHESIS:
- OPT assumes embeddings don't change much layer-to-layer
- Llama's RoPE and architecture cause significant embedding drift
- This file helps prove that DejaVu's predictor fails on Llama

Author: Adapted for Llama 3.2 3B testing
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
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    """Prepare the attention mask for Llama decoder."""
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class LlamaEmbeddings(nn.Module):
    """
    Llama Embeddings - simpler than OPT since Llama uses RoPE (handled in attention)
    
    Key difference from OPT:
    - No learned positional embeddings here
    - RoPE is applied in attention layer
    """
    def __init__(self, config, device="cpu"):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            device=device,
        )

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)
        module = torch.nn.utils.skip_init(cls, config).eval()
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        "pytorch_embs.pt",
                    )
                )
            )
        except:
            print("Cannot load from <model_name>. The model is randomly initialized.")
        return module

    def forward(self, input_ids, past_layer=None, mask=None, **kwargs):
        """
        Forward pass for embeddings.
        
        Unlike OPT, we don't add positional embeddings here.
        RoPE is applied in the attention layer.
        """
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        
        hidden_states = self.embed_tokens(input_ids)
        
        return hidden_states


class LlamaAttention(nn.Module):
    """
    Llama Attention with Rotary Position Embeddings and Grouped Query Attention
    
    Key differences from OPT:
    1. Uses RoPE instead of absolute positional embeddings
    2. Supports Grouped Query Attention (GQA)
    3. No bias in linear layers
    """
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        device="cpu",
    ):
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

        # Llama uses no bias in attention projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, device=device)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device)
        
        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        # For sparse attention prediction (DejaVu)
        self.predictor = None
        self.topk = None

    def _apply(self, fn, recurse=True):
        """
        Override _apply to prevent rotary_emb.inv_freq from being converted to float16.
        
        This is critical because inv_freq values can underflow to zero in float16,
        which corrupts the RoPE cos/sin computation and causes NaN in attention.
        """
        # Apply to all submodules except rotary_emb
        for name, module in self._modules.items():
            if name != 'rotary_emb' and module is not None:
                module._apply(fn)
        
        # Apply fn to parameters and buffers of this module (not submodules)
        for key, param in self._parameters.items():
            if param is not None:
                with torch.no_grad():
                    new_param = fn(param)
                if param.is_leaf:
                    self._parameters[key] = torch.nn.Parameter(new_param, requires_grad=param.requires_grad)
                else:
                    self._parameters[key] = new_param
        
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        
        # For rotary_emb, only move to device but keep dtype as float32
        # by rebuilding it on the correct device
        if hasattr(self, 'rotary_emb') and self.rotary_emb is not None:
            # Get the target device from one of the linear layers
            target_device = self.q_proj.weight.device
            # Recreate rotary_emb fresh on the correct device (keeps float32)
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config).to(target_device)
        
        return self

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Llama attention.
        
        Key difference: applies RoPE to queries and keys.
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get sequence length for past KV
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # DEBUG: Check position_ids
        if self.layer_idx == 0:
            print(f"DEBUG attn: position_ids = {position_ids}")
            print(f"DEBUG attn: query_states has nan = {torch.isnan(query_states).any()}")
        
        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        
        # DEBUG: Check RoPE output
        if self.layer_idx == 0:
            print(f"DEBUG attn: cos has nan = {torch.isnan(cos).any()}, inf = {torch.isinf(cos).any()}")
            print(f"DEBUG attn: sin has nan = {torch.isnan(sin).any()}, inf = {torch.isinf(sin).any()}")
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # DEBUG: Check after RoPE
        if self.layer_idx == 0:
            print(f"DEBUG attn: after RoPE, Q has nan = {torch.isnan(query_states).any()}")
            print(f"DEBUG attn: after RoPE, K has nan = {torch.isnan(key_states).any()}")

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # DEBUG: Check attention weights before mask
        if self.layer_idx == 0:
            print(f"DEBUG attn: attn_weights (before mask) has nan = {torch.isnan(attn_weights).any()}, max = {attn_weights.max():.4f}, min = {attn_weights.min():.4f}")
            if attention_mask is not None:
                print(f"DEBUG attn: attention_mask has nan = {torch.isnan(attention_mask).any()}, max = {attention_mask.max():.4f}, min = {attention_mask.min():.4f}")

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # DEBUG: Check after mask
        if self.layer_idx == 0:
            print(f"DEBUG attn: attn_weights (after mask) has nan = {torch.isnan(attn_weights).any()}")

        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    """
    Llama MLP with Gated Linear Unit
    
    Architecture: gate_proj + up_proj → SiLU(gate) * up → down_proj
    
    Key differences from OPT:
    1. Uses SiLU activation instead of ReLU
    2. Uses gated architecture (gate_proj)
    3. No bias in linear layers
    """
    def __init__(self, config: LlamaConfig, device="cpu"):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Llama MLP has no bias
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, device=device)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """
        Forward pass for Llama MLP.
        
        Note: The gated architecture means sparsity pattern is different from OPT's fc1→fc2.
        In OPT, we check if fc1 output > 0 (ReLU).
        In Llama, we'd need to check if SiLU(gate) * up creates zeros.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    """
    Llama Decoder Layer
    
    Structure:
    - Pre-RMSNorm
    - Self Attention with RoPE
    - Residual connection
    - Pre-RMSNorm
    - Gated MLP
    - Residual connection
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, device="cpu"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # RMSNorm instead of LayerNorm
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Attention
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, device=device)
        
        # MLP - use individual components to match HuggingFace state dict keys
        # HF uses: mlp.gate_proj, mlp.up_proj, mlp.down_proj
        self.mlp = LlamaMLP(config=config, device=device)
        
        # For sparse prediction (DejaVu)
        self.predictor = None
        self.topk = None

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        """Load a pre-trained Llama layer."""
        assert layer_index is not None
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)

        module = torch.nn.utils.skip_init(cls, config, layer_index).eval()
        try:
            state_dict_path = os.path.join(model_path, f"pytorch_{layer_index}.pt")
            state_dict = torch.load(state_dict_path, map_location='cpu')
            
            # Debug: Print keys for first layer
            if layer_index == 0:
                print(f"Loading layer {layer_index} state dict keys: {list(state_dict.keys())[:5]}...")
                print(f"Module expects keys: {list(module.state_dict().keys())[:5]}...")
            
            # Load with strict=False to see what's missing
            missing, unexpected = module.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"WARNING: Layer {layer_index} MISSING keys: {missing}")
            if unexpected:
                print(f"WARNING: Layer {layer_index} UNEXPECTED keys: {unexpected}")
                
        except Exception as e:
            print(f"Cannot load layer {layer_index} from <model_name>. The model is randomly initialized. Error: {e}")

        module.layer_index = layer_index
        return module

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for Llama decoder layer.
        
        Args:
            x: Input hidden states [batch, seq_len, hidden_size]
            layer_past: Past key-value cache
            mask: Attention mask
            position_ids: Position IDs for RoPE
        """
        # Debug: Check for NaN in input
        if self.layer_idx == 0 and torch.isnan(x).any():
            print(f"WARNING: NaN in input to layer {self.layer_idx}")
        
        # Determine past length for attention mask
        if layer_past is not None:
            past_length = layer_past[0].size(2)
        else:
            past_length = 0
            
        # Create attention mask if not provided
        if mask is None:
            mask = torch.ones(
                (x.size(0), x.size(1) + past_length), dtype=torch.bool, device=x.device
            )
        
        # Prepare attention mask
        attention_mask = _prepare_decoder_attention_mask(
            mask, x.shape[:2], x, past_length
        )
        
        # Create position_ids if not provided
        if position_ids is None:
            seq_length = x.shape[1]
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=x.device
            )
            position_ids = position_ids.unsqueeze(0)

        hidden_states = x
        residual = hidden_states

        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Debug: Check after layernorm
        if self.layer_idx == 0 and torch.isnan(hidden_states).any():
            print(f"WARNING: NaN after input_layernorm in layer {self.layer_idx}")
        
        hidden_states, _, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=layer_past,
            use_cache=True,
        )
        
        # Debug: Check after attention
        if self.layer_idx == 0 and torch.isnan(hidden_states).any():
            print(f"WARNING: NaN after attention in layer {self.layer_idx}")
        
        hidden_states = residual + hidden_states

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Debug: Check after MLP
        if self.layer_idx == 0 and torch.isnan(hidden_states).any():
            print(f"WARNING: NaN after MLP in layer {self.layer_idx}")
        
        hidden_states = residual + hidden_states

        return hidden_states, present


class LlamaLMHead(nn.Module):
    """
    Llama Language Model Head
    
    Structure:
    - RMSNorm
    - Linear projection to vocab
    """
    def __init__(self, config: LlamaConfig, device="cpu"):
        super().__init__()
        self.config = config
        
        # Final RMSNorm
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head (no bias)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)
        module = torch.nn.utils.skip_init(cls, config).eval()
        try:
            module.load_state_dict(
                torch.load(
                    os.path.join(
                        model_path,
                        "pytorch_lm_head.pt",
                    )
                )
            )
        except:
            print("Cannot load from <model_name>. The model is randomly initialized.")
        return module

    def forward(self, x, input_ids=None):
        # Debug: check input
        if torch.isnan(x).any():
            print(f"WARNING: NaN in LlamaLMHead input, shape={x.shape}")
        if torch.isinf(x).any():
            print(f"WARNING: Inf in LlamaLMHead input, shape={x.shape}")
        
        x = self.norm(x)
        
        # Debug: check after norm
        if torch.isnan(x).any():
            print(f"WARNING: NaN after LlamaLMHead norm, shape={x.shape}")
        if torch.isinf(x).any():
            print(f"WARNING: Inf after LlamaLMHead norm, shape={x.shape}")
        
        x = self.lm_head(x)
        
        # Debug: check output
        if torch.isnan(x).any():
            print(f"WARNING: NaN in LlamaLMHead output (logits), shape={x.shape}")
        if torch.isinf(x).any():
            print(f"WARNING: Inf in LlamaLMHead output (logits), shape={x.shape}")
        
        return x
