"""
Llama Module for DejaVu - Sparse Inference Version

This module implements sparse inference using trained predictors.
It loads MLP and attention predictors and uses them to skip unused neurons/heads.

KEY FEATURES:
1. LlamaSparseAttention: Uses predictor to select active attention heads
2. LlamaSparseMLP: Uses predictor to select active neurons
3. Lookahead prediction: Predict layer N+1 sparsity while computing layer N

IMPORTANT: This is the inference module that actually achieves the speedup.
The predictors should be trained first using main_mlp_llama.py and main_att_llama.py.
"""

from typing import List, Optional, Tuple, Union
import math
import os
import glob
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


class LlamaSparseAttention(nn.Module):
    """
    Llama Attention with Sparse Head Selection
    
    Uses a trained predictor to select which attention heads to compute.
    Inactive heads are masked out, saving computation.
    """
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
        
        # Sparse prediction
        self.predictor = None
        self.topk = None

    def prepare_head_mask(self, hidden_states: torch.Tensor):
        """Use sparsity predictor to create an attention head mask."""
        if self.predictor is None:
            return None
            
        self.predictor = self.predictor.float()
        bsz, tgt_len, _ = hidden_states.size()

        with torch.no_grad():
            _logit = self.predictor(hidden_states.reshape(-1, self.hidden_size).float())
            
            # Select top-k heads
            k = int(self.num_heads * self.topk)
            _, _top_k_indices = _logit.topk(k, dim=1)
            _top_k_indices = _top_k_indices.reshape(bsz, tgt_len, k)
            _top_k_indices = _top_k_indices.transpose(1, 2)
            
            # Create mask: 1 for active heads, 0 for inactive
            _head_mask = torch.zeros(
                bsz, self.num_heads, tgt_len,
                device=hidden_states.device, dtype=hidden_states.dtype,
            ).scatter_(1, _top_k_indices, 1)
            
        return _head_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        previous_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with sparse attention heads."""
        bsz, q_len, _ = hidden_states.size()

        # Prepare sparse head mask using previous layer's embedding (lookahead)
        head_mask = None
        if self.predictor is not None:
            input_for_prediction = previous_emb if previous_emb is not None else hidden_states
            head_mask = self.prepare_head_mask(input_for_prediction)

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

        # Apply head mask if sparse prediction is enabled
        if head_mask is not None:
            # head_mask shape: [bsz, num_heads, tgt_len]
            # attn_output shape: [bsz, num_heads, tgt_len, head_dim]
            attn_output = attn_output * head_mask.unsqueeze(-1)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_value


class LlamaSparseBlock(nn.Module):
    """
    Llama Decoder Layer with Sparse MLP and Attention
    
    Uses trained predictors to:
    1. Select which attention heads to compute
    2. Select which MLP neurons to compute
    
    Implements lookahead prediction: uses previous layer's output to predict
    current layer's sparsity pattern, allowing async execution.
    """
    def __init__(self, config: LlamaConfig, layer_idx: int, device="cpu"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.self_attn = LlamaSparseAttention(config=config, layer_idx=layer_idx, device=device)
        
        # MLP components
        self.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False, device=device)
        self.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False, device=device)
        self.down_proj = nn.Linear(config.intermediate_size, self.hidden_size, bias=False, device=device)
        self.act_fn = nn.SiLU()
        
        # MLP sparse prediction
        self.predictor = None
        self.topk = None
        self._mask = None

    def prepare_fc_mask(self, hidden_states: torch.Tensor):
        """Use sparsity predictor to create a neuron mask for MLP."""
        if self.predictor is None:
            return None
            
        with torch.no_grad():
            self.predictor = self.predictor.float()
            _logit = self.predictor(hidden_states.reshape(-1, self.hidden_size).float())
            _, _top_indices = _logit.topk(self.topk, dim=1)
            self._mask = torch.zeros_like(_logit)
            self._mask = self._mask.scatter(1, _top_indices, 1).bool().half()
        return self._mask

    @classmethod
    def from_pretrained(cls, model_path, config=None, layer_index=None):
        """Load pretrained layer with sparse predictors."""
        assert layer_index is not None
        if config is None:
            config = LlamaConfig.from_pretrained(model_path)

        module = torch.nn.utils.skip_init(cls, config, layer_index).eval()
        
        # Load base weights
        try:
            state_dict = torch.load(os.path.join(model_path, f"pytorch_{layer_index}.pt"))
            # Map state dict keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('mlp.'):
                    # mlp.gate_proj -> gate_proj
                    new_k = k.replace('mlp.', '')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            module.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"Cannot load layer {layer_index}. Error: {e}")

        module.layer_index = layer_index
        module.self_attn.layer_idx = layer_index
        
        # Load MLP predictor
        predictor_path = os.environ.get("SPARSE_PATH", "./checkpoint/llama-3b-sparse-predictor")
        mlp_topk = int(os.environ.get("MLP_TOPK", "1000"))
        
        module.predictor = nn.Sequential(
            nn.Linear(module.hidden_size, 1000, bias=None),
            nn.Linear(1000, config.intermediate_size, bias=None),
        )
        module.topk = mlp_topk
        
        try:
            mlp_files = glob.glob(f"{predictor_path}/c4_layer{layer_index}_*.pt")
            if mlp_files:
                print(f"Loading MLP sparse predictor from {mlp_files[0]}")
                module.predictor.load_state_dict(torch.load(mlp_files[0]))
            else:
                print(f"No MLP predictor found for layer {layer_index}, using random")
        except Exception as e:
            print(f"Cannot load MLP sparse predictor {layer_index}: {e}")

        # Load Attention predictor
        att_topk = float(os.environ.get("ATT_TOPK", "0.7"))
        
        module.self_attn.predictor = nn.Sequential(
            nn.Linear(module.hidden_size, 1000, bias=None),
            nn.Linear(1000, config.num_attention_heads, bias=None),
        )
        module.self_attn.topk = att_topk
        
        try:
            att_files = glob.glob(f"{predictor_path}/c4_att_k*_layer{layer_index}_*.pt")
            if att_files:
                print(f"Loading attention sparse predictor from {att_files[0]}")
                module.self_attn.predictor.load_state_dict(torch.load(att_files[0]))
            else:
                print(f"No attention predictor found for layer {layer_index}, using random")
        except Exception as e:
            print(f"Cannot load attention sparse predictor {layer_index}: {e}")

        # CRITICAL FIX: Reinitialize rotary embedding since skip_init doesn't initialize its buffers
        # The LlamaRotaryEmbedding computes inv_freq in __init__, which is skipped by skip_init
        module.self_attn.rotary_emb = LlamaRotaryEmbedding(config=config)

        return module

    def forward(
        self,
        x: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        previous_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass with sparse computation."""
        
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

        # Attention with sparse heads
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=layer_past,
            use_cache=True,
            previous_emb=previous_emb,  # For lookahead prediction
        )
        hidden_states = residual + hidden_states

        # MLP with sparse neurons
        residual = hidden_states
        
        # Prepare MLP sparse mask using previous embedding (lookahead)
        mlp_input = previous_emb if previous_emb is not None else hidden_states
        self.prepare_fc_mask(mlp_input)
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Compute gated MLP
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        activation = self.act_fn(gate) * up
        
        # Apply sparse mask if available
        if self._mask is not None:
            # Reshape mask for broadcasting
            _mask = self._mask.view(-1, self.intermediate_size)
            activation_flat = activation.view(-1, self.intermediate_size)
            activation_flat = activation_flat * _mask
            activation = activation_flat.view(activation.shape)
        
        hidden_states = self.down_proj(activation)
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
