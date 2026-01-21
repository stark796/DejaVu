#!/usr/bin/env python
"""
Simplified debug: Compare DejaVu internal operations vs expected.
Tests causal mask and RoPE directly.
"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Testing Causal Mask and Attention")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load DejaVu model
    print("\n[1] Loading DejaVu model...")
    from modules.hf_llama_module import (
        LlamaEmbeddings, LlamaBlock, LlamaLMHead,
        _make_causal_mask, _prepare_decoder_attention_mask
    )
    from transformers import LlamaConfig, AutoTokenizer
    
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    # Prepare input
    print("\n[2] Preparing input...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"  Input: '{text}'")
    print(f"  Tokens: {input_ids.shape}")
    
    with torch.no_grad():
        hidden = dv_emb(input_ids)
        
    print(f"  Hidden shape: {hidden.shape}")
    
    # Test causal mask
    print("\n[3] Testing causal mask...")
    bsz, seq_len, _ = hidden.shape
    
    causal_mask = _make_causal_mask(
        (bsz, seq_len), hidden.dtype, hidden.device, past_key_values_length=0
    )
    print(f"  Causal mask shape: {causal_mask.shape}")
    print(f"  Causal mask (position 0 can see):")
    print(f"    {causal_mask[0, 0, 0, :].tolist()}")
    print(f"  Causal mask (position 5 can see):")
    print(f"    {causal_mask[0, 0, 5, :].tolist()}")
    
    # Expected: position 0 sees only position 0 (rest is -inf)
    # Position 5 sees all positions (all 0)
    pos0_visible = (causal_mask[0, 0, 0, :] == 0).sum().item()
    pos5_visible = (causal_mask[0, 0, 5, :] == 0).sum().item()
    print(f"  Position 0 can see {pos0_visible} positions (expected: 1)")
    print(f"  Position 5 can see {pos5_visible} positions (expected: 6)")
    
    if pos0_visible != 1 or pos5_visible != 6:
        print("\n  *** CAUSAL MASK IS WRONG! ***")
    else:
        print("\n  ✓ Causal mask looks correct")
    
    # Test attention manually
    print("\n[4] Testing attention computation...")
    
    normed = dv_block0.input_layernorm(hidden)
    
    # Get Q, K, V
    q = dv_block0.self_attn.q_proj(normed)
    k = dv_block0.self_attn.k_proj(normed)
    v = dv_block0.self_attn.v_proj(normed)
    
    # Reshape for attention
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)  # [bsz, heads, seq, dim]
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    print(f"  Q shape: {q.shape}")
    print(f"  K shape: {k.shape}")
    print(f"  V shape: {v.shape}")
    
    # RoPE
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
    
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    
    rotary_emb = dv_block0.self_attn.rotary_emb
    cos, sin = rotary_emb(v, position_ids)
    
    print(f"  cos shape: {cos.shape}")
    print(f"  sin shape: {sin.shape}")
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    print(f"  Q after RoPE: range=[{q_rot.min():.4f}, {q_rot.max():.4f}]")
    print(f"  K after RoPE: range=[{k_rot.min():.4f}, {k_rot.max():.4f}]")
    
    # Compute attention weights
    from transformers.models.llama.modeling_llama import repeat_kv
    
    num_kv_groups = num_heads // num_kv_heads
    k_rot = repeat_kv(k_rot, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)
    
    attn_weights = torch.matmul(q_rot, k_rot.transpose(2, 3)) / math.sqrt(head_dim)
    
    print(f"\n  Raw attn weights (before mask): range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    
    # Apply causal mask
    attn_mask = _prepare_decoder_attention_mask(
        None, (bsz, seq_len), hidden, 0
    )
    
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
        print(f"  Attn weights (after mask): range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    
    # Softmax
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden.dtype)
    print(f"  Attn weights (after softmax): range=[{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    
    # Check: first token attention should be 100% on first token
    first_token_attn = attn_weights[0, 0, 0, :]  # head 0, position 0
    print(f"\n  First token attention distribution: {first_token_attn.tolist()}")
    if first_token_attn[0].item() > 0.99:
        print("  ✓ First token correctly attends only to itself")
    else:
        print("  *** First token attention is wrong! ***")
    
    # Run full layer
    print("\n[5] Running full layer...")
    layer_out, _ = dv_block0(hidden, layer_past=None, mask=None)
    print(f"  Layer output: range=[{layer_out.min():.4f}, {layer_out.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
