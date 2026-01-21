#!/usr/bin/env python
"""
Debug RoPE specifically - it's producing NaN.
"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Debugging RoPE NaN Issue")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load DejaVu model
    print("\n[1] Loading DejaVu model...")
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock
    from transformers import LlamaConfig, AutoTokenizer
    
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    # Prepare input
    text = "The cat"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"  Input: '{text}'")
    
    with torch.no_grad():
        hidden = dv_emb(input_ids)
    
    # Get rotary embedding
    print("\n[2] Testing RoPE...")
    
    bsz, seq_len, _ = hidden.shape
    normed = dv_block0.input_layernorm(hidden)
    
    # Q, K projections
    q = dv_block0.self_attn.q_proj(normed)
    k = dv_block0.self_attn.k_proj(normed)
    
    print(f"  Q before reshape: shape={q.shape}, has_nan={torch.isnan(q).any()}, range=[{q.min():.4f}, {q.max():.4f}]")
    print(f"  K before reshape: shape={k.shape}, has_nan={torch.isnan(k).any()}, range=[{k.min():.4f}, {k.max():.4f}]")
    
    # Reshape for attention
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    
    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = dv_block0.self_attn.v_proj(normed)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    print(f"  Q after reshape: shape={q.shape}, has_nan={torch.isnan(q).any()}")
    print(f"  K after reshape: shape={k.shape}, has_nan={torch.isnan(k).any()}")
    
    # Get cos/sin from rotary embedding
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    print(f"  Position IDs: {position_ids}")
    
    rotary_emb = dv_block0.self_attn.rotary_emb
    print(f"  Rotary embedding type: {type(rotary_emb)}")
    
    cos, sin = rotary_emb(v, position_ids)
    
    print(f"\n  cos: shape={cos.shape}, has_nan={torch.isnan(cos).any()}, range=[{cos.min():.4f}, {cos.max():.4f}]")
    print(f"  sin: shape={sin.shape}, has_nan={torch.isnan(sin).any()}, range=[{sin.min():.4f}, {sin.max():.4f}]")
    
    # Try to apply RoPE manually to understand what's happening
    print("\n[3] Applying RoPE manually...")
    
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    
    try:
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        print(f"  Q after RoPE: has_nan={torch.isnan(q_rot).any()}, range=[{q_rot.min():.4f}, {q_rot.max():.4f}]")
        print(f"  K after RoPE: has_nan={torch.isnan(k_rot).any()}, range=[{k_rot.min():.4f}, {k_rot.max():.4f}]")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check what transformers version expects
    print("\n[4] Checking transformers apply_rotary_pos_emb signature...")
    import inspect
    sig = inspect.signature(apply_rotary_pos_emb)
    print(f"  Signature: {sig}")
    
    # Try different argument orders
    print("\n[5] Trying different argument orderings...")
    
    # Maybe the new API has different arguments
    try:
        # Some versions pass position_ids separately
        import transformers
        print(f"  Transformers version: {transformers.__version__}")
    except:
        pass
    
    # Let's implement RoPE manually to test
    print("\n[6] Manual RoPE implementation...")
    
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def manual_apply_rotary(q, k, cos, sin):
        # cos/sin shape: [batch, seq, head_dim]
        # q shape: [batch, num_heads, seq, head_dim]
        
        # Need to broadcast cos/sin to match q shape
        cos = cos.unsqueeze(1)  # [batch, 1, seq, head_dim]
        sin = sin.unsqueeze(1)  # [batch, 1, seq, head_dim]
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    
    q_rot_manual, k_rot_manual = manual_apply_rotary(q, k, cos, sin)
    print(f"  Q after manual RoPE: has_nan={torch.isnan(q_rot_manual).any()}, range=[{q_rot_manual.min():.4f}, {q_rot_manual.max():.4f}]")
    print(f"  K after manual RoPE: has_nan={torch.isnan(k_rot_manual).any()}, range=[{k_rot_manual.min():.4f}, {k_rot_manual.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
