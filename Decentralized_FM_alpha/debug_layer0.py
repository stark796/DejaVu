#!/usr/bin/env python
"""
CRITICAL FIX: Compare HF's global RoPE vs DV's layer RoPE.
The issue is that HF stores rotary_emb at model level, not layer level.
"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Comparing Global RoPE vs Layer RoPE")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load models
    print("\n[1] Loading models...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    config = LlamaConfig.from_pretrained(model_path)
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    # Get both RoPE instances
    hf_rope = hf_model.model.rotary_emb
    dv_rope = dv_block0.self_attn.rotary_emb
    
    print(f"\n[2] RoPE comparison...")
    print(f"  HF RoPE type: {type(hf_rope)}")
    print(f"  DV RoPE type: {type(dv_rope)}")
    
    # Check inv_freq if available
    if hasattr(hf_rope, 'inv_freq') and hf_rope.inv_freq is not None:
        print(f"\n  HF inv_freq[:5]: {hf_rope.inv_freq[:5]}")
    if hasattr(dv_rope, 'inv_freq') and dv_rope.inv_freq is not None:
        print(f"  DV inv_freq[:5]: {dv_rope.inv_freq[:5]}")
    
    # Generate cos/sin
    seq_len = 6
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    dummy_v = torch.randn(1, 8, seq_len, 128, dtype=torch.float16, device="cuda")
    
    hf_cos, hf_sin = hf_rope(dummy_v, position_ids)
    dv_cos, dv_sin = dv_rope(dummy_v, position_ids)
    
    print(f"\n[3] cos/sin comparison...")
    print(f"  HF cos shape: {hf_cos.shape}")
    print(f"  DV cos shape: {dv_cos.shape}")
    
    cos_diff = (hf_cos - dv_cos).abs().max().item()
    sin_diff = (hf_sin - dv_sin).abs().max().item()
    print(f"\n  cos diff: {cos_diff:.6f}")
    print(f"  sin diff: {sin_diff:.6f}")
    
    if cos_diff > 0:
        print(f"\n  *** RoPE cos/sin VALUES DIFFER! ***")
        print(f"  HF cos[0,0,:5]: {hf_cos[0,0,:5]}")
        print(f"  DV cos[0,0,:5]: {dv_cos[0,0,:5]}")
        print(f"  HF sin[0,0,:5]: {hf_sin[0,0,:5]}")
        print(f"  DV sin[0,0,:5]: {dv_sin[0,0,:5]}")
        
        # Check at position 5 (last position)
        print(f"\n  At position 5:")
        print(f"  HF cos[0,5,:5]: {hf_cos[0,5,:5]}")
        print(f"  DV cos[0,5,:5]: {dv_cos[0,5,:5]}")
    
    # Now test: use HF's RoPE in DV's computation
    print(f"\n[4] Testing DV computation with HF's RoPE...")
    
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    with torch.no_grad():
        dv_hidden = dv_emb(input_ids)
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        
        dv_q = dv_block0.self_attn.q_proj(dv_normed).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_k = dv_block0.self_attn.k_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dv_v = dv_block0.self_attn.v_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        from modules.hf_llama_module import _make_causal_mask
        
        # WITH HF's RoPE
        hf_cos, hf_sin = hf_rope(dv_v, position_ids)
        q_hf, k_hf = apply_rotary_pos_emb(dv_q, dv_k, hf_cos, hf_sin)
        
        k_hf = repeat_kv(k_hf, num_heads // num_kv_heads)
        v_exp = repeat_kv(dv_v, num_heads // num_kv_heads)
        
        attn_hf = torch.matmul(q_hf, k_hf.transpose(2, 3)) / math.sqrt(head_dim)
        causal_mask = _make_causal_mask((1, seq_len), dv_hidden.dtype, dv_hidden.device)
        attn_hf = attn_hf + causal_mask
        attn_hf = torch.nn.functional.softmax(attn_hf, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        
        attn_out_hf = torch.matmul(attn_hf, v_exp)
        attn_out_hf = attn_out_hf.transpose(1, 2).contiguous().reshape(1, seq_len, -1)
        attn_out_hf = dv_block0.self_attn.o_proj(attn_out_hf)
        
        hidden_after_attn = dv_hidden + attn_out_hf
        mlp_input = dv_block0.post_attention_layernorm(hidden_after_attn)
        mlp_out = dv_block0.mlp(mlp_input)
        final_with_hf_rope = hidden_after_attn + mlp_out
        
        # Compare with actual HF output
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_layer0_out = hf_out.hidden_states[1]
        
        diff_with_hf_rope = (final_with_hf_rope - hf_layer0_out).abs().max().item()
        print(f"  DV (with HF's RoPE) vs HF layer: {diff_with_hf_rope:.6f}")
        
        # Also compare with DV's own layer output
        dv_layer_out, _ = dv_block0(dv_hidden, layer_past=None, mask=None)
        diff_dv_layer = (dv_layer_out - hf_layer0_out).abs().max().item()
        print(f"  DV layer (own RoPE) vs HF layer: {diff_dv_layer:.6f}")
        
        if diff_with_hf_rope < diff_dv_layer:
            print(f"\n  *** CONFIRMED: Using HF's RoPE reduces diff! ***")
            print(f"  Improvement: {diff_dv_layer - diff_with_hf_rope:.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
