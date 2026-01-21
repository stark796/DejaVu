#!/usr/bin/env python
"""
Debug RoPE configuration: Maybe HF uses different RoPE parameters.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Comparing RoPE Configuration")
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
    
    hf_layer0 = hf_model.model.layers[0]
    
    # Compare RoPE configurations
    print("\n[2] RoPE Configuration Comparison...")
    
    dv_rope = dv_block0.self_attn.rotary_emb
    hf_rope = hf_layer0.self_attn.rotary_emb
    
    print(f"\n  DejaVu RoPE type: {type(dv_rope)}")
    print(f"  HF RoPE type: {type(hf_rope)}")
    
    # Check config attributes
    print(f"\n  Config rope_theta: {config.rope_theta}")
    
    # Check the inv_freq if available
    if hasattr(dv_rope, 'inv_freq'):
        print(f"\n  DV inv_freq shape: {dv_rope.inv_freq.shape if dv_rope.inv_freq is not None else 'None'}")
    else:
        print(f"\n  DV inv_freq: not available as attribute")
        
    if hasattr(hf_rope, 'inv_freq'):
        print(f"  HF inv_freq shape: {hf_rope.inv_freq.shape if hf_rope.inv_freq is not None else 'None'}")
    else:
        print(f"  HF inv_freq: not available as attribute")
    
    # Generate cos/sin for both
    print("\n[3] Comparing cos/sin outputs...")
    seq_len = 6
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    
    # Create dummy value tensor for rotary_emb call
    dummy_v = torch.randn(1, 8, seq_len, 128, dtype=torch.float16, device="cuda")
    
    dv_cos, dv_sin = dv_rope(dummy_v, position_ids)
    hf_cos, hf_sin = hf_rope(dummy_v, position_ids)
    
    print(f"  DV cos shape: {dv_cos.shape}")
    print(f"  HF cos shape: {hf_cos.shape}")
    
    cos_diff = (dv_cos - hf_cos).abs().max().item()
    sin_diff = (dv_sin - hf_sin).abs().max().item()
    
    print(f"  cos diff: {cos_diff:.6f}")
    print(f"  sin diff: {sin_diff:.6f}")
    
    if cos_diff > 0 or sin_diff > 0:
        print("\n  *** RoPE outputs differ! ***")
        print(f"  DV cos[0,0,:5]: {dv_cos[0,0,:5]}")
        print(f"  HF cos[0,0,:5]: {hf_cos[0,0,:5]}")
    else:
        print("\n  RoPE outputs match!")
    
    # Now check: is HF's layer using a DIFFERENT rotary embedding?
    print("\n[4] Checking if HF model has global rotary embedding...")
    
    # Some HF versions store rotary_emb at model level
    if hasattr(hf_model.model, 'rotary_emb'):
        print("  HF model has global rotary_emb")
        model_rope = hf_model.model.rotary_emb
        model_cos, model_sin = model_rope(dummy_v, position_ids)
        model_cos_diff = (model_cos - hf_cos).abs().max().item()
        print(f"  Global vs layer rotary_emb diff: {model_cos_diff:.6f}")
    else:
        print("  HF model does not have global rotary_emb")
    
    # Check rope_scaling config
    print("\n[5] Checking rope_scaling config...")
    print(f"  Config attributes related to rope:")
    for attr in dir(config):
        if 'rope' in attr.lower():
            print(f"    {attr}: {getattr(config, attr, 'N/A')}")
    
    # Check if there's any rope_type or similar
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        print(f"\n  *** rope_scaling is set: {config.rope_scaling} ***")
    
    # Final test: use HF's rotary embedding in our computation
    print("\n[6] Using HF's rotary embedding in manual computation...")
    
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    with torch.no_grad():
        dv_hidden = dv_emb(input_ids)
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        
        # Q, K, V
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        import math
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        
        dv_q = dv_block0.self_attn.q_proj(dv_normed).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_k = dv_block0.self_attn.k_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dv_v = dv_block0.self_attn.v_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # Use HF's rope instead of DV's
        hf_cos, hf_sin = hf_rope(dv_v, position_ids)
        dv_q_hf, dv_k_hf = apply_rotary_pos_emb(dv_q, dv_k, hf_cos, hf_sin)
        
        # Compare with using DV's rope
        dv_cos, dv_sin = dv_rope(dv_v, position_ids)
        dv_q_dv, dv_k_dv = apply_rotary_pos_emb(dv_q, dv_k, dv_cos, dv_sin)
        
        q_rope_diff = (dv_q_hf - dv_q_dv).abs().max().item()
        print(f"  Q with HF rope vs DV rope: {q_rope_diff:.6f}")
        
        # Complete computation with HF's rope
        dv_k_hf = repeat_kv(dv_k_hf, num_heads // num_kv_heads)
        dv_v_exp = repeat_kv(dv_v, num_heads // num_kv_heads)
        
        from modules.hf_llama_module import _make_causal_mask
        causal_mask = _make_causal_mask((1, seq_len), dv_hidden.dtype, dv_hidden.device)
        
        attn = torch.matmul(dv_q_hf, dv_k_hf.transpose(2, 3)) / math.sqrt(head_dim)
        attn = attn + causal_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        attn_out = torch.matmul(attn, dv_v_exp)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(1, seq_len, -1)
        attn_out = dv_block0.self_attn.o_proj(attn_out)
        
        hidden_after_attn = dv_hidden + attn_out
        mlp_input = dv_block0.post_attention_layernorm(hidden_after_attn)
        mlp_out = dv_block0.mlp(mlp_input)
        final_with_hf_rope = hidden_after_attn + mlp_out
        
        # Compare
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_layer0_out = hf_out.hidden_states[1]
        
        print(f"  Manual (HF rope) vs HF layer: {(final_with_hf_rope - hf_layer0_out).abs().max().item():.6f}")
        
        dv_layer_out, _ = dv_block0(dv_hidden, layer_past=None, mask=None)
        print(f"  DV layer vs HF layer: {(dv_layer_out - hf_layer0_out).abs().max().item():.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
