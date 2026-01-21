#!/usr/bin/env python
"""
Debug RoPE: HF stores rotary_emb differently in newer versions.
Let's find it and compare.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Finding and Comparing RoPE")
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
    
    # Find where HF stores rotary_emb
    print("\n[2] Finding HF rotary embedding location...")
    
    # Check model level
    if hasattr(hf_model.model, 'rotary_emb'):
        print("  Found at: hf_model.model.rotary_emb")
        hf_rope = hf_model.model.rotary_emb
    else:
        print("  Not at model level. Checking layer...")
        
    # Check layer level
    if hasattr(hf_layer0.self_attn, 'rotary_emb'):
        print("  Found at: hf_layer0.self_attn.rotary_emb")
        hf_rope = hf_layer0.self_attn.rotary_emb
    else:
        print("  Not at layer.self_attn level")
    
    # List all attributes of self_attn
    print("\n  HF self_attn attributes:")
    for attr in dir(hf_layer0.self_attn):
        if not attr.startswith('_'):
            obj = getattr(hf_layer0.self_attn, attr, None)
            if hasattr(obj, '__class__') and 'Module' in str(type(obj)):
                print(f"    {attr}: {type(obj)}")
    
    # Check DejaVu
    print("\n[3] DejaVu RoPE location...")
    dv_rope = dv_block0.self_attn.rotary_emb
    print(f"  Found at: dv_block0.self_attn.rotary_emb")
    print(f"  Type: {type(dv_rope)}")
    
    # Compare configs
    print("\n[4] Config comparison...")
    print(f"  DV config rope_theta: {config.rope_theta}")
    print(f"  DV head_dim: {config.hidden_size // config.num_attention_heads}")
    
    # Generate cos/sin with DV rope
    print("\n[5] Testing DV RoPE output...")
    seq_len = 6
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    dummy_v = torch.randn(1, 8, seq_len, 128, dtype=torch.float16, device="cuda")
    
    dv_cos, dv_sin = dv_rope(dummy_v, position_ids)
    print(f"  DV cos shape: {dv_cos.shape}")
    print(f"  DV cos[0,0,:5]: {dv_cos[0,0,:5]}")
    print(f"  DV sin[0,0,:5]: {dv_sin[0,0,:5]}")
    
    # Now directly look at what the HF forward does
    print("\n[6] Tracing HF forward exactly...")
    
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    with torch.no_grad():
        # Run full HF model and capture internal states
        hf_out = hf_model(input_ids, output_hidden_states=True, output_attentions=True)
        
        # Get layer 0 output
        hf_layer0_out = hf_out.hidden_states[1]
        
        # Now run DV
        dv_hidden = dv_emb(input_ids)
        dv_layer0_out, _ = dv_block0(dv_hidden, layer_past=None, mask=None)
        
        layer_diff = (dv_layer0_out - hf_layer0_out).abs().max().item()
        print(f"\n  Layer 0 output diff: {layer_diff:.6f}")
        
        # Let's check if HF attention pattern differs
        if hf_out.attentions is not None:
            hf_attn = hf_out.attentions[0]  # Layer 0 attention
            print(f"  HF attention shape: {hf_attn.shape}")
            print(f"  HF attention[0,0,:,:] (first head):")
            print(hf_attn[0, 0])
            
        # Run DV attention and compare patterns
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        
        dv_q = dv_block0.self_attn.q_proj(dv_normed).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_k = dv_block0.self_attn.k_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dv_v = dv_block0.self_attn.v_proj(dv_normed).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # RoPE
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        dv_cos, dv_sin = dv_rope(dv_v, position_ids)
        dv_q_rot, dv_k_rot = apply_rotary_pos_emb(dv_q, dv_k, dv_cos, dv_sin)
        
        # GQA
        import math
        dv_k_rot = repeat_kv(dv_k_rot, num_heads // num_kv_heads)
        dv_v_exp = repeat_kv(dv_v, num_heads // num_kv_heads)
        
        # Attention weights
        dv_attn_weights = torch.matmul(dv_q_rot, dv_k_rot.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply causal mask
        from modules.hf_llama_module import _make_causal_mask
        causal_mask = _make_causal_mask((1, seq_len), dv_hidden.dtype, dv_hidden.device)
        dv_attn_weights = dv_attn_weights + causal_mask
        
        # Softmax
        dv_attn_probs = torch.nn.functional.softmax(dv_attn_weights, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        
        print(f"\n  DV attention[0,0,:,:] (first head):")
        print(dv_attn_probs[0, 0])
        
        if hf_out.attentions is not None:
            attn_diff = (dv_attn_probs - hf_attn).abs().max().item()
            print(f"\n  Attention pattern diff: {attn_diff:.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
