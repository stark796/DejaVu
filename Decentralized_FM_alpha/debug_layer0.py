#!/usr/bin/env python
"""
Debug script to trace exactly where the forward pass diverges.
Compares attention and MLP outputs separately.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    model_path = "./pretrained_models/llama-3.2-3b"
    
    print("=" * 60)
    print("Detailed Layer 0 Trace")
    print("=" * 60)
    
    # 1. Load HuggingFace model
    print("\n[1] Loading HuggingFace model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    # 2. Load DejaVu model
    print("\n[2] Loading DejaVu model...")
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    from transformers import LlamaConfig
    
    config = LlamaConfig.from_pretrained(model_path)
    
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    # 3. Prepare input
    print("\n[3] Preparing input...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"  Input: '{text}'")
    print(f"  Token IDs: {input_ids}")
    
    # 4. Get embeddings
    print("\n[4] Getting embeddings...")
    with torch.no_grad():
        dv_hidden = dv_emb(input_ids)
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        
    print(f"  DV emb: shape={dv_hidden.shape}, range=[{dv_hidden.min():.4f}, {dv_hidden.max():.4f}]")
    print(f"  HF emb: shape={hf_hidden.shape}, range=[{hf_hidden.min():.4f}, {hf_hidden.max():.4f}]")
    emb_diff = (dv_hidden.float() - hf_hidden.float()).abs().max().item()
    print(f"  Diff: {emb_diff:.6f}")
    
    # 5. Manually trace through layer 0
    print("\n[5] Tracing layer 0 step by step...")
    
    hf_layer0 = hf_model.model.layers[0]
    
    with torch.no_grad():
        # Create position_ids
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        print(f"  Position IDs: {position_ids}")
        
        # Input layernorm
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        hf_normed = hf_layer0.input_layernorm(hf_hidden)
        norm_diff = (dv_normed.float() - hf_normed.float()).abs().max().item()
        print(f"\n  After input_layernorm:")
        print(f"    DV: range=[{dv_normed.min():.4f}, {dv_normed.max():.4f}]")
        print(f"    HF: range=[{hf_normed.min():.4f}, {hf_normed.max():.4f}]")
        print(f"    Diff: {norm_diff:.6f}")
        
        # Check RoPE computation
        print(f"\n  Checking RoPE...")
        dv_cos, dv_sin = dv_block0.self_attn.rotary_emb(hf_normed, position_ids)
        hf_cos, hf_sin = hf_layer0.self_attn.rotary_emb(hf_normed, position_ids)
        
        cos_diff = (dv_cos.float() - hf_cos.float()).abs().max().item()
        sin_diff = (dv_sin.float() - hf_sin.float()).abs().max().item()
        print(f"    DV cos shape: {dv_cos.shape}, sin shape: {dv_sin.shape}")
        print(f"    HF cos shape: {hf_cos.shape}, sin shape: {hf_sin.shape}")
        print(f"    cos diff: {cos_diff:.6f}, sin diff: {sin_diff:.6f}")
        
        # Check Q, K, V projections
        print(f"\n  Checking Q, K, V projections...")
        dv_q = dv_block0.self_attn.q_proj(dv_normed)
        dv_k = dv_block0.self_attn.k_proj(dv_normed)
        dv_v = dv_block0.self_attn.v_proj(dv_normed)
        
        hf_q = hf_layer0.self_attn.q_proj(hf_normed)
        hf_k = hf_layer0.self_attn.k_proj(hf_normed)
        hf_v = hf_layer0.self_attn.v_proj(hf_normed)
        
        q_diff = (dv_q.float() - hf_q.float()).abs().max().item()
        k_diff = (dv_k.float() - hf_k.float()).abs().max().item()
        v_diff = (dv_v.float() - hf_v.float()).abs().max().item()
        print(f"    Q diff: {q_diff:.6f}")
        print(f"    K diff: {k_diff:.6f}")
        print(f"    V diff: {v_diff:.6f}")
        
        # Run full attention
        print(f"\n  Running full attention...")
        dv_attn_out, _, _ = dv_block0.self_attn(
            hidden_states=dv_normed,
            position_ids=position_ids,
        )
        
        # For HF we need to create attention mask
        hf_attn_out, _, _ = hf_layer0.self_attn(
            hidden_states=hf_normed,
            position_ids=position_ids,
        )
        
        attn_diff = (dv_attn_out.float() - hf_attn_out.float()).abs().max().item()
        print(f"    DV attn out: range=[{dv_attn_out.min():.4f}, {dv_attn_out.max():.4f}]")
        print(f"    HF attn out: range=[{hf_attn_out.min():.4f}, {hf_attn_out.max():.4f}]")
        print(f"    Attn output diff: {attn_diff:.6f}")
        
        # Residual + post-attn norm
        dv_hidden2 = dv_hidden + dv_attn_out
        hf_hidden2 = hf_hidden + hf_attn_out
        
        dv_mlp_input = dv_block0.post_attention_layernorm(dv_hidden2)
        hf_mlp_input = hf_layer0.post_attention_layernorm(hf_hidden2)
        
        post_norm_diff = (dv_mlp_input.float() - hf_mlp_input.float()).abs().max().item()
        print(f"\n  After post_attention_layernorm:")
        print(f"    Diff: {post_norm_diff:.6f}")
        
        # MLP
        dv_mlp_out = dv_block0.mlp(dv_mlp_input)
        hf_mlp_out = hf_layer0.mlp(hf_mlp_input)
        
        mlp_diff = (dv_mlp_out.float() - hf_mlp_out.float()).abs().max().item()
        print(f"\n  MLP output:")
        print(f"    DV: range=[{dv_mlp_out.min():.4f}, {dv_mlp_out.max():.4f}]")
        print(f"    HF: range=[{hf_mlp_out.min():.4f}, {hf_mlp_out.max():.4f}]")
        print(f"    Diff: {mlp_diff:.6f}")
        
        # Final layer output
        dv_final = dv_hidden2 + dv_mlp_out
        hf_final = hf_hidden2 + hf_mlp_out
        
        final_diff = (dv_final.float() - hf_final.float()).abs().max().item()
        print(f"\n  Final layer 0 output:")
        print(f"    DV: range=[{dv_final.min():.4f}, {dv_final.max():.4f}]")
        print(f"    HF: range=[{hf_final.min():.4f}, {hf_final.max():.4f}]")
        print(f"    Diff: {final_diff:.6f}")
        
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
