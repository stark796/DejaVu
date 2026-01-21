#!/usr/bin/env python
"""
Debug script to trace exactly where the forward pass diverges.
Simplified version that works with newer transformers API.
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
    
    # 5. Trace layer 0
    print("\n[5] Tracing layer 0...")
    
    hf_layer0 = hf_model.model.layers[0]
    
    with torch.no_grad():
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        print(f"  Position IDs: {position_ids}")
        
        # Input layernorm
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        hf_normed = hf_layer0.input_layernorm(hf_hidden)
        norm_diff = (dv_normed.float() - hf_normed.float()).abs().max().item()
        print(f"\n  input_layernorm diff: {norm_diff:.6f}")
        
        # Q, K, V projections
        dv_q = dv_block0.self_attn.q_proj(dv_normed)
        hf_q = hf_layer0.self_attn.q_proj(hf_normed)
        q_diff = (dv_q.float() - hf_q.float()).abs().max().item()
        print(f"  Q projection diff: {q_diff:.6f}")
        
        dv_k = dv_block0.self_attn.k_proj(dv_normed)
        hf_k = hf_layer0.self_attn.k_proj(hf_normed)
        k_diff = (dv_k.float() - hf_k.float()).abs().max().item()
        print(f"  K projection diff: {k_diff:.6f}")
        
        dv_v = dv_block0.self_attn.v_proj(dv_normed)
        hf_v = hf_layer0.self_attn.v_proj(hf_normed)
        v_diff = (dv_v.float() - hf_v.float()).abs().max().item()
        print(f"  V projection diff: {v_diff:.6f}")
        
        # Full layer comparison
        print(f"\n  Running full layer 0...")
        
        # DejaVu - WITHOUT passing position_ids (uses internally generated)
        dv_layer_out, _ = dv_block0(dv_hidden, layer_past=None, mask=None)
        
        # HuggingFace
        hf_layer_out = hf_layer0(hf_hidden, position_ids=position_ids)[0]
        
        layer_diff = (dv_layer_out.float() - hf_layer_out.float()).abs().max().item()
        print(f"    DV layer0 out: range=[{dv_layer_out.min():.4f}, {dv_layer_out.max():.4f}]")
        print(f"    HF layer0 out: range=[{hf_layer_out.min():.4f}, {hf_layer_out.max():.4f}]")
        print(f"    Layer 0 output diff (DV uses internal pos_ids): {layer_diff:.6f}")
        
        # DejaVu - WITH explicit position_ids
        dv_layer_out2, _ = dv_block0(dv_hidden, layer_past=None, mask=None, position_ids=position_ids)
        layer_diff2 = (dv_layer_out2.float() - hf_layer_out.float()).abs().max().item()
        print(f"    Layer 0 output diff (DV uses explicit pos_ids): {layer_diff2:.6f}")
        
        if layer_diff2 > 0.01:
            print("\n  *** OUTPUTS DIFFER SIGNIFICANTLY ***")
            print("  Checking internal difference...")
            
            # Compare DV with and without explicit position_ids
            internal_diff = (dv_layer_out.float() - dv_layer_out2.float()).abs().max().item()
            print(f"    DV (internal pos_ids) vs DV (explicit pos_ids): {internal_diff:.6f}")
            
            if internal_diff > 0.001:
                print("    --> Position IDs are being generated differently!")
            else:
                print("    --> Position IDs match, issue is elsewhere in attention")
        else:
            print("\n  ✓ Layer 0 outputs match well!")
        
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
