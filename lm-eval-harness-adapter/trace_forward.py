#!/usr/bin/env python3
"""
Step-by-step forward pass comparison to find where HF and DejaVu diverge.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

def main():
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    tokenizer_name = "meta-llama/Llama-3.2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("STEP-BY-STEP FORWARD PASS COMPARISON")
    print("="*60)
    
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)
    
    # Load models
    print("\nLoading HuggingFace model...")
    hf_model = LlamaForCausalLM.from_pretrained(
        tokenizer_name, torch_dtype=torch.float16, device_map=device
    )
    
    print("Loading DejaVu model...")
    dv_config = LlamaConfig.from_pretrained(tokenizer_name)
    dv_embeddings = LlamaEmbeddings.from_pretrained(model_path, config=dv_config).to(device).half()
    dv_layer0 = LlamaBlock.from_pretrained(model_path, config=dv_config, layer_index=0).to(device).half()
    
    print("\n" + "="*60)
    print("STEP 1: EMBEDDINGS")
    print("="*60)
    
    with torch.no_grad():
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        dv_hidden = dv_embeddings(input_ids)
        
        diff = (hf_hidden - dv_hidden).abs().max().item()
        print(f"After embeddings - max diff: {diff}")
        print(f"HF sample: {hf_hidden[0, 0, :5].tolist()}")
        print(f"DV sample: {dv_hidden[0, 0, :5].tolist()}")
    
    print("\n" + "="*60)
    print("STEP 2a: INPUT LAYERNORM (Layer 0)")
    print("="*60)
    
    with torch.no_grad():
        hf_layer0 = hf_model.model.layers[0]
        
        # Apply input layernorm manually
        hf_after_ln1 = hf_layer0.input_layernorm(hf_hidden)
        dv_after_ln1 = dv_layer0.input_layernorm(dv_hidden)
        
        diff = (hf_after_ln1 - dv_after_ln1).abs().max().item()
        print(f"After input_layernorm - max diff: {diff}")
        print(f"HF sample: {hf_after_ln1[0, 0, :5].tolist()}")
        print(f"DV sample: {dv_after_ln1[0, 0, :5].tolist()}")
    
    print("\n" + "="*60)
    print("STEP 2b: Q/K/V PROJECTIONS")
    print("="*60)
    
    with torch.no_grad():
        # Q projection
        hf_q = hf_layer0.self_attn.q_proj(hf_after_ln1)
        dv_q = dv_layer0.self_attn.q_proj(dv_after_ln1)
        diff = (hf_q - dv_q).abs().max().item()
        print(f"Q projection - max diff: {diff}")
        
        # K projection
        hf_k = hf_layer0.self_attn.k_proj(hf_after_ln1)
        dv_k = dv_layer0.self_attn.k_proj(dv_after_ln1)
        diff = (hf_k - dv_k).abs().max().item()
        print(f"K projection - max diff: {diff}")
        
        # V projection
        hf_v = hf_layer0.self_attn.v_proj(hf_after_ln1)
        dv_v = dv_layer0.self_attn.v_proj(dv_after_ln1)
        diff = (hf_v - dv_v).abs().max().item()
        print(f"V projection - max diff: {diff}")
    
    print("\n" + "="*60)
    print("STEP 3: FULL LAYER 0 FORWARD")
    print("="*60)
    
    with torch.no_grad():
        # HF uses a different API - need to call with proper arguments
        # Let me trace through what HF does internally
        
        # For DejaVu, use position_ids explicitly
        seq_len = input_ids.size(1)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # DejaVu layer forward
        dv_out, _ = dv_layer0(dv_hidden, layer_past=None, position_ids=position_ids)
        
        # For HF, we need to trace through without using full model
        # Use its rotary embedding
        cos, sin = hf_layer0.self_attn.rotary_emb(hf_v, position_ids)
        
        print(f"Position IDs: {position_ids}")
        print(f"HF cos sample: {cos[0, 0, :5].tolist()}")
        print(f"HF sin sample: {sin[0, 0, :5].tolist()}")
        
        # Check if DejaVu uses same rotary embeddings
        dv_cos, dv_sin = dv_layer0.self_attn.rotary_emb(dv_v, position_ids)
        cos_diff = (cos - dv_cos).abs().max().item()
        sin_diff = (sin - dv_sin).abs().max().item()
        print(f"\nRotary cos diff: {cos_diff}")
        print(f"Rotary sin diff: {sin_diff}")
    
    print("\n" + "="*60)
    print("If rotary embeddings differ, that's the bug!")
    print("="*60)


if __name__ == "__main__":
    main()
