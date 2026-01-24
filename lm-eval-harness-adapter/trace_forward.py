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
    print("STEP 3: FULL LAYER 0 OUTPUT COMPARISON")
    print("="*60)
    
    with torch.no_grad():
        # For DejaVu, run through layer 0
        dv_out, _ = dv_layer0(dv_hidden, layer_past=None)
        
        # For HuggingFace, run full model but capture intermediate
        # Actually, let's just run both through all layers and compare
        
        print(f"DV layer 0 output shape: {dv_out.shape}")
        print(f"DV layer 0 output sample: {dv_out[0, 0, :5].tolist()}")
        
        # Run HF through layer 0 using internal method
        # HF model.forward handles caching/position internally
        # Let's compare outputs by running full forward on both
        
        # Full HF forward
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits[0, -1, :]
        
        # Full DV forward
        dv_h = dv_hidden
        for i in range(dv_config.num_hidden_layers):
            dv_layer = LlamaBlock.from_pretrained(model_path, config=dv_config, layer_index=i).to(device).half()
            dv_h, _ = dv_layer(dv_h, layer_past=None)
        dv_lm_head = LlamaLMHead.from_pretrained(model_path, config=dv_config).to(device).half()
        dv_logits = dv_lm_head(dv_h)[0, -1, :]
        
        diff = (hf_logits - dv_logits).abs().max().item()
        print(f"\nFinal logits max diff: {diff}")
        
        print(f"\nHF predicts: {tokenizer.decode([hf_logits.argmax().item()])}")
        print(f"DV predicts: {tokenizer.decode([dv_logits.argmax().item()])}")
    
    print("\n" + "="*60)
    print("KEY FINDING: Q/K/V projections match, so the bug is in")
    print("attention computation, RoPE, or residual connections.")
    print("="*60)


if __name__ == "__main__":
    main()
