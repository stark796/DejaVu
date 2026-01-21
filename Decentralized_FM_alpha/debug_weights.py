#!/usr/bin/env python
"""
Debug script to compare DejaVu converted weights with HuggingFace original.
This helps diagnose the perplexity discrepancy.
"""

import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    model_path = "./pretrained_models/llama-3.2-3b"
    
    print("=" * 60)
    print("DejaVu Llama Weight Debug")
    print("=" * 60)
    
    # 1. Check if weights exist
    print("\n[1] Checking weight files...")
    required_files = ["pytorch_embs.pt", "pytorch_lm_head.pt", "pytorch_0.pt"]
    for f in required_files:
        path = os.path.join(model_path, f)
        if os.path.exists(path):
            print(f"  ✓ {f} exists")
        else:
            print(f"  ✗ {f} MISSING!")
            return
    
    # 2. Load DejaVu converted weights
    print("\n[2] Loading DejaVu converted weights...")
    dejavu_emb = torch.load(os.path.join(model_path, "pytorch_embs.pt"))
    dejavu_layer0 = torch.load(os.path.join(model_path, "pytorch_0.pt"))
    dejavu_lm_head = torch.load(os.path.join(model_path, "pytorch_lm_head.pt"))
    
    print(f"  Embedding keys: {list(dejavu_emb.keys())}")
    print(f"  Layer 0 keys: {list(dejavu_layer0.keys())[:10]}...")
    print(f"  LM Head keys: {list(dejavu_lm_head.keys())}")
    
    # 3. Load HuggingFace original
    print("\n[3] Loading HuggingFace original for comparison...")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    hf_state = hf_model.state_dict()
    
    # 4. Compare embeddings
    print("\n[4] Comparing embeddings...")
    hf_emb = hf_state["model.embed_tokens.weight"]
    dv_emb = dejavu_emb["embed_tokens.weight"]
    
    print(f"  HF shape: {hf_emb.shape}, dtype: {hf_emb.dtype}")
    print(f"  DV shape: {dv_emb.shape}, dtype: {dv_emb.dtype}")
    
    if hf_emb.shape == dv_emb.shape:
        diff = (hf_emb.float() - dv_emb.float()).abs().max().item()
        print(f"  Max diff: {diff}")
        if diff < 1e-3:
            print("  ✓ Embeddings match!")
        else:
            print("  ✗ Embeddings DIFFER!")
    else:
        print("  ✗ Shape mismatch!")
    
    # 5. Compare layer 0 weights
    print("\n[5] Comparing layer 0 weights...")
    
    comparisons = [
        ("input_layernorm.weight", "model.layers.0.input_layernorm.weight"),
        ("self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("mlp.gate_proj.weight", "model.layers.0.mlp.gate_proj.weight"),
    ]
    
    for dv_key, hf_key in comparisons:
        if dv_key not in dejavu_layer0:
            print(f"  ✗ DV missing: {dv_key}")
            continue
        if hf_key not in hf_state:
            print(f"  ✗ HF missing: {hf_key}")
            continue
            
        dv_w = dejavu_layer0[dv_key]
        hf_w = hf_state[hf_key]
        
        if dv_w.shape != hf_w.shape:
            print(f"  ✗ {dv_key}: shape mismatch {dv_w.shape} vs {hf_w.shape}")
        else:
            diff = (dv_w.float() - hf_w.float()).abs().max().item()
            status = "✓" if diff < 1e-3 else "✗"
            print(f"  {status} {dv_key}: max diff = {diff:.6f}")
    
    # 6. Quick forward pass test
    print("\n[6] Quick forward pass comparison...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    
    hf_model = hf_model.to("cuda").eval()
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        hf_out = hf_model(**inputs, labels=inputs["input_ids"])
        print(f"  HuggingFace loss: {hf_out.loss.item():.4f}")
        print(f"  HuggingFace PPL: {torch.exp(hf_out.loss).item():.2f}")
    
    # 7. Check DejaVu module loading
    print("\n[7] Testing DejaVu module loading...")
    try:
        from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
        
        emb = LlamaEmbeddings.from_pretrained(model_path)
        block = LlamaBlock.from_pretrained(model_path, layer_index=0)
        lm_head = LlamaLMHead.from_pretrained(model_path)
        
        print("  ✓ DejaVu modules loaded successfully")
        
        # Check if weights match
        dv_loaded_emb = emb.embed_tokens.weight.data
        diff = (dv_loaded_emb.float().cpu() - dv_emb.float()).abs().max().item()
        print(f"  Loaded emb vs saved: max diff = {diff:.6f}")
        
    except Exception as e:
        print(f"  ✗ Failed to load DejaVu modules: {e}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
