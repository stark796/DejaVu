#!/usr/bin/env python3
"""
Layer-by-layer comparison: Find where DejaVu diverges from HuggingFace.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

def main():
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    tokenizer_name = "meta-llama/Llama-3.2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test prompt
    test_prompt = "The capital of France is"
    
    print("="*60)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*60)
    
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    hf_model = LlamaForCausalLM.from_pretrained(
        tokenizer_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    hf_config = hf_model.config
    
    # Load DejaVu model
    print("Loading DejaVu model...")
    dv_config = LlamaConfig.from_pretrained(tokenizer_name)
    dv_embeddings = LlamaEmbeddings.from_pretrained(model_path, config=dv_config).to(device).half()
    dv_layers = {}
    for i in range(dv_config.num_hidden_layers):
        dv_layers[f"block{i}"] = LlamaBlock.from_pretrained(
            model_path, config=dv_config, layer_index=i
        ).to(device).half()
    dv_lm_head = LlamaLMHead.from_pretrained(model_path, config=dv_config).to(device).half()
    
    print("\n" + "="*60)
    print("COMPARING EMBEDDINGS")
    print("="*60)
    
    with torch.no_grad():
        # HuggingFace embeddings
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        
        # DejaVu embeddings
        dv_hidden = dv_embeddings(input_ids)
        
        diff = (hf_hidden - dv_hidden).abs().max().item()
        print(f"Embedding difference (max abs): {diff}")
        
        if diff > 0.01:
            print("  -> EMBEDDINGS DON'T MATCH!")
        else:
            print("  -> Embeddings match ✓")
    
    print("\n" + "="*60)
    print("COMPARING LAYERS (first 3 and last)")
    print("="*60)
    
    with torch.no_grad():
        hf_h = hf_hidden
        dv_h = dv_hidden
        dv_past = None
        
        # Compare first 3 layers
        for layer_idx in [0, 1, 2, hf_config.num_hidden_layers - 1]:
            # HuggingFace layer forward
            hf_layer = hf_model.model.layers[layer_idx]
            hf_out = hf_layer(hf_h, position_ids=torch.arange(input_ids.size(1), device=device).unsqueeze(0))
            hf_h = hf_out[0]
            
            # DejaVu layer forward
            dv_layer = dv_layers[f"block{layer_idx}"]
            dv_h, dv_past = dv_layer(dv_h, layer_past=None)
            
            diff = (hf_h - dv_h).abs().max().item()
            mean_diff = (hf_h - dv_h).abs().mean().item()
            
            print(f"Layer {layer_idx}: max_diff={diff:.6f}, mean_diff={mean_diff:.6f}")
            
            if diff > 0.1:
                print(f"  -> LAYER {layer_idx} OUTPUT DIFFERS SIGNIFICANTLY!")
                print(f"     HF  sample: {hf_h[0, 0, :5].tolist()}")
                print(f"     DV  sample: {dv_h[0, 0, :5].tolist()}")
                break
        
    print("\n" + "="*60)
    print("If a layer shows significant difference, the bug is in that layer's forward pass.")
    print("="*60)


if __name__ == "__main__":
    main()
