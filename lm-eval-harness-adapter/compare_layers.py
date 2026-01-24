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
    print("COMPARING FINAL LOGITS")
    print("="*60)
    
    with torch.no_grad():
        # Get HuggingFace output using full model
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits
        
        # Get DejaVu output using full forward pass
        dv_hidden = dv_embeddings(input_ids)
        dv_past = None
        for layer_idx in range(dv_config.num_hidden_layers):
            dv_layer = dv_layers[f"block{layer_idx}"]
            dv_hidden, dv_past = dv_layer(dv_hidden, layer_past=None)
        dv_logits = dv_lm_head(dv_hidden)
        
        # Compare logits
        diff = (hf_logits - dv_logits).abs()
        print(f"Logits max diff: {diff.max().item():.4f}")
        print(f"Logits mean diff: {diff.mean().item():.4f}")
        
        # Compare next token predictions
        hf_next = hf_logits[0, -1, :].argmax().item()
        dv_next = dv_logits[0, -1, :].argmax().item()
        
        print(f"\nHF  next token: '{tokenizer.decode([hf_next])}' (id={hf_next})")
        print(f"DV  next token: '{tokenizer.decode([dv_next])}' (id={dv_next})")
        
        if hf_next == dv_next:
            print("  -> Predictions MATCH ✓")
        else:
            print("  -> Predictions DIFFER!")
            # Show top-5 for both
            hf_top5 = hf_logits[0, -1, :].topk(5)
            dv_top5 = dv_logits[0, -1, :].topk(5)
            print("\nHF top-5:")
            for i in range(5):
                print(f"  {tokenizer.decode([hf_top5.indices[i].item()])}: {hf_top5.values[i].item():.2f}")
            print("\nDV top-5:")
            for i in range(5):
                print(f"  {tokenizer.decode([dv_top5.indices[i].item()])}: {dv_top5.values[i].item():.2f}")


if __name__ == "__main__":
    main()
