#!/usr/bin/env python3
"""
Detailed debug: Check what's in each weight file and verify loading.
"""
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

def main():
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    
    print("="*60)
    print("WEIGHT FILE ANALYSIS")
    print("="*60)
    
    # Check embeddings
    emb_path = os.path.join(model_path, "pytorch_embs.pt")
    if os.path.exists(emb_path):
        sd = torch.load(emb_path, map_location='cpu')
        print(f"\n{emb_path}:")
        for k, v in sd.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"\n{emb_path}: NOT FOUND!")
    
    # Check layer 0
    layer_path = os.path.join(model_path, "pytorch_0.pt")
    if os.path.exists(layer_path):
        sd = torch.load(layer_path, map_location='cpu')
        print(f"\n{layer_path}:")
        for k, v in sd.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"\n{layer_path}: NOT FOUND!")
    
    # Check LM head
    lm_head_path = os.path.join(model_path, "pytorch_lm_head.pt")
    if os.path.exists(lm_head_path):
        sd = torch.load(lm_head_path, map_location='cpu')
        print(f"\n{lm_head_path}:")
        for k, v in sd.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"\n{lm_head_path}: NOT FOUND!")
    
    # Now check what the modules EXPECT
    print("\n" + "="*60)
    print("MODULE EXPECTED KEYS")
    print("="*60)
    
    from transformers import LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-3B")
    
    # Embeddings
    emb_module = torch.nn.utils.skip_init(LlamaEmbeddings, config)
    print(f"\nLlamaEmbeddings expects:")
    for k in emb_module.state_dict().keys():
        print(f"  {k}")
    
    # Layer
    layer_module = torch.nn.utils.skip_init(LlamaBlock, config, 0)
    print(f"\nLlamaBlock expects:")
    for k in layer_module.state_dict().keys():
        print(f"  {k}")
    
    # LM Head
    lm_head_module = torch.nn.utils.skip_init(LlamaLMHead, config)
    print(f"\nLlamaLMHead expects:")
    for k in lm_head_module.state_dict().keys():
        print(f"  {k}")

if __name__ == "__main__":
    main()
