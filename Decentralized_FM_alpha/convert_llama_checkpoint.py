"""
Convert Llama 3.2 Checkpoint for DejaVu Framework

This script converts HuggingFace Llama checkpoints to the format expected
by the DejaVu distributed inference framework.

Usage:
    python convert_llama_checkpoint.py --model-name meta-llama/Llama-3.2-3B --save-path ./pretrained_models/llama-3.2-3b

Output structure:
    save_path/
        config.json         - LlamaConfig
        tokenizer files     - Tokenizer
        pytorch_embs.pt     - Embedding weights
        pytorch_lm_head.pt  - Final norm + LM head weights
        pytorch_0.pt        - Layer 0 weights
        pytorch_1.pt        - Layer 1 weights
        ...

IMPORTANT NOTES:
1. Llama 3.2 3B has 28 layers
2. Uses RMSNorm (no bias) instead of LayerNorm
3. Uses RoPE (computed on the fly, not stored)
4. MLP structure: gate_proj, up_proj, down_proj (gated MLP)
5. No bias in any linear layers
"""

import torch
import json
import os
import argparse
import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def convert_llama_checkpoint(model_name, save_path, use_auth_token=None):
    """
    Convert a HuggingFace Llama checkpoint to DejaVu format.
    
    Args:
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.2-3B")
        save_path: Directory to save converted checkpoint
        use_auth_token: HuggingFace auth token (needed for gated models)
    """
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Loading config from {model_name}...")
    config = AutoConfig.from_pretrained(model_name, token=use_auth_token)
    config.save_pretrained(save_path)
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=use_auth_token)
    tokenizer.save_pretrained(save_path)
    
    print(f"Loading model from {model_name}...")
    print("This may take a while for large models...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=use_auth_token,
            torch_dtype=torch.float16,  # Use fp16 to save memory
            low_cpu_mem_usage=True,
    )
    
    state_dict = model.state_dict()
    
    # Print model structure for debugging
    print("\nModel state dict keys (first 20):")
    for i, k in enumerate(list(state_dict.keys())[:20]):
        print(f"  {k}: {state_dict[k].shape}")
    print("...")
    
    # === Save Embeddings ===
    print('\nSaving embeddings...')
    emb_item = {}
    emb_item['embed_tokens.weight'] = state_dict['model.embed_tokens.weight']
    torch.save(emb_item, os.path.join(save_path, 'pytorch_embs.pt'))
    print(f"  Embedding shape: {emb_item['embed_tokens.weight'].shape}")
    
    # === Save LM Head ===
    print('\nSaving LM head...')
    lm_head_item = {}
    # Llama ties embeddings and LM head weights
    if 'lm_head.weight' in state_dict:
        lm_head_item['lm_head.weight'] = state_dict['lm_head.weight']
    else:
        # If tied, use embedding weights
        lm_head_item['lm_head.weight'] = state_dict['model.embed_tokens.weight']
    
    # Final RMSNorm
    lm_head_item['norm.weight'] = state_dict['model.norm.weight']
    torch.save(lm_head_item, os.path.join(save_path, 'pytorch_lm_head.pt'))
    print(f"  LM head shape: {lm_head_item['lm_head.weight'].shape}")
    print(f"  Final norm shape: {lm_head_item['norm.weight'].shape}")
    
    # === Save Layers ===
    print(f'\nSaving {config.num_hidden_layers} layers...')
    for i in tqdm.tqdm(range(config.num_hidden_layers)):
        layer_prefix = f'model.layers.{i}.'
        
        layer_item = {}
        layer_maps = {k: v for k, v in state_dict.items() if k.startswith(layer_prefix)}
        
        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            layer_item[new_k] = v
        
        torch.save(layer_item, os.path.join(save_path, f'pytorch_{i}.pt'))
        
        # Print layer structure for first layer
        if i == 0:
            print(f"\n  Layer 0 structure:")
            for k, v in layer_item.items():
                print(f"    {k}: {v.shape}")
        
        del layer_item
    
    print(f"\nCheckpoint conversion complete!")
    print(f"Saved to: {save_path}")
    print(f"\nModel info:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.num_attention_heads}")
    print(f"  - Num KV heads: {config.num_key_value_heads}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    print(f"  - Vocab size: {config.vocab_size}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Llama checkpoints for DejaVu')
    parser.add_argument('--model-name', type=str, default='meta-llama/Llama-3.2-3B',
                        help='HuggingFace model name')
    parser.add_argument('--save-path', type=str, default='./pretrained_models/llama-3.2-3b',
                        help='Path to save converted checkpoint')
    parser.add_argument('--auth-token', type=str, default=None,
                        help='HuggingFace auth token for gated models')
    args = parser.parse_args()
    
    convert_llama_checkpoint(
        model_name=args.model_name,
        save_path=args.save_path,
        use_auth_token=args.auth_token,
    )
