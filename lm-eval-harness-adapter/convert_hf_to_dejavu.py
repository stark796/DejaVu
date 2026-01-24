#!/usr/bin/env python3
"""
Re-convert HuggingFace Llama weights to DejaVu format with ALL weights including layernorms.

Usage:
    python convert_hf_to_dejavu.py \
        --hf-model meta-llama/Llama-3.2-3B \
        --output-dir ../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b-fixed
"""

import argparse
import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, default="meta-llama/Llama-3.2-3B",
                        help="HuggingFace model name or path")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for DejaVu format weights")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading HuggingFace model: {args.hf_model}")
    model = LlamaForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        device_map="cpu"  # Load on CPU for conversion
    )
    
    print(f"Model loaded. Extracting weights...")
    state_dict = model.state_dict()
    
    # Print structure
    print("\nHuggingFace model keys structure:")
    layer_keys = [k for k in state_dict.keys() if "layers.0." in k]
    for k in layer_keys:
        print(f"  {k}")
    
    # Map from HF format to DejaVu format
    # HF: model.layers.{i}.input_layernorm.weight
    # HF: model.layers.{i}.post_attention_layernorm.weight
    # HF: model.layers.{i}.self_attn.q_proj.weight
    # etc.
    
    num_layers = model.config.num_hidden_layers
    print(f"\nExtracting {num_layers} layers...")
    
    # Save embeddings
    emb_state = {
        "embed_tokens.weight": state_dict["model.embed_tokens.weight"]
    }
    emb_path = os.path.join(args.output_dir, "pytorch_embs.pt")
    torch.save(emb_state, emb_path)
    print(f"Saved embeddings to {emb_path}")
    
    # Save each layer
    for layer_idx in range(num_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        
        layer_state = {}
        for key, value in state_dict.items():
            if key.startswith(layer_prefix):
                # Remove prefix for DejaVu format
                new_key = key[len(layer_prefix):]
                layer_state[new_key] = value
        
        if layer_idx == 0:
            print(f"\nLayer 0 keys being saved:")
            for k in sorted(layer_state.keys()):
                print(f"  {k}")
        
        layer_path = os.path.join(args.output_dir, f"pytorch_{layer_idx}.pt")
        torch.save(layer_state, layer_path)
        print(f"Saved layer {layer_idx}")
    
    # Save LM head
    lm_head_state = {
        "weight": state_dict["lm_head.weight"]
    }
    # Also need final layernorm
    if "model.norm.weight" in state_dict:
        lm_head_state["norm.weight"] = state_dict["model.norm.weight"]
    
    lm_head_path = os.path.join(args.output_dir, "pytorch_lm_head.pt")
    torch.save(lm_head_state, lm_head_path)
    print(f"Saved LM head to {lm_head_path}")
    
    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer")
    
    # Save config
    model.config.save_pretrained(args.output_dir)
    print(f"Saved config")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    print("\nNow update benchmark_inference.py to use:")
    print(f"  --model-path {args.output_dir}")


if __name__ == "__main__":
    main()
