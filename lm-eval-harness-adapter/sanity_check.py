#!/usr/bin/env python3
"""
Sanity check: Run the model on a simple prompt and verify logprobs make sense.
Compare with HuggingFace reference implementation.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

from transformers import AutoTokenizer, LlamaForCausalLM
import argparse


def test_with_hf_reference():
    """Compare our DejaVu implementation with HuggingFace reference."""
    
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    tokenizer_name = "meta-llama/Llama-3.2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test prompt
    test_prompt = "The capital of France is"
    
    print("="*60)
    print("SANITY CHECK: Comparing DejaVu vs HuggingFace Llama")
    print("="*60)
    print(f"Prompt: '{test_prompt}'")
    print(f"Device: {device}")
    print()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    print(f"Input IDs: {input_ids[0].tolist()}")
    print()
    
    # ---------------------------
    # Test 1: DejaVu implementation
    # ---------------------------
    print("Loading DejaVu model...")
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    from transformers import LlamaConfig
    
    config = LlamaConfig.from_pretrained(tokenizer_name)
    
    embeddings = LlamaEmbeddings.from_pretrained(model_path, config=config).to(device).half()
    layers = {}
    for i in range(config.num_hidden_layers):
        layers[f"block{i}"] = LlamaBlock.from_pretrained(
            model_path, config=config, layer_index=i
        ).to(device).half()
    lm_head = LlamaLMHead.from_pretrained(model_path, config=config).to(device).half()
    
    with torch.no_grad():
        hidden_states = embeddings(input_ids)
        layer_past = None
        for i in range(config.num_hidden_layers):
            layer = layers[f"block{i}"]
            hidden_states, layer_past = layer(hidden_states, layer_past=layer_past)
        logits_dejavu = lm_head(hidden_states)
        
        # Get next token prediction
        next_token_logits = logits_dejavu[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        top5 = probs.topk(5)
        
        print("DejaVu Top 5 next token predictions:")
        for i in range(5):
            token_id = top5.indices[i].item()
            prob = top5.values[i].item()
            token = tokenizer.decode([token_id])
            print(f"  {i+1}. '{token}' (id={token_id}): {prob:.4f}")
    print()
    
    # ---------------------------
    # Test 2: HuggingFace reference (if available)
    # ---------------------------
    try:
        print("Loading HuggingFace reference model (may take time)...")
        hf_model = LlamaForCausalLM.from_pretrained(
            tokenizer_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        with torch.no_grad():
            outputs = hf_model(input_ids)
            logits_hf = outputs.logits
            
            next_token_logits = logits_hf[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            top5 = probs.topk(5)
            
            print("HuggingFace Top 5 next token predictions:")
            for i in range(5):
                token_id = top5.indices[i].item()
                prob = top5.values[i].item()
                token = tokenizer.decode([token_id])
                print(f"  {i+1}. '{token}' (id={token_id}): {prob:.4f}")
    except Exception as e:
        print(f"Could not load HuggingFace reference: {e}")
        print("Skipping HuggingFace comparison.")
    
    print()
    print("="*60)
    print("If DejaVu predicts nonsense (random tokens) but HuggingFace is sensible,")
    print("there's a bug in weight loading or forward pass.")
    print("="*60)


if __name__ == "__main__":
    test_with_hf_reference()
