#!/usr/bin/env python
"""
Direct comparison test: DejaVu forward pass vs HuggingFace forward pass.
This tests the actual model computation, not the pipeline.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    model_path = "./pretrained_models/llama-3.2-3b"
    
    print("=" * 60)
    print("DejaVu vs HuggingFace Forward Pass Comparison")
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
    dv_blocks = []
    for i in range(config.num_hidden_layers):
        block = LlamaBlock.from_pretrained(model_path, layer_index=i).cuda().half()
        dv_blocks.append(block)
    dv_lm_head = LlamaLMHead.from_pretrained(model_path).cuda().half()
    
    # 3. Prepare input
    print("\n[3] Preparing input...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    print(f"  Input: '{text}'")
    print(f"  Token IDs: {input_ids}")
    print(f"  Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
    
    # 4. HuggingFace forward pass
    print("\n[4] HuggingFace forward pass...")
    with torch.no_grad():
        hf_output = hf_model(input_ids, output_hidden_states=True)
        hf_logits = hf_output.logits
        
    print(f"  HF logits shape: {hf_logits.shape}")
    print(f"  HF logits range: [{hf_logits.min():.4f}, {hf_logits.max():.4f}]")
    print(f"  HF logits mean: {hf_logits.mean():.4f}")
    
    # Compute HF log probs for ground truth
    hf_log_probs = torch.nn.functional.log_softmax(hf_logits, dim=-1)
    hf_target_log_probs = torch.gather(
        hf_log_probs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    print(f"  HF target log probs: {hf_target_log_probs}")
    print(f"  HF PPL (sentence): {torch.exp(-hf_target_log_probs.mean()).item():.4f}")
    
    # 5. DejaVu forward pass
    print("\n[5] DejaVu forward pass...")
    with torch.no_grad():
        # Embeddings
        hidden = dv_emb(input_ids)
        print(f"  After emb: shape={hidden.shape}, range=[{hidden.min():.4f}, {hidden.max():.4f}]")
        
        # Check embedding match
        hf_emb = hf_output.hidden_states[0]
        emb_diff = (hidden.float() - hf_emb.float()).abs().max().item()
        print(f"  Embedding diff from HF: {emb_diff:.6f}")
        
        # Layers
        for i, block in enumerate(dv_blocks):
            hidden, _ = block(hidden, layer_past=None, mask=None)
            
            if i < 3 or i == config.num_hidden_layers - 1:
                # Compare with HF hidden states
                hf_hidden = hf_output.hidden_states[i + 1]
                diff = (hidden.float() - hf_hidden.float()).abs().max().item()
                print(f"  After layer {i}: range=[{hidden.min():.4f}, {hidden.max():.4f}], diff from HF: {diff:.6f}")
        
        # LM Head
        dv_logits = dv_lm_head(hidden)
        
    print(f"\n  DV logits shape: {dv_logits.shape}")
    print(f"  DV logits range: [{dv_logits.min():.4f}, {dv_logits.max():.4f}]")
    print(f"  DV logits mean: {dv_logits.mean():.4f}")
    
    # Compare logits
    logits_diff = (dv_logits.float() - hf_logits.float()).abs().max().item()
    print(f"  Logits diff from HF: {logits_diff:.6f}")
    
    # Compute DV log probs
    dv_log_probs = torch.nn.functional.log_softmax(dv_logits, dim=-1)
    dv_target_log_probs = torch.gather(
        dv_log_probs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    print(f"  DV target log probs: {dv_target_log_probs}")
    print(f"  DV PPL (sentence): {torch.exp(-dv_target_log_probs.mean()).item():.4f}")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  HuggingFace PPL: {torch.exp(-hf_target_log_probs.mean()).item():.4f}")
    print(f"  DejaVu PPL:      {torch.exp(-dv_target_log_probs.mean()).item():.4f}")
    print(f"  Logits max diff: {logits_diff:.6f}")
    
    if logits_diff < 0.01:
        print("\n  ✓ Forward passes MATCH!")
    else:
        print("\n  ✗ Forward passes DIFFER - issue is in model computation")
        print("    Check which layer diverges first")

if __name__ == "__main__":
    main()
