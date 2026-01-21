#!/usr/bin/env python
"""
Test: Disable SDPA in HuggingFace and compare again.
The 0.167 diff is likely because HF uses Flash Attention / SDPA.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Testing with SDPA disabled")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load HF model with SDPA disabled
    print("\n[1] Loading HF model with attn_implementation='eager'...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    # Force eager attention (no SDPA)
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager"  # Disable SDPA/Flash Attention
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    # Load DejaVu
    print("\n[2] Loading DejaVu model...")
    config = LlamaConfig.from_pretrained(model_path)
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_blocks = [LlamaBlock.from_pretrained(model_path, layer_index=i).cuda().half() 
                 for i in range(config.num_hidden_layers)]
    dv_lm_head = LlamaLMHead.from_pretrained(model_path).cuda().half()
    
    # Prepare input
    print("\n[3] Running forward pass...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    seq_len = input_ids.size(1)
    print(f"  Input: '{text}', {seq_len} tokens")
    
    with torch.no_grad():
        # DejaVu forward
        dv_hidden = dv_emb(input_ids)
        for i, block in enumerate(dv_blocks):
            dv_hidden, _ = block(dv_hidden, layer_past=None, mask=None)
        dv_logits = dv_lm_head(dv_hidden)
        
        # HF forward
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_logits = hf_out.logits
        
        # Compare layer-by-layer
        print("\n[4] Layer-by-layer comparison (with eager attention):")
        
        dv_hidden2 = dv_emb(input_ids)
        for i in range(config.num_hidden_layers):
            hf_layer_out = hf_out.hidden_states[i + 1]
            dv_hidden2, _ = dv_blocks[i](dv_hidden2, layer_past=None, mask=None)
            diff = (dv_hidden2 - hf_layer_out).abs().max().item()
            if i < 5 or i >= config.num_hidden_layers - 3:
                print(f"  Layer {i}: diff = {diff:.6f}")
            elif i == 5:
                print("  ...")
        
        # Compare logits
        logits_diff = (dv_logits - hf_logits).abs().max().item()
        print(f"\n  Logits diff: {logits_diff:.6f}")
        
        # Compute perplexity
        import torch.nn.functional as F
        
        dv_log_probs = F.log_softmax(dv_logits, dim=-1)
        hf_log_probs = F.log_softmax(hf_logits, dim=-1)
        
        dv_target_lp = torch.gather(dv_log_probs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        hf_target_lp = torch.gather(hf_log_probs[:, :-1], -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        dv_ppl = torch.exp(-dv_target_lp.mean()).item()
        hf_ppl = torch.exp(-hf_target_lp.mean()).item()
        
        print(f"\n[5] Perplexity comparison:")
        print(f"  DejaVu PPL:      {dv_ppl:.2f}")
        print(f"  HuggingFace PPL: {hf_ppl:.2f}")
        
        if abs(dv_ppl - hf_ppl) < 1:
            print("\n  ✓ Perplexities match!")
        else:
            print(f"\n  PPL difference: {abs(dv_ppl - hf_ppl):.2f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
