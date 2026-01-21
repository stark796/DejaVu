#!/usr/bin/env python
"""
Verification test: Confirm the RoPE fix works.
After the fix, DejaVu should match HuggingFace exactly.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("VERIFICATION: Testing RoPE Fix")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load models
    print("\n[1] Loading models...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    config = LlamaConfig.from_pretrained(model_path)
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_blocks = [LlamaBlock.from_pretrained(model_path, layer_index=i).cuda().half() 
                 for i in range(config.num_hidden_layers)]
    dv_lm_head = LlamaLMHead.from_pretrained(model_path).cuda().half()
    
    # Check inv_freq
    print("\n[2] Checking inv_freq initialization...")
    dv_inv_freq = dv_blocks[0].self_attn.rotary_emb.inv_freq
    hf_inv_freq = hf_model.model.rotary_emb.inv_freq
    
    inv_freq_diff = (dv_inv_freq - hf_inv_freq).abs().max().item()
    print(f"  DV inv_freq[:5]: {dv_inv_freq[:5]}")
    print(f"  HF inv_freq[:5]: {hf_inv_freq[:5]}")
    print(f"  inv_freq diff: {inv_freq_diff:.6f}")
    
    if inv_freq_diff < 0.001:
        print("  ✓ inv_freq matches!")
    else:
        print("  ✗ inv_freq still differs!")
        return
    
    # Run forward pass
    print("\n[3] Running forward pass...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    
    with torch.no_grad():
        # DejaVu
        hidden = dv_emb(input_ids)
        for block in dv_blocks:
            hidden, _ = block(hidden, layer_past=None, mask=None)
        dv_logits = dv_lm_head(hidden)
        
        # HuggingFace
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_logits = hf_out.logits
        
        # Compare layer by layer
        print("\n[4] Layer-by-layer comparison:")
        hidden2 = dv_emb(input_ids)
        for i in range(config.num_hidden_layers):
            hf_hidden = hf_out.hidden_states[i + 1]
            hidden2, _ = dv_blocks[i](hidden2, layer_past=None, mask=None)
            diff = (hidden2 - hf_hidden).abs().max().item()
            if i < 3 or i >= config.num_hidden_layers - 2:
                print(f"  Layer {i}: diff = {diff:.6f}")
            elif i == 3:
                print("  ...")
        
        # Logits comparison
        logits_diff = (dv_logits - hf_logits).abs().max().item()
        print(f"\n  Logits diff: {logits_diff:.6f}")
        
        # Perplexity
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
        print(f"  Difference:      {abs(dv_ppl - hf_ppl):.2f}")
        
        if abs(dv_ppl - hf_ppl) < 1:
            print("\n  ✓ SUCCESS! Perplexities match!")
        else:
            print("\n  ✗ Still some difference")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
