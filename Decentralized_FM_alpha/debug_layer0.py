#!/usr/bin/env python
"""
Debug attention mask: The diff persists even with SDPA disabled.
The issue must be in how the AttentionMask is created/applied.
"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Debugging Attention Mask Difference")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load models
    print("\n[1] Loading models...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, _make_causal_mask, _prepare_decoder_attention_mask
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    config = LlamaConfig.from_pretrained(model_path)
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    # Prepare input
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()  # HF provides this
    seq_len = input_ids.size(1)
    bsz = 1
    
    print(f"\n[2] Input analysis...")
    print(f"  Text: '{text}'")
    print(f"  input_ids: {input_ids}")
    print(f"  attention_mask from tokenizer: {attention_mask}")
    
    with torch.no_grad():
        dv_hidden = dv_emb(input_ids)
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        
        print(f"\n[3] Testing different mask scenarios...")
        
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        
        # Run DejaVu with no mask (creates its own)
        dv_out_no_mask, _ = dv_block0(dv_hidden, layer_past=None, mask=None, position_ids=position_ids)
        
        # Run DejaVu with explicit all-ones mask
        ones_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device="cuda")
        dv_out_ones, _ = dv_block0(dv_hidden, layer_past=None, mask=ones_mask, position_ids=position_ids)
        
        # Run DejaVu with tokenizer's attention_mask
        dv_out_tok_mask, _ = dv_block0(dv_hidden, layer_past=None, mask=attention_mask.bool(), position_ids=position_ids)
        
        # Run HF 
        hf_out = hf_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hf_layer0_out = hf_out.hidden_states[1]
        
        # Also run HF without attention_mask
        hf_out_no_mask = hf_model(input_ids, output_hidden_states=True)
        hf_layer0_out_no_mask = hf_out_no_mask.hidden_states[1]
        
        print(f"\n  DV (no mask) vs HF (with mask): {(dv_out_no_mask - hf_layer0_out).abs().max().item():.6f}")
        print(f"  DV (ones mask) vs HF (with mask): {(dv_out_ones - hf_layer0_out).abs().max().item():.6f}")
        print(f"  DV (tok mask) vs HF (with mask): {(dv_out_tok_mask - hf_layer0_out).abs().max().item():.6f}")
        print(f"  DV (no mask) vs HF (no mask): {(dv_out_no_mask - hf_layer0_out_no_mask).abs().max().item():.6f}")
        
        # Check if DV outputs differ based on mask type
        print(f"\n  DV (no mask) vs DV (ones mask): {(dv_out_no_mask - dv_out_ones).abs().max().item():.6f}")
        print(f"  HF (with mask) vs HF (no mask): {(hf_layer0_out - hf_layer0_out_no_mask).abs().max().item():.6f}")
        
        # Now manually trace through HF layer to see what mask they use
        print(f"\n[4] Tracing HF layer 0 attention mask...")
        
        hf_layer0 = hf_model.model.layers[0]
        
        # Get the normed input
        hf_normed = hf_layer0.input_layernorm(hf_hidden)
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        
        print(f"  Layernorm diff: {(hf_normed - dv_normed).abs().max().item():.6f}")
        
        # Check what HF does for attention
        # We need to see how HF creates the causal mask internally
        print(f"\n[5] Checking causal mask generation...")
        
        # DejaVu's causal mask
        dv_causal = _make_causal_mask((bsz, seq_len), dv_hidden.dtype, dv_hidden.device)
        print(f"  DV causal mask shape: {dv_causal.shape}")
        print(f"  DV causal mask [0,0]: {dv_causal[0,0]}")
        
        # What value does DV use for masking?
        min_val = dv_causal.min().item()
        print(f"  DV mask min value: {min_val}")
        
        # Create a 4D mask like HF might use
        # HF typically uses: causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]
        
        print(f"\n[6] Testing if HF applies mask differently...")
        
        # Run DV attention manually with HF's exact computation
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        
        # Q, K, V
        dv_q = dv_block0.self_attn.q_proj(dv_normed)
        dv_k = dv_block0.self_attn.k_proj(dv_normed)
        dv_v = dv_block0.self_attn.v_proj(dv_normed)
        
        dv_q = dv_q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_k = dv_k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dv_v = dv_v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # RoPE
        cos, sin = dv_block0.self_attn.rotary_emb(dv_v, position_ids)
        dv_q, dv_k = apply_rotary_pos_emb(dv_q, dv_k, cos, sin)
        
        # GQA
        dv_k = repeat_kv(dv_k, num_heads // num_kv_heads)
        dv_v = repeat_kv(dv_v, num_heads // num_kv_heads)
        
        # Attention weights
        attn = torch.matmul(dv_q, dv_k.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Try WITHOUT any mask
        attn_no_mask = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        out_no_mask = torch.matmul(attn_no_mask, dv_v)
        
        # Try WITH causal mask
        attn_masked = attn + dv_causal
        attn_with_mask = torch.nn.functional.softmax(attn_masked, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        out_with_mask = torch.matmul(attn_with_mask, dv_v)
        
        print(f"  Attn output (no mask) vs (with mask) diff: {(out_no_mask - out_with_mask).abs().max().item():.6f}")
        
        # Complete the layer
        out_with_mask = out_with_mask.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
        attn_out = dv_block0.self_attn.o_proj(out_with_mask)
        
        hidden_after_attn = dv_hidden + attn_out
        mlp_input = dv_block0.post_attention_layernorm(hidden_after_attn)
        mlp_out = dv_block0.mlp(mlp_input)
        final_out = hidden_after_attn + mlp_out
        
        print(f"\n  Manual computation vs DV layer: {(final_out - dv_out_no_mask).abs().max().item():.6f}")
        print(f"  Manual computation vs HF layer: {(final_out - hf_layer0_out).abs().max().item():.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
