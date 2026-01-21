#!/usr/bin/env python
"""
Detailed comparison of attention computation step-by-step.
Finds exactly where the 0.167 diff originates.
"""

import torch
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("Step-by-step Attention Comparison")
    print("=" * 60)
    
    model_path = "./pretrained_models/llama-3.2-3b"
    
    # Load both models
    print("\n[1] Loading models...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock
    
    # HuggingFace
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    
    # DejaVu
    config = LlamaConfig.from_pretrained(model_path)
    dv_emb = LlamaEmbeddings.from_pretrained(model_path).cuda().half()
    dv_block0 = LlamaBlock.from_pretrained(model_path, layer_index=0).cuda().half()
    
    hf_layer0 = hf_model.model.layers[0]
    
    # Prepare input
    print("\n[2] Preparing input...")
    text = "The cat sat on the"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    seq_len = input_ids.size(1)
    print(f"  Input: '{text}', {seq_len} tokens")
    
    with torch.no_grad():
        dv_hidden = dv_emb(input_ids)
        hf_hidden = hf_model.model.embed_tokens(input_ids)
    
    print(f"  Embedding diff: {(dv_hidden - hf_hidden).abs().max().item():.6f}")
    
    # Step-by-step attention comparison
    print("\n[3] Step-by-step attention comparison...")
    
    with torch.no_grad():
        # Layernorm
        dv_normed = dv_block0.input_layernorm(dv_hidden)
        hf_normed = hf_layer0.input_layernorm(hf_hidden)
        print(f"  Layernorm diff: {(dv_normed - hf_normed).abs().max().item():.6f}")
        
        # Q, K, V
        dv_q = dv_block0.self_attn.q_proj(dv_normed)
        dv_k = dv_block0.self_attn.k_proj(dv_normed)
        dv_v = dv_block0.self_attn.v_proj(dv_normed)
        
        hf_q = hf_layer0.self_attn.q_proj(hf_normed)
        hf_k = hf_layer0.self_attn.k_proj(hf_normed)
        hf_v = hf_layer0.self_attn.v_proj(hf_normed)
        
        print(f"  Q proj diff: {(dv_q - hf_q).abs().max().item():.6f}")
        print(f"  K proj diff: {(dv_k - hf_k).abs().max().item():.6f}")
        print(f"  V proj diff: {(dv_v - hf_v).abs().max().item():.6f}")
        
        # Reshape
        bsz = 1
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // num_heads
        
        dv_q = dv_q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_k = dv_k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        dv_v = dv_v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        hf_q = hf_q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        hf_k = hf_k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        hf_v = hf_v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # RoPE
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        
        # DejaVu RoPE
        dv_cos, dv_sin = dv_block0.self_attn.rotary_emb(dv_v, position_ids)
        
        # HF RoPE - need to access through the model's rotary embedding
        # HF stores rotary embeddings differently depending on version
        try:
            hf_rotary = hf_layer0.self_attn.rotary_emb
            hf_cos, hf_sin = hf_rotary(hf_v, position_ids)
        except:
            hf_cos, hf_sin = dv_cos, dv_sin  # Use DV's if HF API differs
        
        print(f"  cos diff: {(dv_cos - hf_cos).abs().max().item():.6f}")
        print(f"  sin diff: {(dv_sin - hf_sin).abs().max().item():.6f}")
        
        # Apply RoPE
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        
        dv_q_rot, dv_k_rot = apply_rotary_pos_emb(dv_q, dv_k, dv_cos, dv_sin)
        hf_q_rot, hf_k_rot = apply_rotary_pos_emb(hf_q, hf_k, hf_cos, hf_sin)
        
        print(f"  Q after RoPE diff: {(dv_q_rot - hf_q_rot).abs().max().item():.6f}")
        print(f"  K after RoPE diff: {(dv_k_rot - hf_k_rot).abs().max().item():.6f}")
        
        # repeat_kv for GQA
        num_kv_groups = num_heads // num_kv_heads
        
        dv_k_exp = repeat_kv(dv_k_rot, num_kv_groups)
        dv_v_exp = repeat_kv(dv_v, num_kv_groups)
        
        hf_k_exp = repeat_kv(hf_k_rot, num_kv_groups)
        hf_v_exp = repeat_kv(hf_v, num_kv_groups)
        
        print(f"  K after repeat_kv diff: {(dv_k_exp - hf_k_exp).abs().max().item():.6f}")
        print(f"  V after repeat_kv diff: {(dv_v_exp - hf_v_exp).abs().max().item():.6f}")
        
        # Attention weights before mask
        dv_attn = torch.matmul(dv_q_rot, dv_k_exp.transpose(2, 3)) / math.sqrt(head_dim)
        hf_attn = torch.matmul(hf_q_rot, hf_k_exp.transpose(2, 3)) / math.sqrt(head_dim)
        
        print(f"  Attn weights (pre-mask) diff: {(dv_attn - hf_attn).abs().max().item():.6f}")
        
        # Create causal mask (DejaVu style)
        from modules.hf_llama_module import _make_causal_mask
        dv_mask = _make_causal_mask((bsz, seq_len), dv_hidden.dtype, dv_hidden.device)
        
        print(f"  DV mask shape: {dv_mask.shape}")
        
        # Apply mask
        dv_attn_masked = dv_attn + dv_mask
        # For HF, we apply same mask since we're testing parity
        hf_attn_masked = hf_attn + dv_mask
        
        print(f"  Attn weights (post-mask) diff: {(dv_attn_masked - hf_attn_masked).abs().max().item():.6f}")
        
        # Softmax
        dv_attn_probs = torch.nn.functional.softmax(dv_attn_masked, dim=-1, dtype=torch.float32).to(dv_hidden.dtype)
        hf_attn_probs = torch.nn.functional.softmax(hf_attn_masked, dim=-1, dtype=torch.float32).to(hf_hidden.dtype)
        
        print(f"  Attn probs (after softmax) diff: {(dv_attn_probs - hf_attn_probs).abs().max().item():.6f}")
        
        # Attention output
        dv_attn_out = torch.matmul(dv_attn_probs, dv_v_exp)
        hf_attn_out = torch.matmul(hf_attn_probs, hf_v_exp)
        
        print(f"  Attn output (before reshape) diff: {(dv_attn_out - hf_attn_out).abs().max().item():.6f}")
        
        # Reshape and output projection
        dv_attn_out = dv_attn_out.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_heads * head_dim)
        hf_attn_out = hf_attn_out.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_heads * head_dim)
        
        dv_o_out = dv_block0.self_attn.o_proj(dv_attn_out)
        hf_o_out = hf_layer0.self_attn.o_proj(hf_attn_out)
        
        print(f"  O projection diff: {(dv_o_out - hf_o_out).abs().max().item():.6f}")
        
        # Now compare with what the actual layer produces
        print("\n[4] Comparing actual layer outputs...")
        
        # Run DV layer
        dv_layer_out, _ = dv_block0(dv_hidden, layer_past=None, mask=None, position_ids=position_ids)
        
        # For HF, we need to figure out the right API
        # Let's use the full model
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_layer_out = hf_out.hidden_states[1]  # After layer 0
        
        layer_diff = (dv_layer_out - hf_layer_out).abs().max().item()
        print(f"  Actual layer 0 output diff: {layer_diff:.6f}")
        
        if layer_diff > 0.01:
            print("\n  *** Difference found! Comparing DV internal vs expected ***")
            # The difference is between what DV layer produces vs what it should produce
            # (based on our step-by-step computation matching HF)
            
            # What we computed step-by-step (should match HF)
            expected_out = dv_hidden + dv_o_out
            expected_mlp_in = dv_block0.post_attention_layernorm(expected_out)
            expected_mlp_out = dv_block0.mlp(expected_mlp_in)
            expected_layer_out = expected_out + expected_mlp_out
            
            print(f"  Expected (step-by-step) vs actual DV: {(expected_layer_out - dv_layer_out).abs().max().item():.6f}")
            print(f"  Expected (step-by-step) vs actual HF: {(expected_layer_out - hf_layer_out).abs().max().item():.6f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
