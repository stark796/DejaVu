#!/usr/bin/env python3
"""
Compare actual weight VALUES between HuggingFace and DejaVu models.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

def main():
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    tokenizer_name = "meta-llama/Llama-3.2-3B"
    device = "cpu"  # Use CPU so we can compare values precisely
    
    print("="*60)
    print("WEIGHT COMPARISON: HuggingFace vs DejaVu")
    print("="*60)
    
    from transformers import LlamaForCausalLM, LlamaConfig
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    # Load HuggingFace model
    print("\nLoading HuggingFace model...")
    hf_model = LlamaForCausalLM.from_pretrained(
        tokenizer_name,
        torch_dtype=torch.float32,  # Use full precision for comparison
        device_map="cpu"
    )
    
    # Load DejaVu model
    print("Loading DejaVu model...")
    dv_config = LlamaConfig.from_pretrained(tokenizer_name)
    dv_embeddings = LlamaEmbeddings.from_pretrained(model_path, config=dv_config).float()
    dv_layer0 = LlamaBlock.from_pretrained(model_path, config=dv_config, layer_index=0).float()
    dv_lm_head = LlamaLMHead.from_pretrained(model_path, config=dv_config).float()
    
    print("\n" + "="*60)
    print("COMPARING EMBEDDING WEIGHTS")
    print("="*60)
    hf_emb_weight = hf_model.model.embed_tokens.weight
    dv_emb_weight = dv_embeddings.embed_tokens.weight
    diff = (hf_emb_weight - dv_emb_weight).abs().max().item()
    print(f"Embedding weight max diff: {diff}")
    
    print("\n" + "="*60)
    print("COMPARING LAYER 0 WEIGHTS")
    print("="*60)
    
    hf_layer0 = hf_model.model.layers[0]
    
    # Compare layernorm weights
    hf_ln1 = hf_layer0.input_layernorm.weight
    dv_ln1 = dv_layer0.input_layernorm.weight
    diff = (hf_ln1 - dv_ln1).abs().max().item()
    print(f"input_layernorm.weight max diff: {diff}")
    
    hf_ln2 = hf_layer0.post_attention_layernorm.weight
    dv_ln2 = dv_layer0.post_attention_layernorm.weight
    diff = (hf_ln2 - dv_ln2).abs().max().item()
    print(f"post_attention_layernorm.weight max diff: {diff}")
    
    # Compare attention weights
    hf_q = hf_layer0.self_attn.q_proj.weight
    dv_q = dv_layer0.self_attn.q_proj.weight
    diff = (hf_q - dv_q).abs().max().item()
    print(f"self_attn.q_proj.weight max diff: {diff}")
    
    hf_k = hf_layer0.self_attn.k_proj.weight
    dv_k = dv_layer0.self_attn.k_proj.weight
    diff = (hf_k - dv_k).abs().max().item()
    print(f"self_attn.k_proj.weight max diff: {diff}")
    
    hf_v = hf_layer0.self_attn.v_proj.weight
    dv_v = dv_layer0.self_attn.v_proj.weight
    diff = (hf_v - dv_v).abs().max().item()
    print(f"self_attn.v_proj.weight max diff: {diff}")
    
    hf_o = hf_layer0.self_attn.o_proj.weight
    dv_o = dv_layer0.self_attn.o_proj.weight
    diff = (hf_o - dv_o).abs().max().item()
    print(f"self_attn.o_proj.weight max diff: {diff}")
    
    # Compare MLP weights
    hf_gate = hf_layer0.mlp.gate_proj.weight
    dv_gate = dv_layer0.mlp.gate_proj.weight
    diff = (hf_gate - dv_gate).abs().max().item()
    print(f"mlp.gate_proj.weight max diff: {diff}")
    
    hf_up = hf_layer0.mlp.up_proj.weight
    dv_up = dv_layer0.mlp.up_proj.weight
    diff = (hf_up - dv_up).abs().max().item()
    print(f"mlp.up_proj.weight max diff: {diff}")
    
    hf_down = hf_layer0.mlp.down_proj.weight
    dv_down = dv_layer0.mlp.down_proj.weight
    diff = (hf_down - dv_down).abs().max().item()
    print(f"mlp.down_proj.weight max diff: {diff}")
    
    print("\n" + "="*60)
    print("COMPARING LM HEAD WEIGHTS")
    print("="*60)
    
    hf_lm_head_weight = hf_model.lm_head.weight
    dv_lm_head_weight = dv_lm_head.lm_head.weight
    diff = (hf_lm_head_weight - dv_lm_head_weight).abs().max().item()
    print(f"lm_head.weight max diff: {diff}")
    
    hf_final_norm = hf_model.model.norm.weight
    dv_final_norm = dv_lm_head.norm.weight
    diff = (hf_final_norm - dv_final_norm).abs().max().item()
    print(f"final_norm.weight max diff: {diff}")
    
    print("\n" + "="*60)
    print("If any weight shows large diff (>0.01), that's the bug!")
    print("="*60)


if __name__ == "__main__":
    main()
