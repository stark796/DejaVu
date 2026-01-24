#!/usr/bin/env python3
"""
Debug RoPE to find the NaN source.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))

def main():
    model_path = "../Decentralized_FM_alpha/pretrained_models/llama-3.2-3b"
    tokenizer_name = "meta-llama/Llama-3.2-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("DEBUGGING ROPE EMBEDDINGS")
    print("="*60)
    
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    input_ids = tokenizer.encode("The capital of France is", return_tensors="pt").to(device)
    
    # Load HF model
    print("\nLoading HuggingFace model...")
    hf_model = LlamaForCausalLM.from_pretrained(
        tokenizer_name, torch_dtype=torch.float16, device_map=device
    )
    
    # Load DejaVu layer
    print("Loading DejaVu layer 0...")
    dv_config = LlamaConfig.from_pretrained(tokenizer_name)
    dv_layer0 = LlamaBlock.from_pretrained(model_path, config=dv_config, layer_index=0).to(device).half()
    
    # Get embeddings
    with torch.no_grad():
        hf_hidden = hf_model.model.embed_tokens(input_ids)
        dv_embeddings = LlamaEmbeddings.from_pretrained(model_path, config=dv_config).to(device).half()
        dv_hidden = dv_embeddings(input_ids)
    
    print("\n" + "="*60)
    print("COMPARING ROPE EMBEDDINGS")
    print("="*60)
    
    hf_layer0 = hf_model.model.layers[0]
    
    # Get HF rotary_emb info
    print(f"\nHF rotary_emb type: {type(hf_layer0.self_attn.rotary_emb)}")
    print(f"HF rotary_emb device: {next(hf_layer0.self_attn.rotary_emb.parameters(), torch.tensor([0])).device if list(hf_layer0.self_attn.rotary_emb.parameters()) else 'no parameters'}")
    if hasattr(hf_layer0.self_attn.rotary_emb, 'inv_freq'):
        print(f"HF inv_freq shape: {hf_layer0.self_attn.rotary_emb.inv_freq.shape if hf_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"HF inv_freq device: {hf_layer0.self_attn.rotary_emb.inv_freq.device if hf_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"HF inv_freq dtype: {hf_layer0.self_attn.rotary_emb.inv_freq.dtype if hf_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"HF inv_freq sample: {hf_layer0.self_attn.rotary_emb.inv_freq[:5] if hf_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
    
    print(f"\nDV rotary_emb type: {type(dv_layer0.self_attn.rotary_emb)}")
    if hasattr(dv_layer0.self_attn.rotary_emb, 'inv_freq'):
        print(f"DV inv_freq shape: {dv_layer0.self_attn.rotary_emb.inv_freq.shape if dv_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"DV inv_freq device: {dv_layer0.self_attn.rotary_emb.inv_freq.device if dv_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"DV inv_freq dtype: {dv_layer0.self_attn.rotary_emb.inv_freq.dtype if dv_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
        print(f"DV inv_freq sample: {dv_layer0.self_attn.rotary_emb.inv_freq[:5] if dv_layer0.self_attn.rotary_emb.inv_freq is not None else None}")
    else:
        print("DV rotary_emb has no inv_freq attribute!")
    
    # Check rotary_emb config
    print(f"\nDV rotary_emb config attributes:")
    for attr in ['dim', 'max_position_embeddings', 'base', 'rope_type']:
        if hasattr(dv_layer0.self_attn.rotary_emb, attr):
            print(f"  {attr}: {getattr(dv_layer0.self_attn.rotary_emb, attr)}")
    
    # Now call rotary_emb manually
    print("\n" + "="*60)
    print("CALLING ROTARY_EMB MANUALLY")
    print("="*60)
    
    with torch.no_grad():
        # Prepare inputs like in attention
        hf_after_ln1 = hf_layer0.input_layernorm(hf_hidden)
        v = hf_layer0.self_attn.v_proj(hf_after_ln1)
        v = v.view(1, 6, hf_layer0.self_attn.num_key_value_heads, hf_layer0.self_attn.head_dim).transpose(1, 2)
        
        position_ids = torch.arange(0, 6, dtype=torch.long, device=device).unsqueeze(0)
        
        print(f"v shape: {v.shape}, device: {v.device}, dtype: {v.dtype}")
        print(f"position_ids: {position_ids}")
        
        # Call HF rotary_emb
        print("\nCalling HF rotary_emb...")
        hf_cos, hf_sin = hf_layer0.self_attn.rotary_emb(v, position_ids)
        print(f"HF cos has nan: {torch.isnan(hf_cos).any()}, shape: {hf_cos.shape}")
        print(f"HF sin has nan: {torch.isnan(hf_sin).any()}, shape: {hf_sin.shape}")
        print(f"HF cos sample: {hf_cos[0, :5, 0]}")
        
        # Call DV rotary_emb
        print("\nCalling DV rotary_emb...")
        dv_after_ln1 = dv_layer0.input_layernorm(dv_hidden)
        dv_v = dv_layer0.self_attn.v_proj(dv_after_ln1)
        dv_v = dv_v.view(1, 6, dv_layer0.self_attn.num_key_value_heads, dv_layer0.self_attn.head_dim).transpose(1, 2)
        
        try:
            dv_cos, dv_sin = dv_layer0.self_attn.rotary_emb(dv_v, position_ids)
            print(f"DV cos has nan: {torch.isnan(dv_cos).any()}, shape: {dv_cos.shape}")
            print(f"DV sin has nan: {torch.isnan(dv_sin).any()}, shape: {dv_sin.shape}")
            print(f"DV cos sample: {dv_cos[0, :5, 0]}")
            
            print(f"\ncos diff: {(hf_cos - dv_cos).abs().max().item()}")
            print(f"sin diff: {(hf_sin - dv_sin).abs().max().item()}")
        except Exception as e:
            print(f"DV rotary_emb failed with: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
