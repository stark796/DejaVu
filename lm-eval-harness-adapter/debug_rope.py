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
    
    # Load DejaVu layer
    print("Loading DejaVu layer 0...")
    dv_config = LlamaConfig.from_pretrained(tokenizer_name)
    dv_layer0 = LlamaBlock.from_pretrained(model_path, config=dv_config, layer_index=0).to(device).half()
    
    # Get embeddings
    dv_embeddings = LlamaEmbeddings.from_pretrained(model_path, config=dv_config).to(device).half()
    with torch.no_grad():
        dv_hidden = dv_embeddings(input_ids)
    
    print("\n" + "="*60)
    print("EXAMINING DV ROTARY_EMB")
    print("="*60)
    
    dv_rotary = dv_layer0.self_attn.rotary_emb
    
    print(f"DV rotary_emb type: {type(dv_rotary)}")
    print(f"DV rotary_emb config: {dv_rotary.config if hasattr(dv_rotary, 'config') else 'N/A'}")
    
    # Check all attributes
    print(f"\nDV rotary_emb attributes:")
    for attr in dir(dv_rotary):
        if not attr.startswith('_'):
            try:
                val = getattr(dv_rotary, attr)
                if isinstance(val, torch.Tensor):
                    print(f"  {attr}: Tensor shape={val.shape}, device={val.device}, dtype={val.dtype}")
                    if val.numel() < 20:
                        print(f"    values: {val}")
                    else:
                        print(f"    sample: {val.flatten()[:5]}")
                    if torch.isnan(val).any():
                        print(f"    WARNING: Contains NaN!")
                elif not callable(val):
                    print(f"  {attr}: {val}")
            except:
                pass
    
    # Check registered buffers
    print(f"\nDV rotary_emb buffers:")
    for name, buf in dv_rotary.named_buffers():
        print(f"  {name}: shape={buf.shape}, device={buf.device}, dtype={buf.dtype}")
        if torch.isnan(buf).any():
            print(f"    WARNING: Contains NaN!")
    
    # Check registered parameters
    print(f"\nDV rotary_emb parameters:")
    for name, param in dv_rotary.named_parameters():
        print(f"  {name}: shape={param.shape}, device={param.device}, dtype={param.dtype}")
        if torch.isnan(param).any():
            print(f"    WARNING: Contains NaN!")
    
    print("\n" + "="*60)
    print("CREATING FRESH ROTARY_EMB FOR COMPARISON")
    print("="*60)
    
    # Create a fresh LlamaRotaryEmbedding on CUDA with same config
    print(f"\nCreating fresh LlamaRotaryEmbedding with config...")
    print(f"  rope_theta: {dv_config.rope_theta}")
    print(f"  max_position_embeddings: {dv_config.max_position_embeddings}")
    print(f"  head_dim: {dv_config.hidden_size // dv_config.num_attention_heads}")
    
    fresh_rotary = LlamaRotaryEmbedding(config=dv_config).to(device)
    
    print(f"\nFresh rotary_emb buffers after .to(device):")
    for name, buf in fresh_rotary.named_buffers():
        print(f"  {name}: shape={buf.shape}, device={buf.device}, dtype={buf.dtype}")
        if torch.isnan(buf).any():
            print(f"    WARNING: Contains NaN!")
    
    # Now call both and compare
    print("\n" + "="*60)
    print("CALLING ROTARY_EMB")
    print("="*60)
    
    with torch.no_grad():
        # Prepare inputs like in attention
        dv_after_ln1 = dv_layer0.input_layernorm(dv_hidden)
        dv_v = dv_layer0.self_attn.v_proj(dv_after_ln1)
        head_dim = dv_layer0.self_attn.head_dim
        num_kv_heads = dv_layer0.self_attn.num_key_value_heads
        dv_v = dv_v.view(1, 6, num_kv_heads, head_dim).transpose(1, 2)
        
        position_ids = torch.arange(0, 6, dtype=torch.long, device=device).unsqueeze(0)
        
        print(f"v shape: {dv_v.shape}, device: {dv_v.device}, dtype: {dv_v.dtype}")
        print(f"position_ids: {position_ids}")
        
        # Call DV rotary_emb (the one from LlamaBlock)
        print("\nCalling DV layer's rotary_emb...")
        try:
            dv_cos, dv_sin = dv_rotary(dv_v, position_ids)
            print(f"DV cos has nan: {torch.isnan(dv_cos).any()}, shape: {dv_cos.shape}, dtype: {dv_cos.dtype}")
            print(f"DV sin has nan: {torch.isnan(dv_sin).any()}, shape: {dv_sin.shape}, dtype: {dv_sin.dtype}")
            if not torch.isnan(dv_cos).any():
                print(f"DV cos sample: {dv_cos[0, :3, :3]}")
        except Exception as e:
            print(f"DV rotary_emb failed with: {e}")
            import traceback
            traceback.print_exc()
        
        # Call fresh rotary_emb
        print("\nCalling fresh rotary_emb...")
        try:
            fresh_cos, fresh_sin = fresh_rotary(dv_v.float(), position_ids)  # Use float for comparison
            print(f"Fresh cos has nan: {torch.isnan(fresh_cos).any()}, shape: {fresh_cos.shape}, dtype: {fresh_cos.dtype}")
            print(f"Fresh sin has nan: {torch.isnan(fresh_sin).any()}, shape: {fresh_sin.shape}, dtype: {fresh_sin.dtype}")
            if not torch.isnan(fresh_cos).any():
                print(f"Fresh cos sample: {fresh_cos[0, :3, :3]}")
        except Exception as e:
            print(f"Fresh rotary_emb failed with: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
