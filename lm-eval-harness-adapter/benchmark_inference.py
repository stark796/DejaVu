"""
Benchmark Inference Script for DejaVu Llama

This script reads prompts from lm-eval-harness and runs DejaVu inference,
outputting logprobs in the format expected by evaluate_task_result.py.

Usage:
    python benchmark_inference.py \
        --input-file mmlu_input.jsonl \
        --output-file mmlu_result.jsonl \
        --model-path ../checkpoint/llama-3b \
        --model-type llama  # or llama-sparse
"""

import argparse
import json
import os
import sys
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))


def load_model(model_path, model_type, device, config_name=None):
    """Load the DejaVu model (dense or sparse).
    
    Args:
        model_path: Local path to DejaVu checkpoint
        model_type: 'llama' or 'llama-sparse'
        device: torch device
        config_name: HuggingFace model name for config (e.g., 'meta-llama/Llama-3.2-3B')
    """
    from transformers import LlamaConfig
    
    # Load config from HuggingFace if config_name provided, else try local
    if config_name:
        config = LlamaConfig.from_pretrained(config_name)
    else:
        config = LlamaConfig.from_pretrained(model_path)
    
    if model_type == "llama-sparse":
        from modules.hf_llama_module_sparse import LlamaEmbeddings, LlamaSparseBlock, LlamaLMHead
        num_layers = config.num_hidden_layers
        
        # Load embeddings
        embeddings = LlamaEmbeddings.from_pretrained(model_path, config=config).to(device).half()
        
        # Load layers
        layers = {}
        sparse_path = os.environ.get("SPARSE_PATH", "../checkpoint/llama-3b-sparse-predictor")
        for i in range(num_layers):
            layers[f"block{i}"] = LlamaSparseBlock.from_pretrained(
                model_path, config=config, layer_index=i, sparse_path=sparse_path
            ).to(device).half()
        
        # Load LM head
        lm_head = LlamaLMHead.from_pretrained(model_path, config=config).to(device).half()
        
    else:  # Dense
        from modules.hf_llama_module import LlamaEmbeddings, LlamaBlock, LlamaLMHead
        num_layers = config.num_hidden_layers
        
        # Load embeddings
        embeddings = LlamaEmbeddings.from_pretrained(model_path, config=config).to(device).half()
        
        # Load layers
        layers = {}
        for i in range(num_layers):
            layers[f"block{i}"] = LlamaBlock.from_pretrained(
                model_path, config=config, layer_index=i
            ).to(device).half()
        
        # Load LM head
        lm_head = LlamaLMHead.from_pretrained(model_path, config=config).to(device).half()
    
    return embeddings, layers, lm_head, config


def run_inference(prompt, tokenizer, embeddings, layers, lm_head, config, device):
    """Run inference on a single prompt and return logprobs."""
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Embeddings
        hidden_states = embeddings(input_ids)
        
        # Forward through layers
        # IMPORTANT: For a fresh forward pass (non-incremental), each layer should get layer_past=None
        # Passing one layer's KV cache to the next layer would corrupt attention!
        for i in range(config.num_hidden_layers):
            layer = layers[f"block{i}"]
            # Note: We discard the returned 'present' since we're not doing incremental generation
            hidden_states, _ = layer(hidden_states, layer_past=None)
        
        # LM head
        logits = lm_head(hidden_states)
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get tokens and their logprobs
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        token_logprobs = [None]  # First token has no logprob
        top_logprobs = [{}]
        
        for i in range(1, len(tokens)):
            token_id = input_ids[0, i].item()
            logprob = log_probs[0, i-1, token_id].item()
            token_logprobs.append(logprob)
            
            # Get top logprob (for greedy correctness check)
            top_token_id = log_probs[0, i-1].argmax().item()
            top_token = tokenizer.convert_ids_to_tokens([top_token_id])[0]
            top_logprob = log_probs[0, i-1, top_token_id].item()
            top_logprobs.append({top_token: top_logprob})
    
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs
    }



def run_generation(prompt, max_tokens, stop, tokenizer, embeddings, layers, lm_head, config, device):
    """Run greedy generation on a single prompt."""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids.clone()
    
    # Cache for efficient generation (if layers supported it, but here we might recompute or need to handle layer_past carefully)
    # The existing run_inference uses a specific layer_past pattern.
    # To be safe and simple (for short generation), we can recompute.
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass - each layer gets layer_past=None for fresh pass
            hidden_states = embeddings(generated_ids)
            for i in range(config.num_hidden_layers):
                layer = layers[f"block{i}"]
                hidden_states, _ = layer(hidden_states, layer_past=None)
            
            logits = lm_head(hidden_states)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check stop conditions
            # 1. EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 2. Stop sequences (if any)
            if stop:
                decoded_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:])
                if isinstance(stop, str):
                    if stop in decoded_text:
                        break
                elif isinstance(stop, list):
                    if any(s in decoded_text for s in stop):
                        break

    # Decode only the new tokens
    new_tokens = generated_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Trim stop sequence if present
    if stop:
        if isinstance(stop, str):
            if stop in generated_text:
                generated_text = generated_text.split(stop)[0]
        elif isinstance(stop, list):
             for s in stop:
                 if s in generated_text:
                     generated_text = generated_text.split(s)[0]
                     
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL from generate_task_data.py")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL for evaluate_task_result.py")
    parser.add_argument("--model-path", type=str, default="../checkpoint/llama-3b")
    parser.add_argument("--tokenizer-name", type=str, default="meta-llama/Llama-3.2-3B", 
                        help="HuggingFace model name for tokenizer (default: meta-llama/Llama-3.2-3B)")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "llama-sparse"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path} (type: {args.model_type})")
    
    # Load tokenizer from HuggingFace (not local path)
    print(f"Loading tokenizer from {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = True
    
    # Load model (use tokenizer_name for config since local path doesn't have config.json)
    embeddings, layers, lm_head, config = load_model(
        args.model_path, args.model_type, args.device, config_name=args.tokenizer_name
    )
    
    # Read input prompts
    prompts = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts.append(item)
    
    # Check for existing results to resume
    start_idx = 0
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                if line.strip():
                    start_idx += 1
        if start_idx > 0:
            print(f"Found {start_idx} existing results in {args.output_file}. Resuming from index {start_idx}...")
    
    prompts = prompts[start_idx:]
    print(f"Processing {len(prompts)} prompts...")
    
    # Run inference and write results
    # Append mode if resuming
    mode = "a" if start_idx > 0 else "w"
    processed = 0
    with open(args.output_file, mode) as f:
        for item in tqdm(prompts):
            prompt = item["prompt"]
            max_tokens = item.get("max_tokens", 0)
            request_type = item.get("request_type", "language-model-inference")
            
            if request_type == "generate_until":
                # Generation request
                stop = item.get("stop", None)
                generated_text = run_generation(prompt, max_tokens, stop, tokenizer, embeddings, layers, lm_head, config, args.device)
                result = {
                    "request": item,
                    "result": [generated_text] # lm-eval expects list of strings for generate_until
                }
            else:
                # Loglikelihood request
                logprobs = run_inference(prompt, tokenizer, embeddings, layers, lm_head, config, args.device)
                result = {
                    "request": item,
                    "result": {
                        "choices": [{
                            "logprobs": logprobs
                        }]
                    }
                }
            
            f.write(json.dumps(result) + "\n")
            processed += 1
            
            # Clear GPU cache every 10 prompts to prevent OOM
            if processed % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
