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
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Decentralized_FM_alpha'))


def load_model(model_path, model_type, device):
    """Load the DejaVu model (dense or sparse)."""
    from transformers import LlamaConfig
    
    config = LlamaConfig.from_pretrained(model_path)
    
    if model_type == "llama-sparse":
        from modules.hf_llama_module_sparse import GPTEmbeddings, LlamaSparseBlock, GPTLMHead
        num_layers = config.num_hidden_layers
        
        # Load embeddings
        embeddings = GPTEmbeddings.from_pretrained(model_path, config=config).to(device).half()
        
        # Load layers
        layers = {}
        sparse_path = os.environ.get("SPARSE_PATH", "../checkpoint/llama-3b-sparse-predictor")
        for i in range(num_layers):
            layers[f"block{i}"] = LlamaSparseBlock.from_pretrained(
                model_path, config=config, layer_index=i, sparse_path=sparse_path
            ).to(device).half()
        
        # Load LM head
        lm_head = GPTLMHead.from_pretrained(model_path, config=config).to(device).half()
        
    else:  # Dense
        from modules.hf_llama_module import GPTEmbeddings, LlamaBlock, GPTLMHead
        num_layers = config.num_hidden_layers
        
        # Load embeddings
        embeddings = GPTEmbeddings.from_pretrained(model_path, config=config).to(device).half()
        
        # Load layers
        layers = {}
        for i in range(num_layers):
            layers[f"block{i}"] = LlamaBlock.from_pretrained(
                model_path, config=config, layer_index=i
            ).to(device).half()
        
        # Load LM head
        lm_head = GPTLMHead.from_pretrained(model_path, config=config).to(device).half()
    
    return embeddings, layers, lm_head, config


def run_inference(prompt, tokenizer, embeddings, layers, lm_head, config, device):
    """Run inference on a single prompt and return logprobs."""
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Embeddings
        hidden_states = embeddings(input_ids)
        
        # Forward through layers
        layer_past = None
        previous_emb = None
        for i in range(config.num_hidden_layers):
            layer = layers[f"block{i}"]
            if hasattr(layer, 'forward') and 'previous_emb' in layer.forward.__code__.co_varnames:
                hidden_states, layer_past = layer(hidden_states, layer_past=layer_past, previous_emb=previous_emb)
            else:
                hidden_states, layer_past = layer(hidden_states, layer_past=layer_past)
            previous_emb = hidden_states
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL from generate_task_data.py")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL for evaluate_task_result.py")
    parser.add_argument("--model-path", type=str, default="../checkpoint/llama-3b")
    parser.add_argument("--model-type", type=str, default="llama", choices=["llama", "llama-sparse"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path} (type: {args.model_type})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    embeddings, layers, lm_head, config = load_model(args.model_path, args.model_type, args.device)
    
    # Read input prompts
    prompts = []
    with open(args.input_file, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                prompts.append(item)
    
    print(f"Processing {len(prompts)} prompts...")
    
    # Run inference and write results
    with open(args.output_file, "w") as f:
        for item in tqdm(prompts):
            prompt = item["prompt"]
            
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
    
    print(f"Results written to {args.output_file}")


if __name__ == "__main__":
    main()
