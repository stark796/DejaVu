"""
Embedding Drift Measurement for DejaVu Hypothesis Testing

This script measures how much token embeddings change layer-to-layer in Llama 3.2 3B.

KEY HYPOTHESIS:
- DejaVu assumes embeddings are stable across layers (works for OPT)
- Llama's RoPE and architecture cause significant embedding drift
- If cosine similarity drops significantly, DejaVu's predictor cannot work

WHAT THIS PROVES:
- "The token embeddings (vectors) change significantly from layer to layer 
   and thus the predictor can't be used for layer downstream."

Usage:
    python measure_embedding_drift.py --model-path ./pretrained_models/llama-3.2-3b

Output:
    - Layer-by-layer cosine similarity table
    - Comparison with expected OPT behavior
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import os


def compute_cosine_similarity(x, y):
    """Compute mean cosine similarity between two tensors."""
    # Flatten to (batch*seq, hidden)
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    
    # Normalize
    x_norm = F.normalize(x_flat, p=2, dim=-1)
    y_norm = F.normalize(y_flat, p=2, dim=-1)
    
    # Cosine similarity per position
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    
    return cos_sim.mean().item(), cos_sim.std().item()


def compute_l2_distance(x, y):
    """Compute mean L2 distance between two tensors."""
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    
    l2_dist = torch.norm(x_flat - y_flat, p=2, dim=-1)
    return l2_dist.mean().item(), l2_dist.std().item()


def measure_embedding_drift(model_path, test_texts=None, device="cuda"):
    """
    Measure embedding drift across layers in Llama.
    
    Returns:
        dict: Results containing cosine similarities and L2 distances
    """
    print("=" * 70)
    print("EMBEDDING DRIFT MEASUREMENT - DejaVu Hypothesis Testing")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(model_path)
    
    # Load with output_hidden_states=True to get all layer outputs
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    
    # Default test texts if none provided
    if test_texts is None:
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, transformers have revolutionized natural language processing.",
            "DejaVu proposes contextual sparsity for efficient LLM inference.",
            "The capital of France is Paris, which is known for the Eiffel Tower.",
            "Artificial intelligence is transforming how we interact with technology.",
        ]
    
    print(f"\nProcessing {len(test_texts)} test sequences...")
    
    all_results = []
    
    for text_idx, text in enumerate(test_texts):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # hidden_states is tuple of (num_layers + 1) tensors
        # Index 0 is embedding output, indices 1-N are layer outputs
        hidden_states = outputs.hidden_states
        
        # Store results for this text
        text_results = {
            "text": text[:50] + "..." if len(text) > 50 else text,
            "num_tokens": inputs["input_ids"].shape[1],
            "consecutive_cosine": [],  # Similarity between layer i and i+1
            "from_layer0_cosine": [],  # Similarity between layer 0 and layer i
            "consecutive_l2": [],
            "from_layer0_l2": [],
        }
        
        # Layer 0 hidden state (after embedding, before any transformer block)
        layer0_hidden = hidden_states[0]
        
        for layer_idx in range(len(hidden_states) - 1):
            current_hidden = hidden_states[layer_idx]
            next_hidden = hidden_states[layer_idx + 1]
            
            # Consecutive layer similarity
            cos_mean, cos_std = compute_cosine_similarity(current_hidden, next_hidden)
            l2_mean, l2_std = compute_l2_distance(current_hidden, next_hidden)
            
            text_results["consecutive_cosine"].append((cos_mean, cos_std))
            text_results["consecutive_l2"].append((l2_mean, l2_std))
            
            # Similarity from layer 0
            if layer_idx > 0:
                cos_mean_0, cos_std_0 = compute_cosine_similarity(layer0_hidden, hidden_states[layer_idx])
                l2_mean_0, l2_std_0 = compute_l2_distance(layer0_hidden, hidden_states[layer_idx])
                text_results["from_layer0_cosine"].append((cos_mean_0, cos_std_0))
                text_results["from_layer0_l2"].append((l2_mean_0, l2_std_0))
        
        all_results.append(text_results)
    
    # Aggregate results across all texts
    num_layers = config.num_hidden_layers
    
    print("\n" + "=" * 70)
    print("RESULTS: Consecutive Layer Cosine Similarity")
    print("=" * 70)
    print("(Lower values = more embedding drift = DejaVu predictor fails)")
    print("-" * 70)
    print(f"{'Layer Transition':<20} {'Mean Cosine Sim':<18} {'Std Dev':<12} {'Interpretation'}")
    print("-" * 70)
    
    avg_consecutive = []
    for layer_idx in range(num_layers):
        cos_values = [r["consecutive_cosine"][layer_idx][0] for r in all_results]
        avg_cos = np.mean(cos_values)
        std_cos = np.std(cos_values)
        avg_consecutive.append(avg_cos)
        
        # Interpretation
        if avg_cos > 0.95:
            interp = "✓ Stable"
        elif avg_cos > 0.85:
            interp = "~ Moderate drift"
        elif avg_cos > 0.70:
            interp = "⚠ Significant drift"
        else:
            interp = "✗ Severe drift"
        
        print(f"Layer {layer_idx:2d} → {layer_idx+1:2d}      {avg_cos:.4f}            {std_cos:.4f}       {interp}")
    
    print("-" * 70)
    print(f"{'AVERAGE':<20} {np.mean(avg_consecutive):.4f}")
    print()
    
    # Similarity from layer 0 (cumulative drift)
    print("\n" + "=" * 70)
    print("RESULTS: Similarity to Layer 0 (Cumulative Drift)")
    print("=" * 70)
    print("(Shows how far embeddings have drifted from initial representation)")
    print("-" * 70)
    print(f"{'Layer':<15} {'Cosine Sim to Layer 0':<25} {'Interpretation'}")
    print("-" * 70)
    
    avg_from_0 = []
    for layer_idx in range(1, num_layers):
        cos_values = [r["from_layer0_cosine"][layer_idx-1][0] for r in all_results]
        avg_cos = np.mean(cos_values)
        avg_from_0.append(avg_cos)
        
        if avg_cos > 0.90:
            interp = "✓ Close to original"
        elif avg_cos > 0.70:
            interp = "~ Moderate drift"
        elif avg_cos > 0.50:
            interp = "⚠ Significant drift"
        else:
            interp = "✗ Very different"
        
        print(f"Layer {layer_idx:2d}        {avg_cos:.4f}                    {interp}")
    
    print("-" * 70)
    
    # Final analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: DejaVu Hypothesis Verification")
    print("=" * 70)
    
    final_layer_sim = avg_from_0[-1] if avg_from_0 else 0
    avg_consec_sim = np.mean(avg_consecutive)
    
    print(f"""
Model: Llama 3.2 3B ({num_layers} layers)

Key Metrics:
  - Average consecutive layer similarity: {avg_consec_sim:.4f}
  - Final layer similarity to layer 0:   {final_layer_sim:.4f}

DejaVu Assumption:
  - Assumes embeddings are stable layer-to-layer
  - Predictor at layer 0 should predict activity at layer N
  - This works for OPT (with ~0.95+ consecutive similarity)

Llama Results:
  - Consecutive similarity: {avg_consec_sim:.4f}
  - Cumulative drift by final layer: {1 - final_layer_sim:.2%} from original
""")
    
    if final_layer_sim < 0.70 or avg_consec_sim < 0.90:
        print("""
CONCLUSION: ✗ DejaVu FAILS on Llama
  - Embeddings change significantly layer-to-layer
  - A predictor trained on layer 0 input CANNOT accurately predict
    which neurons/heads will be active in downstream layers
  - The fundamental assumption of DejaVu is violated
""")
        hypothesis_verified = True
    else:
        print("""
CONCLUSION: Results inconclusive
  - Embeddings are relatively stable
  - Further investigation needed
""")
        hypothesis_verified = False
    
    print("=" * 70)
    
    # Save results
    results = {
        "model": model_path,
        "num_layers": num_layers,
        "hidden_size": config.hidden_size,
        "avg_consecutive_cosine": avg_consec_sim,
        "final_layer_to_layer0_cosine": final_layer_sim,
        "consecutive_cosine_per_layer": avg_consecutive,
        "from_layer0_cosine_per_layer": [0.0] + avg_from_0,  # Include layer 0 as 1.0
        "hypothesis_verified": hypothesis_verified,
        "interpretation": "DejaVu fails on Llama" if hypothesis_verified else "Inconclusive",
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Measure embedding drift in Llama for DejaVu hypothesis testing")
    parser.add_argument("--model-path", type=str, default="./pretrained_models/llama-3.2-3b",
                        help="Path to converted Llama checkpoint")
    parser.add_argument("--output-file", type=str, default="./embedding_drift_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()
    
    # Run measurement
    results = measure_embedding_drift(args.model_path, device=args.device)
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
