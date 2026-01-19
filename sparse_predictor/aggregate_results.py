


"""
Aggregate Sparse Predictor Results for DejaVu Paper-Style Output

This script parses the training logs and generates a summary table
matching the DejaVu paper's format.

Usage:
    python aggregate_results.py --log-dir ./logs --model llama-3b

Output:
    - CSV file with per-layer metrics
    - Summary statistics
    - Comparison with OPT baseline
"""

import argparse
import os
import re
import json
import csv
from pathlib import Path


def parse_training_log(log_file):
    """Parse a single training log file to extract final metrics."""
    metrics = {
        "Recall": None,
        "True Sparsity": None,
        "Classifier Sparsity": None,
        "Loss": None,
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find the last [Valid] line which has final metrics
    # Format: [Epoch X] [Valid] Loss: 0.1234 Loss Weight: 0.1234 Recall: 0.9500 Classifier Sparsity: 4096.00 True Sparsity: 4000.00
    valid_pattern = r'\[Valid\].*?Recall:\s*([\d.]+).*?Classifier Sparsity:\s*([\d.]+).*?True Sparsity:\s*([\d.]+)'
    matches = re.findall(valid_pattern, content)
    
    if matches:
        last_match = matches[-1]
        metrics["Recall"] = float(last_match[0])
        metrics["Classifier Sparsity"] = float(last_match[1])
        metrics["True Sparsity"] = float(last_match[2])
    
    # Also try to find from checkpoint filename if saved
    # Format: c4_layer0_recall-0.9500-sparsity-4096.pt
    checkpoint_pattern = r'recall-([\d.]+)-sparsity-([\d.]+)'
    ckpt_matches = re.findall(checkpoint_pattern, content)
    if ckpt_matches and metrics["Recall"] is None:
        last_ckpt = ckpt_matches[-1]
        metrics["Recall"] = float(last_ckpt[0])
        metrics["Classifier Sparsity"] = float(last_ckpt[1])
    
    return metrics


def parse_checkpoint_dir(checkpoint_dir):
    """Parse checkpoint filenames to extract metrics."""
    results = {}
    
    for ckpt_file in Path(checkpoint_dir).glob("*.pt"):
        # Format: c4_layer0_recall-0.9500-sparsity-4096.pt
        match = re.search(r'layer(\d+).*?recall-([\d.]+).*?sparsity-([\d.]+)', ckpt_file.name)
        if match:
            layer = int(match.group(1))
            recall = float(match.group(2))
            sparsity = float(match.group(3))
            results[layer] = {
                "Recall": recall,
                "Classifier Sparsity": sparsity,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate sparse predictor results")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory with training logs")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory with checkpoint files")
    parser.add_argument("--model", type=str, default="llama-3b", help="Model name")
    parser.add_argument("--output", type=str, default="results_summary.csv", help="Output CSV file")
    parser.add_argument("--num-layers", type=int, default=28, help="Number of layers")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"DejaVu Sparse Predictor Results - {args.model.upper()}")
    print("=" * 70)
    
    results = {}
    
    # Try to parse from logs
    if os.path.exists(args.log_dir):
        for layer in range(args.num_layers):
            log_file = os.path.join(args.log_dir, f"c4_mlp_out_{layer}.txt")
            if os.path.exists(log_file):
                metrics = parse_training_log(log_file)
                if metrics["Recall"] is not None:
                    results[layer] = metrics
    
    # Try to parse from checkpoints
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        ckpt_results = parse_checkpoint_dir(args.checkpoint_dir)
        for layer, metrics in ckpt_results.items():
            if layer not in results:
                results[layer] = metrics
            else:
                results[layer].update(metrics)
    
    if not results:
        print("\nNo results found. Make sure training has completed.")
        print(f"Looked in: {args.log_dir}")
        if args.checkpoint_dir:
            print(f"Also looked in: {args.checkpoint_dir}")
        return
    
    # Print results table
    print("\n" + "-" * 70)
    print(f"{'Layer':<10} {'Recall':<15} {'True Sparsity':<18} {'Classifier Sparsity':<20}")
    print("-" * 70)
    
    recalls = []
    true_sparsities = []
    classifier_sparsities = []
    
    for layer in sorted(results.keys()):
        r = results[layer]
        recall = r.get("Recall", "N/A")
        true_sp = r.get("True Sparsity", "N/A")
        cls_sp = r.get("Classifier Sparsity", "N/A")
        
        if isinstance(recall, float):
            recalls.append(recall)
            print(f"{layer:<10} {recall:<15.4f} {true_sp if isinstance(true_sp, str) else f'{true_sp:<18.1f}'} {cls_sp if isinstance(cls_sp, str) else f'{cls_sp:<20.1f}'}")
        else:
            print(f"{layer:<10} {recall:<15} {true_sp:<18} {cls_sp:<20}")
        
        if isinstance(true_sp, float):
            true_sparsities.append(true_sp)
        if isinstance(cls_sp, float):
            classifier_sparsities.append(cls_sp)
    
    print("-" * 70)
    
    # Summary statistics
    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        min_recall = min(recalls)
        max_recall = max(recalls)
        
        print(f"\n{'SUMMARY':^70}")
        print("-" * 70)
        print(f"Average Recall:  {avg_recall:.4f}")
        print(f"Min Recall:      {min_recall:.4f}")
        print(f"Max Recall:      {max_recall:.4f}")
        print(f"Layers Trained:  {len(recalls)}/{args.num_layers}")
        
        if true_sparsities:
            print(f"Avg True Sparsity:       {sum(true_sparsities)/len(true_sparsities):.1f}")
        if classifier_sparsities:
            print(f"Avg Classifier Sparsity: {sum(classifier_sparsities)/len(classifier_sparsities):.1f}")
    
    # Comparison with OPT
    print("\n" + "=" * 70)
    print("COMPARISON WITH OPT (DejaVu Paper Baseline)")
    print("=" * 70)
    print("""
OPT-175B (from paper):
  - Recall: ~0.95+ across most layers
  - Predictor successfully predicts active neurons
  - DejaVu achieves 2x speedup

{model}:
  - Average Recall: {avg_recall:.4f}
  - {'✓ Similar to OPT - DejaVu works' if avg_recall > 0.90 else '✗ SIGNIFICANTLY LOWER - DejaVu FAILS'}
""".format(model=args.model.upper(), avg_recall=avg_recall if recalls else 0))
    
    if recalls and avg_recall < 0.90:
        print("""
CONCLUSION:
  The sparse predictor achieves significantly lower recall on Llama compared to OPT.
  This confirms the hypothesis that DejaVu's approach fails on Llama because
  token embeddings change significantly layer-to-layer due to:
  1. Rotary Position Embeddings (RoPE)
  2. SiLU activation (vs ReLU in OPT)
  3. Different architecture causing embedding drift
""")
    
    print("=" * 70)
    
    # Save to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "Recall", "True Sparsity", "Classifier Sparsity"])
        for layer in sorted(results.keys()):
            r = results[layer]
            writer.writerow([
                layer,
                r.get("Recall", ""),
                r.get("True Sparsity", ""),
                r.get("Classifier Sparsity", ""),
            ])
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
