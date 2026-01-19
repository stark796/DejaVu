"""
Cross-Layer Sparse Predictor Training for Llama 3.2 3B

This script trains predictors that use layer 0 input to predict layer N sparsity.
This is the TRUE test of embedding drift - if Llama embeddings change significantly
across layers, cross-layer prediction will be much worse than same-layer.

Usage:
    # Test layer 0 → layer 14 (baseline should be similar to same-layer)
    python main_mlp_llama_crosslayer.py --target-layer 14

    # Test layer 0 → layer 27 (maximum drift if it exists)
    python main_mlp_llama_crosslayer.py --target-layer 27
"""

import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset, DataLoader
from trainer_mlp import train

# Data paths
DATA_PATH = "../Decentralized_FM_alpha/data/llama_3b_c4"

# Llama 3.2 3B config
CONFIG = {
    'num_layer': 28,
    'd': 3072,              # hidden_size
    'intermediate': 8192,   # intermediate_size
    'N': 80000,             # samples to use
}


class CrossLayerDataset(Dataset):
    """Dataset that pairs layer 0 input with target layer labels."""
    
    def __init__(self, query, labels, n, is_train):
        self.query = query    # From layer 0
        self.labels = labels  # From target layer
        self.n = n
        self.is_train = is_train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.is_train:
            x = torch.Tensor(self.query[idx])
            y = torch.Tensor(self.labels[idx])
        else:
            x = torch.Tensor(self.query[-idx-1])
            y = torch.Tensor(self.labels[-idx-1])
        return x, y


def get_crosslayer_data(target_layer):
    """Load layer 0 input and target layer labels."""
    
    # Query: ALWAYS from layer 0
    query_path = f"{DATA_PATH}/mlp_x_0.mmap"
    print(f"Loading QUERY from layer 0: {query_path}")
    query = np.array(np.memmap(
        query_path, dtype='float16', mode='r',
        shape=(CONFIG['N'], CONFIG['d'])
    ))
    
    # Labels: from target layer
    label_path = f"{DATA_PATH}/mlp_label_{target_layer}.mmap"
    print(f"Loading LABELS from layer {target_layer}: {label_path}")
    labels = np.array(np.memmap(
        label_path, dtype='float16', mode='r',
        shape=(CONFIG['N'], CONFIG['intermediate'])
    ))
    
    return query, labels


def generate_label(y, threshold=0.0):
    """Generate binary labels from MLP activations."""
    return (torch.abs(y) > threshold).float()


def evaluate(model, device, loader, threshold=0.0):
    """Evaluate cross-layer predictor."""
    model.eval()
    
    metrics = {
        "Recall": [],
        "True Sparsity": [],
        "Classifier Sparsity": [],
        "Loss": [],
    }
    
    with torch.no_grad():
        for x, y in loader:
            y = y.float().to(device)
            y = generate_label(y, threshold)
            
            logits = model(x.to(device))
            probs = logits.sigmoid()
            preds = probs >= 0.5
            
            # Calculate recall
            miss = (y.int() - preds.int()) > 0
            recall = (y.sum(dim=1) - miss.sum(dim=1)) / (y.sum(dim=1) + 1e-8)
            
            metrics["Recall"].append(recall.mean().item())
            metrics["True Sparsity"].append(y.sum(dim=1).mean().item())
            metrics["Classifier Sparsity"].append(preds.sum(dim=1).float().mean().item())
            
            # Loss
            weight = (y.sum() / y.numel()) + 0.005
            loss_weight = y * (1 - weight) + weight
            loss = torch.nn.functional.binary_cross_entropy(probs, y, weight=loss_weight)
            metrics["Loss"].append(loss.item())
    
    return {k: np.mean(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Cross-Layer Sparse Predictor")
    parser.add_argument("--target-layer", type=int, required=True,
                       help="Target layer to predict (0-27)")
    parser.add_argument("--D", type=int, default=1000, help="Low-rank dimension")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(f"CROSS-LAYER PREDICTION: Layer 0 → Layer {args.target_layer}")
    print("=" * 60)
    
    # Load data
    query, labels = get_crosslayer_data(args.target_layer)
    
    # Create datasets
    total = len(query)
    num_train = int(0.95 * total)
    num_test = total - num_train
    
    print(f"Query shape: {query.shape} (from layer 0)")
    print(f"Label shape: {labels.shape} (from layer {args.target_layer})")
    print(f"Train: {num_train}, Test: {num_test}")
    
    train_ds = CrossLayerDataset(query, labels, num_train, True)
    test_ds = CrossLayerDataset(query, labels, num_test, False)
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False)
    
    # Create predictor
    predictor = torch.nn.Sequential(
        torch.nn.Linear(CONFIG['d'], args.D, bias=None),
        torch.nn.Linear(args.D, CONFIG['intermediate'], bias=None),
    )
    
    print(f"\nPredictor: {CONFIG['d']} → {args.D} → {CONFIG['intermediate']}")
    print("\nTraining...")
    
    best_model, eval_result = train(
        predictor, train_loader, test_loader, args, device, verbal=True
    )
    
    # Final results
    print("\n" + "=" * 60)
    print("CROSS-LAYER PREDICTION RESULTS")
    print("=" * 60)
    print(f"Query Layer: 0")
    print(f"Target Layer: {args.target_layer}")
    print(f"Recall: {eval_result['Recall']:.4f}")
    print(f"True Sparsity: {eval_result['True Sparsity']:.0f}")
    print(f"Classifier Sparsity: {eval_result['Classifier Sparsity']:.0f}")
    print("=" * 60)
    print("\nCOMPARE WITH SAME-LAYER RESULTS:")
    print(f"  Same-layer (L{args.target_layer}→L{args.target_layer}): ~97-98% recall")
    print(f"  Cross-layer (L0→L{args.target_layer}): {eval_result['Recall']*100:.2f}% recall")
    print(f"  Difference: Shows embedding drift impact!")
    print("=" * 60)


if __name__ == "__main__":
    main()
