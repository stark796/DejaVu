"""
Sparse Predictor Training for Llama 3.2 3B

This script trains the MLP sparse predictor for Llama 3.2 3B.

KEY DIFFERENCES FROM OPT:
1. Llama uses SiLU (not ReLU), so sparsity pattern is different
2. Llama has gated MLP: gate_proj and up_proj, so intermediate_size is the label dimension
3. We expect WORSE prediction accuracy due to embedding drift in Llama

IMPORTANT FOR DEJAVU HYPOTHESIS:
- If predictor accuracy is significantly lower than OPT, it proves the hypothesis
- OPT: embeddings stable layer-to-layer → good prediction
- Llama: embeddings change significantly → poor prediction

Usage:
    python main_mlp_llama.py --model llama-3b --dataset c4 --L 0
    
    For all layers:
    ./run_c4_mlp_llama.sh
"""

import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_mlp import train

# Data paths for Llama models
DATA = {
    "llama-3b": {
        "c4": "../Decentralized_FM_alpha/data/llama_3b_c4",
    },
}

MODEL_CHOICES = ['llama-3b']
DATA_CHOICES = ['c4']

# Configuration for Llama 3.2 3B
# Note: These values are for Llama 3.2 3B specifically
CONFIG = {
    'llama-3b': {
        'num_layer': 28,           # 28 layers
        'ckt_storage': "bylayer",
        'd': 3072,                 # hidden_size
        'intermediate': 8192,       # intermediate_size (for MLP)
        'h': 24,                   # num_attention_heads
        'kv_h': 8,                 # num_key_value_heads (GQA)
        'N': 80000,               # number of samples (reduced to avoid edge of collected data)
    },
}


class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train):
        self.X = X
        self.Y = Y
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        
        # Check for degenerate case
        if y.sum() == 0:
            print("Warning: all zero y encountered")
        return x, y


def get_data(args, layer_idx):
    """Load training data for a specific layer."""
    config = CONFIG[args.model]
    

    if config['ckt_storage'] == "bylayer":
        # Load MLP input (query)
        # Lookahead: For layer N predictor, use layer N-1 output as input
        # For layer 0, we rely on the embedding output (layer -1 effectively)
        # but in our data, mlp_x_0 is the input to layer 0, which is fine.
        # So we use input from previous layer idx
        # logic: input to predictor of layer k is output of layer k-1
        # in the recorded data, mlp_x_k is input to layer k.
        # So we want mlp_x_{k} for layer k?
        # WAIT. In DejaVu, we predict layer k sparsity using layer k-1 output.
        # mlp_x_k IS the input to layer k (which is roughly output of layern k-1 + attn k-1).
        # So using mlp_x_k seems correct for "current layer input"?
        #
        # Re-reading paper/logic:
        # "We predict layer l's sparsity using layer l-1's activation."
        # mlp_x_l is the INPUT to layer l.
        # So mlp_x_l IS the output of the previous block (residual stream).
        # So if we want to predict layer l sparsity, we have mlp_x_l available.
        #
        # HOWEVER, the User's "Previous Session Summary" EXPLICITLY stated:
        # "Fix lookahead training: use layer N-1 input to predict layer N MLP"
        # "Input (query) for Layer layer_idx is now loaded from mlp_x_{max(0, layer_idx - 1)}.mmap"
        #
        # Let's trust the User's explicit prior instruction/summary about "Lookahead".
        # If I am at layer L, I want to compute sparsity for layer L.
        # If I use `mlp_x_{L-1}`, I am using the input to layer L-1.
        # This means I am predicting Layer L sparsity using Layer L-2 output???
        #
        # Let's finding the logic.
        # To pipeline, we need to predict Layer L *before* we compute Layer L-1?
        # Or in parallel?
        # DejaVu uses asynchronous lookahead.
        # It predicts Layer L+1 while Layer L is computing?
        # If so, predicting Layer L needs input available BEFORE Layer L input is ready?
        # i.e. using Layer L-1 input?
        # Yes, using Input(L-1) to predict Sparsity(L).
        #
        # So: To train predictor for Layer L (Label L):
        # Input should be mlp_x_{L-1}.
        
        query_idx = max(0, layer_idx - 1)
        path = f"{DATA[args.model][args.dataset]}/mlp_x_{query_idx}.mmap"
        print(f"Reading query from {path} (Lookahead: using input from layer {query_idx} to predict layer {layer_idx})")
        query = np.array(np.memmap(
            path, dtype='float16', mode='r',
            shape=(config['N'], config['d'])
        ))
        
        # Load MLP activation label
        # For Llama, this is the output of SiLU(gate) * up, shape is intermediate_size
        path = f"{DATA[args.model][args.dataset]}/mlp_label_{layer_idx}.mmap"
        print(f"Reading MLP label from {path}")
        label = np.array(np.memmap(
            path, dtype='float16', mode='r',
            shape=(config['N'], config['intermediate'])
        ))
        
        return query, label


def generate_label_llama(y, threshold=0.0):
    """
    Generate binary labels for Llama MLP sparsity.
    
    Unlike OPT which uses ReLU (output > 0 is active),
    Llama uses SiLU which doesn't create strict zeros.
    
    We use a threshold to determine "active" neurons.
    This is a key difference and may affect predictor quality.
    """
    # Absolute value thresholding: neurons with significant magnitude are "active"
    one_hot = (torch.abs(y) > threshold).to(y.dtype)
    return one_hot


def evaluate_llama(model, device, loader, args, threshold=0.0, smalltest=False):
    """Evaluate predictor with Llama-specific label generation."""
    model.eval()
    
    eval_metrics = {
        "Loss": [],
        "Loss Weight": [],
        "Recall": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y = batch
            y = y.float().to(device)
            y = generate_label_llama(y, threshold)
            
            logits = model(x.to(device))
            probs = logits.sigmoid()
            preds = probs >= 0.5
            
            dif = y.int() - preds.int()
            miss = dif > 0.0  # classifier didn't activate target neuron
            
            weight = (y.sum() / y.numel()) + 0.005
            loss_weight = y * (1 - weight) + weight
            
            eval_metrics["Loss Weight"] += [weight.item()]
            eval_metrics["Loss"] += [
                torch.nn.functional.binary_cross_entropy(
                    probs, y, weight=loss_weight
                ).item()
            ]
            
            eval_metrics["Recall"] += [
                ((y.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item())
            ]
            eval_metrics["True Sparsity"] += [y.sum(dim=1).float().mean().item()]
            eval_metrics["Classifier Sparsity"] += [preds.sum(dim=1).float().mean().item()]
            
            if batch_idx >= 100 and smalltest:
                break
    
    for k, v in eval_metrics.items():
        eval_metrics[k] = np.array(v).mean()
    
    eval_metrics["Recall"] = eval_metrics["Recall"] / (eval_metrics["True Sparsity"] + 1e-8)
    return eval_metrics


def create_dataset(query, labels, args):
    """Create training and test dataloaders."""
    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)
    
    print(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    print(f"# training data: {num_train}, # test data: {num_test}")
    
    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)
    
    train_dataloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    
    return train_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description="Llama MLP Sparse Predictor Training")
    parser.add_argument("--model", type=str, default="llama-3b", choices=MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="c4", choices=DATA_CHOICES)
    parser.add_argument("--L", type=int, default=0, help="which layer")
    parser.add_argument("--D", type=int, default=1000, help="low rank dimension")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--threshold", type=float, default=0.0, 
                       help="Activation threshold for Llama SiLU outputs")
    args = parser.parse_args()
    
    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = CONFIG[args.model]
    
    print("=" * 40, f"Layer {args.L}", "=" * 40)
    print(f"Model: {args.model}")
    print(f"Hidden size: {config['d']}")
    print(f"Intermediate size: {config['intermediate']}")
    
    query, labels = get_data(args, args.L)
    train_loader, test_loader = create_dataset(query, labels, args)
    
    # Predictor: maps hidden_size → intermediate_size
    # Same architecture as OPT but different dimensions
    predictor = torch.nn.Sequential(
        torch.nn.Linear(config['d'], args.D, bias=None),
        torch.nn.Linear(args.D, config['intermediate'], bias=None),
    )
    
    print(f"\nPredictor architecture:")
    print(f"  Input: {config['d']} (hidden_size)")
    print(f"  Hidden: {args.D} (low rank)")
    print(f"  Output: {config['intermediate']} (intermediate_size)")
    
    print("\nStart Training")
    best_model, eval_result = train(
        predictor, train_loader, test_loader, args, device, verbal=True
    )
    
    # Save checkpoint
    import os
    checkpoint_dir = f"../checkpoint/llama-{args.model.replace('llama-', '')}-sparse-predictor"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    path = f"{checkpoint_dir}/{args.dataset}_layer{args.L}_recall-{eval_result['Recall']:.4f}-sparsity-{eval_result['Classifier Sparsity']:.0f}.pt"
    torch.save(best_model, path)
    print(f"\nSaved checkpoint to: {path}")
    
    # Print final results for comparison with OPT
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Compare with OPT to verify DejaVu hypothesis)")
    print("=" * 60)
    print(f"Layer {args.L}:")
    print(f"  Recall: {eval_result['Recall']:.4f}")
    print(f"  True Sparsity: {eval_result['True Sparsity']:.0f}")
    print(f"  Classifier Sparsity: {eval_result['Classifier Sparsity']:.0f}")
    print(f"  Loss: {eval_result['Loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
